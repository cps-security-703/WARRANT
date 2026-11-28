"""Co-simulation master: one Python loop, two clocks.

Per fine period delta (steps not yet built are marked with their phase):
  1. transport   advance SUMO; low-SOC triggers, hub occupancy, queues, spillback
  2. bridge in   new occupancies -> ACN-Sim plug-in events
  3. derating    apply the latest d_h to each hub's aggregate limit
  4. charging    step ACN-Sim one period
  5. shields     PCC limiter / hub admission                       (Phase 4)
                 (a tripped hub's open breaker acts on the same physical path)
  6. grid        PCC meter -> hub loads; power flow; true state, reported telemetry, next d_h;
                 transformer protection; IDS observes charging-side signals (detection only)
  7. bridge out  release finished sessions, write SOC back to SUMO
  8. attacks     scripted attackers act on their injection point for the next step
  9. log
Every contract period Delta: aggregate contract signals, publish a_h to routing, log.

The grid is optional: a config without a `grid` section (Phase 0) runs at fixed hub limits.
"""
from pathlib import Path

import numpy as np

from seams.attacks import make_attacks
from seams.baselines.ids import make_ids
from seams.baselines.oracle import Oracle
from seams.baselines.protection import apply_baseline
from seams.contracts.monitors import SeamMonitor
from seams.contracts.params import Theta, load_theta
from seams.contracts.renegotiate import Renegotiator
from seams.contracts.signals import aggregate
from seams.contracts.verifier import v1_compatibility, v2_refinement
from seams.shields.hub_admission import HubAdmission
from seams.shields.pcc_limiter import PccLimiter
from seams.cosim.clock import Clock
from seams.cosim.config import resolve_path
from seams.cosim.logger import RunLogger
from seams.domains.bridge.handoff import Bridge
from seams.domains.charging.acn_env import AcnHub
from seams.domains.grid.derating import DeratingController
from seams.domains.grid.dss_env import HOME, GridAdapter
from seams.domains.grid.loadshape import BackgroundLoad
from seams.domains.grid.pcc_meter import PccMeter
from seams.domains.grid.protection import TransformerProtection
from seams.domains.grid.telemetry import Telemetry
from seams.domains.transport.fleet import Fleet
from seams.domains.transport.routing import HubView
from seams.domains.transport.sumo_env import SumoAdapter
from seams.scenarios.build_scenario import write_run_files


class Master:
    def __init__(self, cfg, run_id, run_dir=None):
        cfg = apply_baseline(cfg)
        self.cfg = cfg
        self.run_id = run_id
        self.seed = int(cfg["run"]["seed"])
        # independent seeded streams, so adding a consumer never shifts another's draws
        # (spawn(n) children do not depend on n, so a new stream never shifts the others)
        trips_rng, load_rng, self.attack_rng, telemetry_rng, home_rng, self.av_rng = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(6)
        )
        self.clock = Clock.from_config(cfg)
        self.run_dir = Path(run_dir) if run_dir else resolve_path(cfg["run"]["out_dir"]) / run_id
        self.logger = RunLogger(self.run_dir, run_id, cfg)
        sumocfg = write_run_files(cfg, self.run_dir / "sumo", trips_rng)
        self.sumo = SumoAdapter(cfg, sumocfg, self.seed)
        self.hubs = {
            h["id"]: AcnHub(h["id"], h["n_evse"], {**cfg["evse"], **h.get("evse", {})}, self.clock.acn_period_min)
            for h in cfg["hubs"]
        }
        self.bridge = Bridge(cfg, self.hubs, self.clock)
        self.fleet = None

        self.grid = None
        self.home = None
        if "grid" in cfg:
            self.grid = GridAdapter(cfg)
            self.background = BackgroundLoad(cfg, load_rng)
            self.home = None
            if cfg["grid"].get("home_ev", {}).get("enabled"):
                from seams.domains.charging.home import HomeCharging
                self.home = HomeCharging(cfg, self.grid.home_elements(), home_rng)
                self.grid.set_home_weights(self.home.charger_counts)
            self.telemetry = Telemetry(self.hubs, cfg, telemetry_rng)
            if self.background.day:
                self.logger.log("events", t_s=0.0, k=0, event=f"background_day:{self.background.day}")
            self.pcc = PccMeter(self.hubs)
            self.derating = DeratingController(cfg)
            self.derating_on = cfg["grid"]["derating"].get("enabled", True)
            self.protection = TransformerProtection(cfg, self.clock.fine_period_s) if "protection" in cfg["grid"] else None
            self.phi = cfg["grid"]["phi_safe"]
            self.d_kw = {h: self.derating.p_full[h] for h in self.hubs}
        self.ids = make_ids(cfg)
        self._ids_alarmed = set()
        self.response = None  # detect-and-respond baseline (evaluation only)
        if self.grid and cfg.get("ids_response", {}).get("enabled"):
            from seams.baselines.bdd import AlarmResponse
            self.response = AlarmResponse(cfg, list(self.hubs), self.clock.fine_period_s)
        self.oracle = Oracle(cfg, self.hubs, home=self.home is not None) if cfg.get("baseline") == "oracle" else None
        self.monitor = None
        self.theta = None
        if self.grid and cfg.get("contracts", {}).get("monitor"):
            c = cfg["contracts"]
            self.theta = load_theta(cfg)
            self.monitor = SeamMonitor(self.theta, list(self.hubs), self.grid.regulators, c["windows"], c["tol"],
                                       parties=list(self.hubs) + ([HOME] if self.home is not None else []))

        # the home portfolio's power range for the large-signal sensitivity: its static P_bar,
        # or every charger at full rate before a contract exists
        if getattr(self, "home", None) is not None:
            hc = cfg["grid"]["home_ev"]
            self.home_p_max = (float(self.theta.P_bar[HOME]) if self.theta is not None and HOME in self.theta.P_bar
                               else float(hc["chargers"]) * float(hc.get("max_kw", 7.2)))

        # Shields (trusted layer), enforcing the theta in force
        sh = cfg.get("shields", {})
        self.limiters = {}
        if sh.get("pcc_limiter") and self.theta is not None:
            for h, hub in self.hubs.items():
                self.limiters[h] = PccLimiter(h, hub.voltage_v)
                self.limiters[h].set_limit(self.theta.P_bar[h])
        self.admission = None
        if sh.get("hub_admission") and self.theta is not None:
            self.admission = HubAdmission(list(self.hubs), cfg["contracts"]["windows"])
            self.admission.set_quota(self.theta.lam_bar)

        # Verifier: V1 once per theta, V2 at every contract boundary
        self.verifier = cfg.get("verifier") if self.theta is not None else None
        if self.verifier:
            self.v1 = v1_compatibility(self.theta, cfg["contracts"]["windows"], self.verifier["horizon_periods"])
            self.certified = None  # verdict at the last boundary

        # Renegotiation (Phase 5): the renegotiator owns the theta in force
        self.reneg = None
        self._published = None  # last (S, T, Sq) the grid published
        self.history = {}  # contract period -> (signals, Sq), for distrust anchors
        if self.verifier and cfg.get("renegotiation", {}).get("enabled"):
            safe = Theta.load(resolve_path(cfg["renegotiation"]["theta_safe_file"]))
            self.reneg = Renegotiator(cfg, self.theta, safe)

        # Interface bus: what the charging domain published at the last contract boundary.
        self.published = {h: {"a": len(hub.station_ids), "queue": 0} for h, hub in self.hubs.items()}
        # Injection points for the attack library: hook(views, vid) -> views
        self.view_hooks = []
        self.attacks = make_attacks(cfg, self)
        self._period = []  # fine-step records of the current contract period

    # --- interfaces ---------------------------------------------------------
    def hub_views(self, vid=None):
        """What a vehicle's routing believes about each hub: the published a_h, after any
        attack hooks (vid=None is the unattacked view)."""
        views = {
            h: HubView(availability=self.published[h]["a"], queue=self.published[h]["queue"],
                       n_evse=len(hub.station_ids))
            for h, hub in self.hubs.items()
        }
        if self.oracle is None:  # the oracle's routing never sees spoofed inputs
            for hook in self.view_hooks:
                views = hook(views, vid)
        return views

    def apply_theta(self, theta):
        """Switch the contract in force: shields, monitors and the honest charging limit."""
        self.theta = theta
        for h, lim in self.limiters.items():
            lim.set_limit(theta.P_bar[h])
        if self.admission:
            self.admission.set_quota(theta.lam_bar)
        if self.monitor:
            self.monitor.set_theta(theta)

    def _log_negotiation(self, t_s, rec):
        """One row per negotiation (RQ4 metrics) and the full transcript in negotiations.jsonl."""
        import json
        u = self.reneg.llm.usage
        row = {k: rec[k] for k in ("kc", "trigger", "default_status", "agreement", "timeout", "chosen", "chosen_source",
                                   "rounds", "n_candidates", "n_certified", "n_malformed", "latency_s", "verify_ms",
                                   "graph_wall_s")}
        row["skipped"] = rec.get("skipped")
        row.update({f"default_cost_{r}": c for r, c in rec["default_cost"].items()})
        row.update({f"chosen_cost_{r}": c for r, c in (rec["chosen_cost"] or {}).items()})
        self.logger.log("negotiation", t_s=t_s, llm_calls=u["calls"], llm_api_calls=u["api_calls"],
                        llm_cost_usd=u["cost_usd"], **row)
        with open(self.logger.run_dir / "negotiations.jsonl", "a") as f:
            f.write(json.dumps({"t_s": t_s, **rec}, default=str) + "\n")

    def offline(self, hub, k):
        return self.grid is not None and self.protection is not None and self.protection.is_offline(hub, k)

    # --- loop ---------------------------------------------------------------
    def run(self):
        self.start()
        try:
            self.advance(self.clock.fine_steps(self.cfg["run"]["horizon_s"]))
        finally:
            self.sumo.close()
            self.finish()
        return self.run_dir

    def start(self):
        """Start SUMO; the loop then runs in advance() calls (run(), or an adaptive attacker's
        environment that steps the co-simulation period by period)."""
        self.sumo.start()
        self.fleet = Fleet(self.cfg, self.sumo, self.admission)
        self.av_fleet = None
        if (self.cfg.get("av_fleet") or {}).get("n"):
            from seams.domains.transport.av import AvFleet
            self.av_fleet = AvFleet(self.cfg, self.sumo, self.av_rng)
        self.k = 0  # next fine step
        if self.verifier:
            self.logger.log("events", t_s=0.0, k=0, event=f"V1:{'ok' if self.v1.ok else 'fail'}",
                            hub=self.v1.check, vehicle_id=None)

    def advance(self, n_fine):
        for _ in range(n_fine):
            self.fine_step(self.k)
            if self.clock.is_contract_boundary(self.k):
                self.contract_step(self.k)
            self.k += 1

    def fine_step(self, k):
        t0 = k * self.clock.fine_period_s  # start of this fine period
        attacking = {a.target: a.origin for a in self.attacks if a.active}  # set at step 8 of k-1

        # 1. transport
        self.sumo.advance(self.clock.sumo_steps_per_fine)
        t_s = self.sumo.time()
        if self.av_fleet is not None:
            self.av_fleet.step(t_s, self.fleet)
        ts = self.fleet.step(t_s, self.hub_views)

        # 2. bridge in
        for vid, hub in ts.newly_parked:
            v = self.fleet.vehicles[vid]
            self.bridge.plug_in(vid, hub, k, self.sumo.charge_wh(vid), v.capacity_wh, v.trip_type,
                                v.dwell_s, v.request_kwh)
            self.logger.log("events", t_s=t_s, k=k, event="plugin", hub=hub, vehicle_id=vid)

        # 3. derating (grid -> charging), from the previous grid solve; an honest charging
        #    operator also keeps to its contracted P_bar (the limiter only catches a dishonest one)
        if self.grid and self.derating_on:
            for h, hub in self.hubs.items():
                allow = self.d_kw[h]
                if self.theta is not None:
                    allow = min(allow, float(self.theta.P_bar[h]))
                if self.oracle is not None:
                    allow = min(allow, self.oracle.limit[h])
                hub.set_aggregate_limit(allow * 1000.0 / hub.voltage_v)

        # 4-5. charging; on the physical path: PCC limiter shield, then a tripped hub's open breaker
        for h, hub in self.hubs.items():
            filters = [self.limiters[h]] if h in self.limiters else []
            if self.oracle is not None:
                filters.append(self.oracle.limiters[h])
            if self.offline(h, k):
                filters.append(hub.open_breaker)
            hub.step(schedule_filters=filters)

        # 6. grid
        grid_rec = {h: {} for h in self.hubs}
        if self.grid:
            p_pcc = self.pcc.measure(self.hubs)
            bg = self.background.step(t0)
            self.grid.set_background(bg)
            self._bg_max = max(getattr(self, "_bg_max", 0.0), bg)  # within the contract period
            home_kw = home_req = None
            if self.home is not None:
                # an honest aggregator keeps its home portfolio within the contracted P_bar; the
                # trusted demand-response cap (shields.home_cap, a utility-signed OpenADR limit
                # the chargers obey) binds a dishonest one as well
                comply = cap = None
                if self.theta is not None and HOME in self.theta.P_bar:
                    comply = float(self.theta.P_bar[HOME])
                    if self.cfg.get("shields", {}).get("home_cap"):
                        cap = comply
                if self.oracle is not None:  # the oracle dispatches the portfolio directly
                    cap = min(cap if cap is not None else float("inf"), self.oracle.limit[HOME])
                self.grid.set_home_power(self.home.step(t0, comply, cap))
                home_kw, home_req = self.home.total_kw(), self.home.requested_kw
            for h, p in p_pcc.items():
                self.grid.set_hub_power(h, p)
            state = self.grid.solve()
            v_rep = self.telemetry.report(state.v_lv)
            v_mv_rep = self.telemetry.report_mv(state.v_mv)
            # derating input: reported voltage, or the trusted estimate for distrusted hubs
            v_in = dict(state.v_lv) if self.oracle is not None else dict(v_rep)
            for h in (self.reneg.distrusted() if self.reneg else ()):
                v_in[h] = self.reneg.v_est(h, p_pcc, v_mv_rep)
            for h in (self.response.active() if self.response else ()):
                v_in[h] = self.response.estimate(h, p_pcc, v_mv_rep)
            if self.derating_on:
                self.d_kw = self.derating.allowances(v_in)  # applied at the next fine step
            taps = ",".join(map(str, state.taps))
            for h in self.hubs:
                v = state.v_lv[h]
                grid_rec[h] = dict(
                    P_pcc=p_pcc[h], V_true=v, V_mv=state.v_mv[h], V_reported=v_rep[h], V_mv_reported=v_mv_rep[h],
                    V_derating=v_in[h],
                    xfmr_loading=state.xfmr_loading[h], d_h=self.d_kw[h],
                    margin_v=min(v - self.phi["v_min"], self.phi["v_max"] - v),
                    margin_loading=self.phi["loading_max"] - state.xfmr_loading[h],
                    bg_mult=bg, home_kw=home_kw, home_req_kw=home_req, taps=taps, converged=state.converged, offline=self.offline(h, k),
                )
            if self.oracle is not None:
                p_now = dict(p_pcc, **({HOME: home_kw} if self.home is not None else {}))
                self.oracle.update(k, self.grid, state, p_now)
            if self.protection:
                for h in self.protection.update(k, state.xfmr_loading, self.background.ambient_c(t0)):
                    self.logger.log("events", t_s=t_s, k=k, event="trip", hub=h, vehicle_id=None)
                for h, hot in getattr(self.protection, "hotspot", {}).items():
                    grid_rec[h]["hotspot_c"] = hot
        if self.ids and self.grid:
            ctx = {h: {"hub": hub, "v_true": grid_rec[h]["V_true"], "v_rep": grid_rec[h]["V_reported"],
                       "v_mv_rep": grid_rec[h]["V_mv_reported"], "p": grid_rec[h]["P_pcc"],
                       "queue": ts.queue[h]} for h, hub in self.hubs.items()}
            for h, (alarm, score) in self.ids.observe(t_s, ctx).items():
                grid_rec[h].update(ids_alarm=alarm, ids_score=score)
                if alarm and h not in self._ids_alarmed:  # rising edge = one alarm episode
                    self.logger.log("events", t_s=t_s, k=k, event="ids_alarm", hub=h, vehicle_id=None)
                for name, alarms in getattr(self.ids, "per_detector", {}).items():
                    if alarms.get(h) and (name, h) not in self._ids_alarmed:
                        self.logger.log("events", t_s=t_s, k=k, event=f"ids_alarm:{name}", hub=h, vehicle_id=None)
                    self._ids_alarmed.add((name, h)) if alarms.get(h) else self._ids_alarmed.discard((name, h))
                if alarm:
                    self._ids_alarmed.add(h)
                else:
                    self._ids_alarmed.discard(h)
            if self.response:
                alarms = getattr(self.ids, "per_detector", {}).get(self.response.detector, {})
                rec = {h: grid_rec[h] for h in self.hubs}
                for event, h in self.response.update(
                        alarms, {h: r["V_reported"] for h, r in rec.items()},
                        {h: r["V_mv_reported"] for h, r in rec.items()}, {h: r["P_pcc"] for h, r in rec.items()}):
                    self.logger.log("events", t_s=t_s, k=k, event=event, hub=h, vehicle_id=None)

        # 7. bridge out
        for session, reason in self.bridge.due_releases():
            vid = session.vehicle_id
            self.bridge.release(session, reason)
            session.charge_before_release_wh = self.sumo.charge_wh(vid)
            self.sumo.set_charge_wh(vid, self.bridge.new_charge_wh(session))
            session.charge_after_release_wh = self.sumo.charge_wh(vid)
            self.fleet.release(vid, t_s)
            self.logger.log("events", t_s=t_s, k=k, event=f"release:{reason}", hub=session.hub, vehicle_id=vid)

        # 8. attacks act on their injection points for the next fine period
        t_next = (k + 1) * self.clock.fine_period_s
        for attack in self.attacks:
            was = attack.active
            attack.step(t_next)
            if attack.active != was:
                self.logger.log("events", t_s=t_next, k=k + 1,
                                event=f"attack_{'onset' if attack.active else 'offset'}:{attack.origin}",
                                hub=attack.target, vehicle_id=None)

        # 9. log
        for h, hub in self.hubs.items():
            rec = dict(
                t_fine=k, hub=h, arrivals=ts.arrivals[h], queue=ts.queue[h],
                spillback_m=ts.spillback_m[h], occupied_evse=len(self.bridge.occupancy[h]),
                P_sched=hub.power_kw(), P_req=hub.last_requested_kw, limit_a=hub.limit_a, **grid_rec[h],
                attack_active=attacking.get(h), refused=ts.refused[h],
                limiter_active=bool(self.limiters[h].active) if h in self.limiters else False,
            )
            self.logger.log("fine", **rec)
            self._period.append(rec)

    def contract_step(self, k):
        """Aggregate the period's fine records into contract signals, run the monitors and the
        attribution rule, have the grid publish S and T, and publish a_h to routing."""
        kc = self.clock.contract_index(k)
        mon = {}
        if self.monitor:
            sig = aggregate(kc, self._period, list(self.hubs), float(self.cfg["evse"]["voltage_v"]),
                            home=self.home is not None)
            rows, blames, first, resid = self.monitor.update(sig)
            # the grid publishes S, T (and Sq) at its own cadence, independent of Delta
            c = self.cfg["contracts"]
            every = float(c.get("sensitivity_every_s", self.clock.contract_period_s))
            t_end = (k + 1) * self.clock.fine_period_s
            if self._published is None or t_end % every < 1e-9:
                S, T = self.grid.sensitivity(float(c["sensitivity_dp_kw"]))
                Sq = None
                if c.get("epsilon"):  # Phase 4+: large-signal model for V2
                    p_max = {h: self.derating.p_full[h] for h in self.hubs}
                    if self.home is not None:
                        p_max[HOME] = self.home_p_max
                    Sq = self.grid.sensitivity_qss(p_max)
                self._published = (S, T, Sq)
            S, T, Sq = self._published
            self.monitor.publish(S, T)
            t_s = (k + 1) * self.clock.fine_period_s
            for d, h in blames:
                self.logger.log("events", t_s=t_s, k=k, event=f"blame:{d}", hub=h, vehicle_id=None)
            for d, h in first or []:
                self.logger.log("events", t_s=t_s, k=k, event=f"incident:{d}", hub=h, vehicle_id=None)
            cert = {}
            self.history[kc] = (sig, Sq)
            keep = self.reneg.K_ext + 2 if self.reneg else 2
            for old in [p for p in self.history if p < kc - keep]:
                del self.history[old]
            if self.reneg:
                prev = self.theta.version
                decision, reneg_events = self.reneg.step(kc, sig, Sq, blames, self.history)
                if decision.status == "fallback" and decision.certified:
                    # theta_safe's certificate holds up to its background scope only
                    scope = self.reneg.safe.meta.get("safe_certificate", {}).get("scope_bg_max")
                    if scope is not None and self._bg_max > scope:
                        decision.certified = False
                        decision.reason += "; background above theta_safe's certified scope"
                for name, h in reneg_events:
                    self.logger.log("events", t_s=t_s, k=k, event=name, hub=h, vehicle_id=None)
                for rec in self.reneg.pop_negotiations():
                    self._log_negotiation(t_s, rec)
                self.apply_theta(decision.theta)  # atomic switch for the next period
                if decision.theta.version != prev:
                    self.logger.log("events", t_s=t_s, k=k, event=f"theta:{decision.status}",
                                    hub=decision.theta.version, vehicle_id=None)
                cert = dict(certified=decision.certified, v2_ms=decision.solve_ms, status=decision.status,
                            reason=decision.reason, modes=",".join(sorted(decision.modes)),
                            v2_cex=str(decision.cex) if decision.cex else None,
                            **{f"Pbar_{j}": float(decision.theta.P_bar[j]) for j in decision.theta.P_bar})
            elif self.verifier:
                anchor = {"V": sig.V_rep, "P": sig.P, "S": Sq}
                if self.theta.eps is None:
                    raise RuntimeError("theta has no epsilon section; run seams.experiments.calibrate")
                self.certified = v2_refinement(
                    self.theta, anchor, self.theta.eps, self.cfg["grid"]["phi_safe"],
                    self.verifier["horizon_periods"], self.verifier.get("use_epsilon", True),
                )
                cert = dict(certified=self.certified.ok and self.v1.ok, v2_ms=self.certified.solve_ms,
                            v2_cex=str(self.certified.counterexample) if self.certified.counterexample else None)
            for h in self.hubs:
                mon[h] = {f"rho_{n}": float(v) for n, v in rows[h].items()}
                mon[h].update({f"resid_{w}": float(resid[(h, w)]) for w in self.monitor.windows})
                mon[h].update({f"S_{j}": S[h][j] for j in S[h]})
                if Sq is not None:
                    mon[h].update({f"Sq_{j}": Sq[h][j] for j in Sq[h]})
                mon[h].update(
                    lam_sig=sig.lam[h], P_req_h=sig.P_req[h], S_hh=S[h][h],
                    blamed=",".join(sorted(d for d, hh in blames if hh == h)),
                    theta=self.monitor.theta_version, **cert,
                )
            if HOME in rows:  # the aggregator's home portfolio: its own row of the power seam
                self.logger.log("contract", k=kc, hub=HOME, P_h=sig.P[HOME],
                                **{f"rho_{n}": float(v) for n, v in rows[HOME].items()},
                                P_req_h=sig.P_req[HOME], theta=self.monitor.theta_version,
                                blamed=",".join(sorted(d for d, hh in blames if hh == HOME)), **cert)
        self._bg_max = 0.0
        if self.admission:
            self.admission.roll()
        for h in self.hubs:
            recs = [r for r in self._period if r["hub"] == h]
            queue = recs[-1]["queue"]
            if self.offline(h, k + 1):
                a_h = -len(self.hubs[h].station_ids) - queue  # charging operator knows it is down
            else:
                a_h = self.bridge.free_evses(h) - queue  # free EVSEs minus queue pressure
            self.published[h] = {"a": a_h, "queue": queue}
            row = dict(
                k=kc, hub=h, lambda_h=sum(r["arrivals"] for r in recs),
                P_h=float(np.mean([r.get("P_pcc", r["P_sched"]) for r in recs])),
                a_h=a_h, queue=queue,
            )
            if self.grid:
                row.update(
                    V_rep_h=float(np.mean([r["V_reported"] for r in recs])),
                    d_h=float(np.mean([r["d_h"] for r in recs])),
                )
            self.logger.log("contract", **row, **mon.get(h, {}))
        self._period = []

    def finish(self):
        for s in self.bridge.active.values():  # still plugged at the horizon: energy so far
            if s.vehicle_id in self.hubs[s.hub].evs:
                s.delivered_kwh = self.hubs[s.hub].delivered_kwh(s.vehicle_id)
        for s in self.bridge.finished + list(self.bridge.active.values()):
            v = self.fleet.vehicles.get(s.vehicle_id) if self.fleet else None
            self.logger.log(
                "session", vehicle_id=s.vehicle_id, hub=s.hub, station_id=s.station_id,
                trip_type=s.trip_type, arrival=s.arrival_k, departure=s.departure_k,
                requested_kWh=s.requested_kwh, delivered_kWh=s.delivered_kwh,
                unserved_kWh=s.unserved_kwh, released_k=s.released_k,
                released_reason=s.released_reason or "active_at_end",
                charge_at_plugin_wh=s.charge_at_plugin_wh,
                charge_before_release_wh=s.charge_before_release_wh,
                charge_after_release_wh=s.charge_after_release_wh,
                vehicle_arrived=bool(v and v.state.value == "arrived"),
            )
        for vid, v in sorted((self.fleet.vehicles if self.fleet else {}).items()):
            self.logger.log(
                "trips", vehicle_id=vid, trip_type=v.trip_type, depart_s=v.depart_s, arrive_s=v.arrive_s,
                charged=v.visits > 0, stranded=v.stranded, refusals=v.refusals,
            )
        if self.home is not None:  # unserved home energy counts as availability cost too
            horizon = float(self.cfg["run"]["horizon_s"])
            for s in self.home.sessions:
                self.logger.log("home", element=s.element, arr_s=s.arr_s, dep_s=s.dep_s,
                                requested_kWh=s.kwh, delivered_kWh=s.delivered,
                                unserved_kWh=s.remaining(), ends_in_horizon=s.dep_s <= horizon)
        self.logger.close()
