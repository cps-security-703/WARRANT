import json
import time
from dataclasses import dataclass, field
from fractions import Fraction

import z3

from seams.contracts.params import outward
from seams.cosim.config import resolve_path
from seams.contracts.verifier import Q, v1_compatibility, v2_refinement


@dataclass
class Decision:
    theta: object
    status: str             # static | renegotiated | fallback
    reason: str
    solve_ms: float
    certified: bool
    modes: dict = field(default_factory=dict)
    cex: dict = None


def optimize_P_bar(theta, anchor, eps, phi, K, caps, floors, extra, weights):
    t0 = time.perf_counter()
    hubs = sorted(anchor["V"])     # voltage-constrained
    parties = sorted(anchor["P"])  # power parties (hubs, HOME)
    opt = z3.Optimize()
    pb = {j: z3.Real(f"pb_{j}") for j in parties}
    for j in parties:
        cap = Fraction(caps[j])
        if j in eps["P_load_max"]:
            cap = min(cap, eps["P_load_max"][j])
        opt.add(pb[j] >= Q(floors[j]), pb[j] <= Q(cap))
    v_min = Q(phi["v_min"])
    for h in hubs:
        for k in range(1, K + 1):
            bound = eps["D"][h][k] + eps["lin"][h] + Fraction(extra.get(h, {}).get(k, 0))
            base = Q(Fraction(anchor["V"][h])) - Q(bound)
            # worst case over P_j in [0, pb_j]: a linear function is minimized at a box corner
            # the worst corner is P_j = pb_j where S_hj < 0 and P_j = 0 where S_hj >= 0, so
            # one constraint per (h, k) replaces the 2^n corners
            terms = [
                Q(Fraction(anchor["S"][h][j])) * ((pb[j] if anchor["S"][h][j] < 0 else Q(0)) - Q(Fraction(anchor["P"][j])))
                for j in parties
            ]
            opt.add(base + z3.Sum(terms) >= v_min)
    opt.maximize(z3.Sum([Q(Fraction(weights.get(j, 1))) * pb[j] for j in parties]))
    if opt.check() != z3.sat:
        return None, 1000.0 * (time.perf_counter() - t0)
    m = opt.model()
    out = {j: outward(Fraction(str(m.eval(pb[j]).as_fraction())), "down") for j in parties}
    return out, 1000.0 * (time.perf_counter() - t0)


def laxity_weights(cfg):
    import numpy as np
    import pandas as pd
    from seams.scenarios.build_scenario import PROCESSED
    pool = pd.read_parquet(PROCESSED / "sessions.parquet")
    evse_kw = float(cfg["evse"]["max_rate_a"]) * float(cfg["evse"]["voltage_v"]) / 1000.0
    rate = pool.rate_kw.fillna(evse_kw).clip(lower=0.1)
    lax = ((pool.dwell_s / 3600.0) - pool.kwh / rate).clip(lower=0.0)
    med = lax.groupby(pool.profile).median()
    parties = {h["id"]: h["profile"] for h in cfg["hubs"]}
    if (cfg["grid"].get("home_ev") or {}).get("enabled"):
        parties["HOME"] = "home"
    return {j: float(1.0 / (1.0 + med[prof])) for j, prof in parties.items() if not np.isnan(med.get(prof, np.nan))}


class Renegotiator:
    def __init__(self, cfg, theta_static, theta_safe):
        r = cfg["renegotiation"]
        self.cfg = cfg
        self.static, self.safe = theta_static, theta_safe
        self.eps = theta_static.eps
        # RQ3's "without eps" variant (use_epsilon: false) drops both terms: certificates
        # (optimizer and V2) take the anchor's linear prediction as exact. drop_eps drops one,
        # D (drift) or lin (linearization), for the cost decomposition (evaluation only)
        drop = set(cfg["verifier"].get("drop_eps") or ())
        if not cfg["verifier"].get("use_epsilon", True):
            drop = {"D", "lin"}
        if drop - {"D", "lin"}:
            raise ValueError(f"verifier.drop_eps: {sorted(drop)!r}")
        if "D" in drop:
            self.eps = dict(self.eps, D={h: {k: Fraction(0) for k in d} for h, d in self.eps["D"].items()})
        if "lin" in drop:
            self.eps = dict(self.eps, lin={h: Fraction(0) for h in self.eps["lin"]})
        self.phi = cfg["grid"]["phi_safe"]
        self.windows = cfg["contracts"]["windows"]
        self.K = int(cfg["verifier"]["horizon_periods"])
        self.K_ext = max(self.eps["D"][next(iter(self.eps["D"]))])
        self.quiet_n = int(r["exit_quiet_periods"])
        self.cap_frac = Fraction(str(r["charging_cap_frac"]))
        self.weights = r.get("weights") or {}
        if self.weights == "laxity":
            self.weights = laxity_weights(cfg)
        self.budget_ms = float(r["time_budget_ms"])
        self.on_decert = bool(r.get("renegotiate_on_decertification", True))
        self.attribution = r.get("attribution", "blame")
        if self.attribution not in ("blame", "blind"):
            raise ValueError(f"renegotiation.attribution: {self.attribution!r}")
        self.modes = {}      # (domain, hub) -> {"quiet": n, ...}
        self.theta = theta_static
        self.v1_static = v1_compatibility(theta_static, self.windows, self.K)
        self.n_versions = 0
        self.estimator = r.get("distrust_estimator", "anchor")
        self.last_anchor = {}  # hub -> anchor info of its last distrust episode ("anchor")
        if self.estimator in ("mv", "both"):
            with open(resolve_path(cfg["ids"]["bdd_file"])) as f:
                self.mv = json.load(f)["hubs"]
            missing = [h for h, p in self.mv.items() if "est_err_max" not in p]
            if missing:
                raise RuntimeError(f"{cfg['ids']['bdd_file']} has no est_err_max for {missing}; "
                                   "re-run seams.experiments.calibrate")
        elif self.estimator not in ("anchor", "both"):
            raise ValueError(f"renegotiation.distrust_estimator: {self.estimator!r}")
        # Phase 8: operator agents negotiate on top of the classical path
        self.negotiator, self.negotiations, self.plan, self.n_plans = None, [], None, 0
        self.last_status, self.last_negotiation, self._ctx = None, -10 ** 9, None
        a = cfg.get("agents") or {}
        if a.get("enabled"):
            from seams.agents.llm import LLM
            from seams.agents.operator_agent import OperatorAgent
            from seams.agents.protocol import Negotiation
            llm = LLM(a["backbones"][a["backbone"]], resolve_path(a["cache"]), a.get("mode", "record"))
            byz = a.get("byzantine") or {}  # RQ5: {role, strategy, misreport_scale}
            if byz and byz.get("role") not in a["roles"]:
                raise ValueError(f"agents.byzantine.role: {byz.get('role')!r}")
            self.negotiator = Negotiation([OperatorAgent(r, llm, cfg, byz if r == byz.get("role") else None)
                                           for r in a["roles"]], cfg)
            self.cooldown = int(a["cooldown_periods"])
            self.min_valid = int(a.get("min_validity_leads", self.K))
            self.plan_mode = a.get("plan_mode", "lock")
            if self.plan_mode not in ("lock", "floor"):
                raise ValueError(f"agents.plan_mode: {self.plan_mode!r}")
            self.llm = llm

    # --- degraded modes ---------------------------------------------------------
    def _update_modes(self, kc, sig, blames, history, events):
        blamed = set(blames)
        if blamed and self.attribution == "blind":
            blamed = {(d, h) for d in ("transport", "grid") for h in sig.V_rep}
            blamed |= {("charging", j) for j in self.static.P_bar}
        for d, h in blamed:
            if (d, h) not in self.modes:
                info = {"quiet": 0, "consistent": 0, "since": kc}
                if d == "grid" and self.estimator in ("anchor", "both"):
                    prev = self.last_anchor.get(h)
                    if prev is not None and kc - prev["anchor_k"] + self.K <= self.K_ext:
                        info.update(prev)  # caught before: keep the last clean anchor
                    else:
                        back = kc - max(self.windows) - 1  # before the longest physics window
                        if back not in history:
                            back = min(history)
                        a_sig, a_Sq = history[back]
                        info.update(anchor_k=back, V0=a_sig.V_rep[h], P0=dict(a_sig.P), Sq=a_Sq[h])
                        self.last_anchor[h] = {k: info[k] for k in ("anchor_k", "V0", "P0", "Sq")}
                self.modes[(d, h)] = info
                events.append((f"mode_enter:{d}", h))
        for (d, h), info in list(self.modes.items()):
            info["quiet"] = 0 if (d, h) in blamed else info["quiet"] + 1
            done = info["quiet"] >= self.quiet_n
            if d == "grid":
                # V_rep is plausible again once it agrees with the trusted estimate within the
                # estimate's own uncertainty (drift since the anchor + linearization), with the
                # drift capped at the V2 horizon: an uncapped D(elapsed) keeps growing until it
                # exceeds a still-active falsification, which would then be re-trusted. A
                # falsification within D(K) + eps_lin can still be: the residual attack surface.
                tol = self.v_est_err(h, kc)
                ok = abs(sig.V_rep[h] - self.v_est(h, sig.P, sig.V_mv_rep)) <= tol
                if self.estimator == "both":
                    p = self.mv[h]
                    ok = ok and abs(sig.V_rep[h] - (p["a"] + p["b"] * sig.V_mv_rep[h] + p["c"] * sig.P[h])) \
                        <= p["est_err_max"]
                info["consistent"] = info["consistent"] + 1 if ok else 0
                done = info["consistent"] >= self.quiet_n
                if self.attribution == "blind":  # blind: also quiet, so untouched hubs don't flap
                    done = done and info["quiet"] >= self.quiet_n
            if done:
                del self.modes[(d, h)]
                events.append((f"mode_exit:{d}", h))

    def distrusted(self):
        return {h for d, h in self.modes if d == "grid"}

    def v_est(self, h, P, V_mv=None):
        if self.estimator == "mv":
            p = self.mv[h]
            return p["a"] + p["b"] * V_mv[h] + p["c"] * P[h]
        m = self.modes[("grid", h)]
        return m["V0"] + sum(m["Sq"][j] * (P[j] - m["P0"][j]) for j in P)

    def v_est_err(self, h, kc):
        if self.estimator == "mv":
            return float(self.mv[h]["est_err_max"])
        elapsed = min(kc - self.modes[("grid", h)]["anchor_k"], self.K)
        return float(self.eps["D"][h][max(1, elapsed)] + self.eps["lin"][h])

    # --- one boundary -------------------------------------------------------------
    def step(self, kc, sig, Sq, blames, history):
        self._ctx = None
        decision, events = self._classical(kc, sig, Sq, blames, history)
        if self.negotiator is not None:
            decision = self._agents(kc, decision, events, history)
        self.last_status = decision.status
        return decision, events

    def _context(self, reason):
        from seams.agents.gate import Context
        c = self._ctx
        base = self.static.replace("deg", P_bar=dict(c["caps"]), P_assumed=dict(c["caps"]))
        return Context(kc=c["kc"], base=base, safe=self.safe, anchor=c["anchor"], eps=self.eps, phi=self.phi,
                       K=self.K, windows=self.windows, extra=c["extra"], caps=c["caps"], floors=c["floors"],
                       reason=reason, modes=tuple(sorted(f"{d}:{h}" for d, h in self.modes)),
                       min_valid=self.min_valid)

    def _agents(self, kc, decision, events, history):
        import dataclasses
        from seams.agents import gate
        if self._ctx is None:
            self.plan = None
            return decision
        entered = any(name.startswith("mode_enter") for name, _ in events)
        if self.plan is not None and not entered:
            lead = kc - self.plan["kc"] + 1
            if lead <= self.plan["sched"].valid:
                ctx = dataclasses.replace(self._context(self.plan["reason"]), base=self.plan["base"])
                out = self._plan_decision(ctx, gate.shifted(self.plan["sched"], lead - 1, self.K), decision, lead)
                if out is not None:
                    return out
                if self.plan_mode == "lock":
                    events.append(("plan_decertified", self.plan["chosen"]))
                    self.plan = None
                # floor: the classical decision stands this period; the plan stays for later leads
            else:
                events.append(("plan_expired", self.plan["chosen"]))
                self.plan = None
        trigger = "mode_enter" if entered else (
            "fallback" if decision.status == "fallback" and self.last_status != "fallback" else None)
        if trigger is None or kc - self.last_negotiation < self.cooldown:
            return decision
        self.last_negotiation = kc
        label = (f"theta_safe (P_bar {self._fmt(self.safe.P_bar)})" if decision.status == "fallback"
                 else f"classical optimizer contract (P_bar {self._fmt(decision.theta.P_bar)})")
        ctx = self._context(decision.reason)
        sched, rec = self.negotiator.run(ctx, history, decision.theta, label)
        rec.update(kc=kc, trigger=trigger, default_status=decision.status, default_reason=decision.reason)
        self.negotiations.append(rec)
        events.append((f"negotiation:{'agreed' if sched is not None else 'none'}", rec["chosen"]))
        if sched is None:
            return decision
        self.n_plans += 1
        self.plan = {"kc": kc, "sched": sched, "base": ctx.base, "reason": decision.reason, "n": self.n_plans,
                     "chosen": rec["chosen"]}
        out = self._plan_decision(ctx, sched, decision, 1)
        return decision if out is None else out

    def _plan_decision(self, ctx, sched, decision, lead):
        from seams.agents import gate
        t0 = time.perf_counter()
        tag = f"agreed-{self.plan['n']}-L{lead}"
        if self.plan_mode == "floor":
            sched = gate.floored(sched, decision.theta, ctx.base)
            if sched is None:
                return None
        if not gate.certify_schedule(ctx, sched).ok:
            return None
        theta = gate.lead_theta(ctx.base, sched, 1, tag)
        reason = f"plan {self.plan['n']} lead {lead}" + (f" over {decision.status}" if self.plan_mode == "floor" else "")
        return self._install(theta, "agreed", reason, _ms(t0), [])[0]

    @staticmethod
    def _fmt(P):
        return {j: round(float(v), 1) for j, v in P.items()}

    def pop_negotiations(self):
        out, self.negotiations = self.negotiations, []
        return out

    def _classical(self, kc, sig, Sq, blames, history):
        events = []
        self._update_modes(kc, sig, blames, history, events)
        V = dict(sig.V_rep)
        extra = {}
        for h in self.distrusted():
            V[h] = self.v_est(h, sig.P, sig.V_mv_rep)
            if self.estimator == "mv":
                # V_est stands in for the clean V_rep the drift bounds are anchored on
                extra[h] = {k: Fraction(str(self.mv[h]["est_err_max"])) for k in range(1, self.K + 1)}
                continue
            elapsed = kc - self.modes[("grid", h)]["anchor_k"]
            if elapsed + self.K > self.K_ext:  # anchor too old for any drift bound we have
                return self._fallback(f"distrust anchor of {h} too old", 0.0, events)
            extra[h] = {k: self.eps["D"][h][elapsed + k] - self.eps["D"][h][k] for k in range(1, self.K + 1)}
        anchor = {"V": V, "P": dict(sig.P), "S": Sq}
        caps = {h: Fraction(self.static.P_bar[h]) for h in self.static.P_bar}
        for d, h in self.modes:
            if d == "charging":
                caps[h] = min(caps[h], self.cap_frac * Fraction(self.static.P_bar[h]))
        floors = {h: Fraction(self.safe.P_bar[h]) for h in caps}
        self._ctx = dict(kc=kc, anchor=anchor, extra=extra, caps=caps, floors=floors)

        t0 = time.perf_counter()
        if not self.modes:
            v2 = v2_refinement(self.static, anchor, self.eps, self.phi, self.K)
            if v2.ok and self.v1_static.ok:
                return self._install(self.static, "static", "nominal", _ms(t0), events)
            if not self.on_decert:
                return self._install(self.static, "static", "uncertified (decertification trigger off)",
                                     _ms(t0), events, certified=False, cex=v2.counterexample)
            reason = "decertified"
        else:
            reason = "blame:" + ",".join(sorted(f"{d}:{h}" for d, h in self.modes))

        pb, ms = optimize_P_bar(self.static, anchor, self.eps, self.phi, self.K, caps, floors, extra, self.weights)
        if pb is None:
            return self._fallback(f"{reason}; optimizer infeasible", ms, events)
        if ms > self.budget_ms:
            return self._fallback(f"{reason}; optimizer over budget ({ms:.0f} ms)", ms, events)
        self.n_versions += 1
        cand = self.static.replace(f"reneg-{self.n_versions}", P_bar=pb, P_assumed=pb)
        v1 = v1_compatibility(cand, self.windows, self.K)
        v2 = v2_refinement(cand, anchor, self.eps, self.phi, self.K, extra=extra)
        exit_ok = v2_refinement(self.safe, anchor, self.eps, self.phi, self.K, extra=extra).ok
        if not (v1.ok and v2.ok and exit_ok):
            why = "V1" if not v1.ok else "V2" if not v2.ok else "theta_safe unreachable"
            return self._fallback(f"{reason}; candidate failed {why}", _ms(t0), events)
        return self._install(cand, "renegotiated", reason, _ms(t0), events)

    def _install(self, theta, status, reason, ms, events, certified=True, cex=None):
        self.theta = theta
        return Decision(theta, status, reason, ms, certified, {f"{d}:{h}" for d, h in self.modes}, cex), events

    def _fallback(self, reason, ms, events):
        # theta_safe carries an offline certificate (worst-case AC solves over the envelope)
        return self._install(self.safe, "fallback", reason, ms, events,
                             certified=bool(self.safe.meta.get("offline_certified")))


def _ms(t0):
    return 1000.0 * (time.perf_counter() - t0)
