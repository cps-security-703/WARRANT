import shutil
from pathlib import Path

import numpy as np
import yaml

from seams.cosim.config import load_config, resolve_path

# scenario -> (attack file, strength key, default strength); strengths are the RQ1 maxima
SCENARIOS = {
    "normal": (None, None, None),
    "fdi": ("seams/config/attacks/fdi.yaml", "bias_pu", -0.04),
    "fdi_morning": ("seams/config/attacks/fdi_morning.yaml", "bias_pu", -0.06),
    "navigation": ("seams/config/attacks/navigation.yaml", "fraction", 1.0),
    "aggregator": ("seams/config/attacks/aggregator.yaml", "fraction", 1.0),
    # demo scenario (not in the paper's campaigns): redirect + lure, strength = lure_fraction
    "navigation_redirect": ("seams/config/attacks/navigation_redirect.yaml", "lure_fraction", 1.0),
}
# condition -> (config, overrides, label); the same three conditions as the RQ1 campaign
CONDITIONS = {
    "protection_only": ("seams/config/phase2.yaml", {}, "Protection only (no WARRANT)"),
    "static_contracts": ("seams/config/phase5.yaml", {"renegotiation.enabled": False}, "Static contracts"),
    "warrant": ("seams/config/phase5.yaml", {}, "WARRANT (contracts + renegotiation)"),
}
# vehicle display codes
DRIVING, PARKED = 0, 9
SEEK0, QUEUE0 = 1, 5      # + hub index


def build_config(spec):
    path, over, _ = CONDITIONS[spec["condition"]]
    over = dict(over)
    over.update({"run.seed": int(spec["seed"]), "evse.scheduler": spec["scheduler"],
                 "scenario.feeder": spec["feeder"], "scenario.electrification": str(spec["electrification"])})
    attacks = []
    afile, key, default = SCENARIOS[spec["scenario"]]
    if afile:
        attacks = yaml.safe_load(resolve_path(afile).read_text())["attacks"]
        strength = default if spec.get("strength") is None else spec["strength"]
        attacks = [dict(a, **{key: strength}) if key in a else a for a in attacks]
    over["attacks"] = attacks
    return load_config(path, over)


def _feeder_topology():
    import opendssdirect as dss
    bus = lambda b: b.split(".")[0].lower()
    edges = []
    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        edges.append((bus(dss.Lines.Bus1()), bus(dss.Lines.Bus2()), "line", int(dss.Lines.Phases())))
    for name in dss.Transformers.AllNames():
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        b = [bus(x) for x in dss.CktElement.BusNames()]
        if len(b) >= 2:
            edges.append((b[0], b[1], "reg" if name.lower().startswith("reg") else "xfmr", 3))
    nodes = [bus(n) for n in dss.Circuit.AllNodeNames()]
    names = sorted(set(nodes))
    index = np.array([names.index(n) for n in nodes])
    return names, index, edges


def _bus_voltages(names, index):
    import opendssdirect as dss
    pu = np.asarray(dss.Circuit.AllBusMagPu(), dtype=float)
    out = np.full(len(names), np.inf)
    np.minimum.at(out, index, np.where(pu > 0.1, pu, np.inf))  # ignore dead nodes
    out[np.isinf(out)] = np.nan
    return out.astype(np.float32)


def run_stream(spec, q):
    try:
        _run(spec, q)
    except Exception as e:  # report instead of dying silently
        import traceback
        q.put(("error", f"{spec['condition']}: {type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _run(spec, q):
    from seams.cosim.master import Master
    from seams.domains.transport.fleet import VState
    cfg = build_config(spec)
    run_dir = Path(spec["run_dir"])
    shutil.rmtree(run_dir, ignore_errors=True)  # the run logger appends; start clean
    q.put(("status", f"{spec['condition']}: building scenario (network, trips, contracts)..."))
    m = Master(cfg, run_dir.name, run_dir=run_dir)
    hub_ids = list(m.hubs)
    hidx = {h: i for i, h in enumerate(hub_ids)}
    names, index, edges = _feeder_topology()

    events = []
    log = m.logger.log

    def log_hook(table, **row):
        if table == "events":
            events.append((float(row.get("t_s") or 0.0), str(row.get("event")), row.get("hub")))
        return log(table, **row)
    m.logger.log = log_hook

    fine_s = m.clock.fine_period_s
    start_s, end_s = float(spec["start_s"]), float(spec["end_s"])
    sub = max(1, int(round(float(spec["sub_s"]) / float(cfg["clock"]["sumo_step_s"]))))
    snaps, capture = [], [False]

    def snapshot():
        conn = m.sumo.conn
        ids = conn.vehicle.getIDList()
        xy = np.empty((len(ids), 2), np.float32)
        code = np.zeros(len(ids), np.int8)
        vehicles = m.fleet.vehicles if m.fleet else {}
        for i, vid in enumerate(ids):
            xy[i] = conn.vehicle.getPosition(vid)
            v = vehicles.get(vid)
            if v is None:
                continue
            if v.state is VState.PARKED:
                code[i] = PARKED
            elif v.state is VState.SEEKING and v.hub in hidx:
                queued = v.reached  # crossed the hub's approach geofence: in its queue
                code[i] = (QUEUE0 if queued else SEEK0) + hidx[v.hub]
        keep = code != PARKED  # parked vehicles are drawn as hub ports, not on the road
        snaps.append((conn.simulation.getTime(), xy[keep], code[keep]))

    def advance(n_steps):  # SumoAdapter.advance, plus snapshots
        s = m.sumo
        s.arrived = set()
        for i in range(n_steps):
            s.conn.simulationStep()
            s.arrived.update(s.conn.simulation.getArrivedIDList())
            if capture[0] and (i + 1) % sub == 0:
                snapshot()

    m.start()
    m.sumo.advance = advance
    q.put(("header", {
        "condition": spec["condition"], "label": CONDITIONS[spec["condition"]][2],
        "scenario": spec["scenario"], "hubs": cfg["hubs"], "attacks": cfg.get("attacks", []),
        "n_evse": {h: len(hub.station_ids) for h, hub in m.hubs.items()},
        "port_kw": float(cfg["evse"]["max_rate_a"]) * float(cfg["evse"]["voltage_v"]) / 1000.0,
        "buses": names, "edges": edges, "feeder": cfg["scenario"]["feeder"],
        "phi": cfg["grid"]["phi_safe"], "net": str(resolve_path(cfg["sumo"]["scenario_dir"]) / "net.net.xml"),
        "warrant": m.reneg is not None, "fine_s": fine_s,
    }))

    n = m.clock.fine_steps(cfg["run"]["horizon_s"])
    for k in range(n):
        t_s = (k + 1) * fine_s
        if t_s > end_s:
            break
        capture[0] = t_s > start_s - fine_s
        snaps.clear()
        m.k = k
        m.fine_step(k)
        rows = {r["hub"]: r for r in m._period[-len(hub_ids):]}
        bus_v = _bus_voltages(names, index) if capture[0] else None
        if m.clock.is_contract_boundary(k):
            m.contract_step(k)
        m.k = k + 1
        hubs = {}
        for h, hub in m.hubs.items():
            r = rows[h]
            occ = m.bridge.occupancy[h]
            hubs[h] = {
                "occupied": len(occ), "queue": int(r["queue"]), "P": float(r.get("P_pcc", r["P_sched"])),
                "V_true": r.get("V_true"), "V_rep": r.get("V_reported"), "V_der": r.get("V_derating"),
                "loading": r.get("xfmr_loading"), "allow": float(m.d_kw[h]) if m.grid else None,
                "cap": float(m.theta.P_bar[h]) if m.theta is not None else None,
                "attack": r.get("attack_active"), "offline": bool(r.get("offline")),
                "port_kw": (hub.last_rates_a * hub.voltage_v / 1000.0).astype(np.float32),
                "port_busy": np.array([sid in occ for sid in hub.station_ids]),
            }
        modes = sorted(f"{d}:{h}" for d, h in m.reneg.modes) if m.reneg else []
        q.put(("frame", {
            "t_s": t_s, "hubs": hubs, "bus_v": bus_v, "snaps": list(snaps),
            "home_kw": rows[hub_ids[0]].get("home_kw"), "bg": rows[hub_ids[0]].get("bg_mult"),
            "events": events[:], "modes": modes,
        }))
        events.clear()
    m.sumo.close()
    q.put(("done", None))
