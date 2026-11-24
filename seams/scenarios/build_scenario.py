import json
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from seams.cosim.config import resolve_path

NET_FILE = "net.net.xml"
NET_STAMP = "net.stamp.json"


def sumo_home(cfg):
    return Path(os.environ.get("SUMO_HOME") or Path(cfg["sumo"]["sumo_home"]).expanduser())


def build_net(cfg):
    scen = resolve_path(cfg["sumo"]["scenario_dir"])
    scen.mkdir(parents=True, exist_ok=True)
    net, stamp = scen / NET_FILE, scen / NET_STAMP
    params = cfg["network"]
    if net.exists() and stamp.exists() and json.loads(stamp.read_text()) == params:
        return net
    bin_dir = sumo_home(cfg) / "bin"
    if params.get("kind", "grid") == "osm":
        cmd = [
            str(bin_dir / "netconvert"), "--osm-files", str(resolve_path(params["osm_file"])),
            "--type-files", str(sumo_home(cfg) / "data" / "typemap" / "osmNetconvert.typ.xml"),
            *params.get("netconvert_options", []), "-o", str(net),
        ]
    else:
        cmd = [
            str(bin_dir / "netgenerate"), "--grid",
            f"--grid.number={params['grid_number']}",
            f"--grid.length={params['grid_length_m']}",
            f"--default.lanenumber={params['lanes']}",
            "-o", str(net),
        ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    stamp.write_text(json.dumps(params))
    return net


def parking_area_id(hub_id):
    return f"pa_{hub_id}"


def parking_area_start(cfg, hub):
    if "pa_start_m" in hub:
        return float(hub["pa_start_m"])
    return 0.3 * float(cfg["network"]["grid_length_m"])


def parking_area_end(cfg, hub):
    if "pa_end_m" in hub:
        return float(hub["pa_end_m"])
    return 0.7 * float(cfg["network"]["grid_length_m"])


def road_edges(cfg):
    tools = str(sumo_home(cfg) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import sumolib
    net = sumolib.net.readNet(str(build_net(cfg)))
    return sorted(e.getID() for e in net.getEdges() if e.allows("passenger"))


def generate_vehicles(cfg, rng):
    g = cfg["fleet"]["generator"]
    hubs = {h["edge"] for h in cfg["hubs"]}
    edges = [e for e in road_edges(cfg) if e not in hubs]
    weights = np.asarray(g["arrival_hourly"], dtype=float)
    hours = rng.choice(24, size=g["n_vehicles"], p=weights / weights.sum())
    departs = np.sort(hours * 3600 + rng.integers(0, 3600, size=g["n_vehicles"]))
    types = sorted(g["trip_mix"])
    mix = np.asarray([g["trip_mix"][t] for t in types], dtype=float)
    vehicles = []
    for i, depart in enumerate(departs):
        origin, dest = rng.choice(len(edges), size=2, replace=False)
        lo, hi = g["low_soc_range"] if rng.random() < g["charging_fraction"] else g["high_soc_range"]
        vehicles.append({
            "id": f"ev{i:05d}",
            "from": edges[origin], "to": edges[dest],
            "depart_s": int(depart),
            "init_soc": round(float(rng.uniform(lo, hi)), 4),
            "trip_type": types[rng.choice(len(types), p=mix / mix.sum())],
        })
    return vehicles


PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def _edge_geometry(cfg):
    tools = str(sumo_home(cfg) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import sumolib
    net = sumolib.net.readNet(str(build_net(cfg)))
    # real networks have one-way streets and dead ends: keep the largest strongly connected
    # component, so every origin reaches every hub and every destination
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = [e for e in net.getEdges() if e.allows("passenger")]
    index = {e.getID(): i for i, e in enumerate(edges)}
    pairs = [(index[e.getID()], index[o.getID()]) for e in edges for o in e.getOutgoing() if o.getID() in index]
    rows, cols = zip(*pairs)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(edges), len(edges)))
    _, label = connected_components(graph, directed=True, connection="strong")
    main = np.bincount(label).argmax()
    strong = {e.getID() for e, lab in zip(edges, label) if lab == main}
    missing = [h["edge"] for h in cfg["hubs"] if h["edge"] not in strong]
    if missing:
        raise ValueError(f"hub edges outside the network's main strongly connected component: {missing}")
    ids, xy = [], []
    for e in net.getEdges():
        if e.getID() in strong and e.getLength() >= 20.0:
            shape = np.asarray(e.getShape())
            ids.append(e.getID())
            xy.append(shape.mean(axis=0))
    return net, ids, np.asarray(xy)


def generate_vehicles_empirical(cfg, rng):
    import pandas as pd
    g = cfg["fleet"]["generator"]
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    soc = np.load(PROCESSED / "soc_start.npy")
    km = np.load(PROCESSED / "trip_km.npy")
    start_hour = np.load(PROCESSED / "trip_start_hour.npy")
    net, ids, xy = _edge_geometry(cfg)
    hub_edges = {h["edge"] for h in cfg["hubs"]}
    keep = np.array([i not in hub_edges for i in ids])
    ids, xy = [i for i, k in zip(ids, keep) if k], xy[keep]
    detour = float(g.get("detour_factor", 1.3))      # network distance / straight-line distance
    speed = float(g.get("approach_speed_mps", 9.0))  # urban average, for departure timing only
    km_max = float(g.get("approach_km_max", 4.0))
    km = km[(km >= 0.3) & (km <= km_max)]

    trips = []
    for hub in cfg["hubs"]:
        pool = sessions[sessions.profile == hub["profile"]].reset_index(drop=True)
        hub_xy = np.asarray(net.getEdge(hub["edge"]).getShape()).mean(axis=0)
        dist = np.linalg.norm(xy - hub_xy, axis=1)
        near = np.flatnonzero(dist <= float(g.get("destination_radius_m", 400.0)))
        n = rng.poisson(float(g["sessions_per_port_day"][hub["id"]]) * hub["n_evse"])
        for j in rng.choice(len(pool), size=n):
            row = pool.iloc[int(j)]
            d_m = float(rng.choice(km)) * 1000.0
            band = np.flatnonzero(np.abs(dist * detour - d_m) <= max(150.0, 0.15 * d_m))
            origin = int(rng.choice(band)) if len(band) else int(np.argmax(dist))
            dest = int(rng.choice(near))
            depart = float(row.arr_s) - dist[origin] * detour / speed - float(g.get("parking_margin_s", 60.0))
            trips.append({
                "from": ids[origin], "to": ids[dest], "depart_s": int(max(0.0, depart)),
                "init_soc": round(float(rng.choice(soc)), 4), "trip_type": hub["profile"],
                "wants_charge": True, "dwell_s": round(float(row.dwell_s)), "request_kwh": round(float(row.kwh), 3),
            })
    p = start_hour / start_hour.sum()
    hours = rng.choice(24, size=int(g.get("through_vehicles", 0)), p=p)
    for hour in hours:
        o, d = rng.choice(len(ids), size=2, replace=False)
        trips.append({
            "from": ids[o], "to": ids[d], "depart_s": int(hour * 3600 + rng.integers(0, 3600)),
            "init_soc": round(float(rng.uniform(*g.get("through_soc_range", [0.5, 0.9]))), 4), "trip_type": "through",
        })
    trips.sort(key=lambda v: (v["depart_s"], v["from"], v["to"]))
    return [dict(v, id=f"ev{i:05d}") for i, v in enumerate(trips)]


def fleet_vehicles(cfg, rng):
    fleet = cfg["fleet"]
    if "vehicles" in fleet:
        return fleet["vehicles"]
    if fleet["generator"].get("kind") == "empirical":
        return generate_vehicles_empirical(cfg, rng)
    return generate_vehicles(cfg, rng)


def write_run_files(cfg, run_dir, rng=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    net = build_net(cfg)

    # Hubs: one off-road parkingArea per hub, capacity = EVSE count. A full hub makes
    # arriving vehicles wait on the access lane, which is the transport-side spillback.
    lines = ["<additional>"]
    for hub in cfg["hubs"]:
        start, end = parking_area_start(cfg, hub), parking_area_end(cfg, hub)
        lines.append(
            f'    <parkingArea id="{parking_area_id(hub["id"])}" lane="{hub["edge"]}_0" '
            f'startPos="{start:.1f}" endPos="{end:.1f}" roadsideCapacity="{hub["n_evse"]}" '
            f'onRoad="false"/>'
        )
    lines.append("</additional>")
    (run_dir / "hubs.add.xml").write_text("\n".join(lines) + "\n")

    ev = cfg["ev"]
    lines = [
        "<routes>",
        '    <vType id="ev" vClass="passenger" emissionClass="Energy/unknown">',
        '        <param key="has.battery.device" value="true"/>',
        f'        <param key="device.battery.capacity" value="{ev["capacity_wh"]}"/>',
        "    </vType>",
    ]
    av = (cfg.get("av_fleet") or {}).get("vehicle")
    if (cfg.get("av_fleet") or {}).get("n") and av:
        # the AV fleet's own vehicle (config av_fleet.vehicle): SUMO energy-model parameters
        lines += [
            f'    <vType id="av" vClass="passenger" emissionClass="Energy/unknown" mass="{av["mass_kg"]}">',
            '        <param key="has.battery.device" value="true"/>',
            f'        <param key="device.battery.capacity" value="{av["capacity_wh"]}"/>',
            *[f'        <param key="{k}" value="{v}"/>' for k, v in av.get("energy", {}).items()],
            "    </vType>",
        ]
    vehicles = sorted(fleet_vehicles(cfg, rng), key=lambda v: (v["depart_s"], v["id"]))
    for v in vehicles:
        charge = v["init_soc"] * ev["capacity_wh"]
        lines += [
            f'    <trip id={quoteattr(v["id"])} type="ev" depart="{v["depart_s"]}" '
            f'from="{v["from"]}" to="{v["to"]}">',
            f'        <param key="device.battery.chargeLevel" value="{charge:.3f}"/>',
            f'        <param key="tripType" value="{v["trip_type"]}"/>',
            *([f'        <param key="wantsCharge" value="1"/>'] if v.get("wants_charge") else []),
            *([f'        <param key="dwellS" value="{v["dwell_s"]}"/>'] if v.get("dwell_s") is not None else []),
            *([f'        <param key="requestKWh" value="{v["request_kwh"]}"/>'] if v.get("request_kwh") is not None else []),
            "    </trip>",
        ]
    lines.append("</routes>")
    (run_dir / "routes.rou.xml").write_text("\n".join(lines) + "\n")

    # SUMO splits file lists on spaces, and the project path contains spaces, so the shared
    # network is linked into the run directory and referenced by a bare file name.
    link = run_dir / NET_FILE
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(net)
    clock, sumo = cfg["clock"], cfg["sumo"]
    sumocfg = run_dir / "sim.sumocfg"
    sumocfg.write_text(
        "<configuration>\n"
        "    <input>\n"
        f'        <net-file value="{NET_FILE}"/>\n'
        '        <route-files value="routes.rou.xml"/>\n'
        '        <additional-files value="hubs.add.xml"/>\n'
        "    </input>\n"
        "    <time>\n"
        f'        <step-length value="{clock["sumo_step_s"]}"/>\n'
        "    </time>\n"
        "    <processing>\n"
        # SUMO's default teleports a vehicle stuck for 300 s past its stop, so a car waiting
        # for a charger would silently vanish from the queue; vehicles wait instead, and a
        # junction blocker is ignored after a while so a queue cannot deadlock the network
        f'        <time-to-teleport value="{sumo.get("time_to_teleport_s", -1)}"/>\n'
        f'        <ignore-junction-blocker value="{sumo.get("ignore_junction_blocker_s", 60)}"/>\n'
        "    </processing>\n"
        "</configuration>\n"
    )
    return sumocfg
