import numpy as np

from seams.domains.transport.fleet import VState
from seams.scenarios.build_scenario import PROCESSED, _edge_geometry

TRIP_TYPE = "av"


class AvFleet:
    def __init__(self, cfg, sumo, rng):
        a = cfg["av_fleet"]
        self.cfg, self.sumo, self.rng = cfg, sumo, rng
        veh = a.get("vehicle")
        self.type_id = "av" if veh else "ev"
        self.capacity_wh = float(veh["capacity_wh"] if veh else cfg["ev"]["capacity_wh"])
        self.charge_soc = float(a["charge_soc"])
        self.idle_s = tuple(float(x) for x in a["idle_s"])
        self.end_s = float(a["end_s"])
        _, ids, xy = _edge_geometry(cfg)
        hub_edges = {h["edge"] for h in cfg["hubs"]}
        keep = np.array([i not in hub_edges for i in ids])
        self.edges, self.xy = [i for i, k in zip(ids, keep) if k], xy[keep]
        km = np.load(PROCESSED / "trip_km.npy")
        g = cfg["fleet"]["generator"]
        self.km = km[(km >= 0.3) & (km <= float(g.get("approach_km_max", 4.0)))]
        self.detour = float(g.get("detour_factor", 1.3))
        n = int(a["n"])
        start = rng.uniform(*[float(x) for x in a["start_s"]], size=n)
        self.avs = [{
            "id": f"av{i:02d}", "edge": self.edges[int(rng.integers(len(self.edges)))],
            "soc_wh": float(a["init_soc"]) * self.capacity_wh, "next_s": float(start[i]),
            "ride": None, "dest": None, "rides": 0, "seen": False, "km": 0.0, "wh_used": 0.0, "ride_m": 0.0,
        } for i in range(n)]
        self.by_id = {av["id"]: av for av in self.avs}
        self.forced = {}   # av id -> done (sent to charge by a compromised dispatcher)
        self.draining = set()  # av ids a compromised dispatcher keeps on long rides, no charging

    @staticmethod
    def av_of(vid):
        return vid.split("_r")[0] if vid and vid.startswith("av") and "_r" in vid else None

    def force(self, members):
        for m in members:
            self.forced.setdefault(m, False)

    def drain(self, members):
        self.draining = set(members)

    def _destination(self, origin, longest=False):
        o = self.xy[self.edges.index(origin)] if origin in self.edges else self.xy[int(self.rng.integers(len(self.xy)))]
        d_m = (float(self.km.max()) if longest else float(self.rng.choice(self.km))) * 1000.0
        dist = np.linalg.norm(self.xy - o, axis=1) * self.detour
        band = np.flatnonzero(np.abs(dist - d_m) <= max(150.0, 0.15 * d_m))
        return self.edges[int(self.rng.choice(band))] if len(band) else self.edges[int(np.argmax(dist))]

    def _start_ride(self, av, t_s):
        for _ in range(10):
            dest = self._destination(av["edge"], longest=av["id"] in self.draining)
            if dest == av["edge"]:
                continue
            vid = f"{av['id']}_r{av['rides']:03d}"
            if self.sumo.add_trip(vid, av["edge"], dest, self.type_id):
                av.update(ride=vid, dest=dest, rides=av["rides"] + 1, seen=False)
                return
        av["next_s"] = t_s + self.idle_s[0]  # no route now; try again later

    def step(self, t_s, fleet):
        present = set(self.sumo.vehicle_ids())
        for av in self.avs:
            vid = av["ride"]
            if vid is None:
                continue
            if vid in present:
                if not av["seen"]:  # inserted during the last advance: carry the battery over
                    self.sumo.set_charge_wh(vid, av["soc_wh"])
                    low = av["soc_wh"] / self.capacity_wh < self.charge_soc and av["id"] not in self.draining
                    pending = self.forced.get(av["id"]) is False
                    if low or pending:
                        self.sumo.set_param(vid, "wantsCharge", "1")
                        if pending:
                            self.forced[av["id"]] = True
                    av["seen"], av["ride_m"] = True, 0.0
                else:
                    cur, d_m = self.sumo.charge_wh(vid), self.sumo.distance_m(vid)
                    # driving ledger (charging raises the charge, so only drops count as use)
                    av["wh_used"] += max(0.0, av["soc_wh"] - cur)
                    av["km"] += max(0.0, d_m - av["ride_m"]) / 1000.0
                    av["soc_wh"], av["ride_m"] = cur, d_m
                    v = fleet.vehicles.get(vid) if fleet is not None else None
                    if self.forced.get(av["id"]) is False and v is not None and v.state is VState.DRIVING and v.visits == 0:
                        v.wants_charge = True
                        self.forced[av["id"]] = True
            elif av["seen"] or vid in self.sumo.arrived:
                av.update(ride=None, edge=av["dest"], seen=False)
                av["next_s"] = t_s + (0.0 if av["id"] in self.draining else float(self.rng.uniform(*self.idle_s)))
        for av in self.avs:
            if av["ride"] is None and t_s < self.end_s:
                if self.forced.get(av["id"]) is False or av["id"] in self.draining:
                    av["next_s"] = min(av["next_s"], t_s)
                if av["next_s"] <= t_s:
                    self._start_ride(av, t_s)
