"""SUMO adapter. Every TraCI call the testbed makes goes through this class, so a change in
SUMO's API (battery parameter names, stop handling) stays local.

Verified against SUMO 1.28: battery charge is `device.battery.chargeLevel` in Wh (get/set),
capacity is `device.battery.capacity` in Wh.
"""
import itertools
import os
from pathlib import Path
import sys

from seams.scenarios.build_scenario import parking_area_id, parking_area_start, sumo_home

CHARGE = "device.battery.chargeLevel"
CAPACITY = "device.battery.capacity"
PARK_FOREVER_S = 1e7  # parking stop duration; the master ends it with resume()

_labels = itertools.count()


def _import_traci(cfg):
    tools = str(sumo_home(cfg) / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import traci
    return traci


class SumoAdapter:
    def __init__(self, cfg, sumocfg_path, seed):
        self.cfg = cfg
        self.sumocfg_path = sumocfg_path
        self.seed = seed
        self.conn = None
        self._traci = _import_traci(cfg)
        self.arrived = set()  # vehicles that finished their trip during the last advance()
        self._exc = self._traci.TraCIException

    def start(self):
        binary = "sumo-gui" if self.cfg["sumo"].get("gui") else "sumo"
        cmd = [
            str(sumo_home(self.cfg) / "bin" / binary),
            "-c", str(self.sumocfg_path),
            "--seed", str(self.seed),
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--no-warnings", "true",
            "--error-log", str(Path(self.sumocfg_path).parent / "sumo_errors.log"),
        ]
        os.environ.setdefault("SUMO_HOME", str(sumo_home(self.cfg)))
        if self.cfg["sumo"].get("libsumo"):
            # in-process SUMO (same API, one simulation per process): the whole co-simulation
            # then lives in one process, which os.fork can snapshot (seams.attacks.adaptive).
            # libsumo links its own Arrow: once imported, pandas.read_parquet fails in that
            # process (writing still works), so build one Master per process and read results
            # elsewhere.
            import libsumo
            libsumo.start(cmd)
            self._exc = libsumo.TraCIException
            self.conn = libsumo
            return
        label = f"seams-{next(_labels)}"
        self._traci.start(cmd, label=label)
        self.conn = self._traci.getConnection(label)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # --- time ---------------------------------------------------------------
    def advance(self, n_steps):
        self.arrived = set()
        for _ in range(n_steps):
            self.conn.simulationStep()
            self.arrived.update(self.conn.simulation.getArrivedIDList())

    def time(self):
        return self.conn.simulation.getTime()

    # --- vehicles -----------------------------------------------------------
    def vehicle_ids(self):
        return self.conn.vehicle.getIDList()

    def param(self, vid, key):
        return self.conn.vehicle.getParameter(vid, key)

    def charge_wh(self, vid):
        return float(self.conn.vehicle.getParameter(vid, CHARGE))

    def capacity_wh(self, vid):
        return float(self.conn.vehicle.getParameter(vid, CAPACITY))

    def set_charge_wh(self, vid, wh):
        self.conn.vehicle.setParameter(vid, CHARGE, f"{wh:.6f}")

    def distance_m(self, vid):
        return self.conn.vehicle.getDistance(vid)

    def set_param(self, vid, key, value):
        self.conn.vehicle.setParameter(vid, key, value)

    def add_trip(self, vid, from_edge, to_edge, type_id="ev"):
        """Insert a vehicle now on the fastest route between two edges (an AV ride). Returns
        False if there is no route."""
        edges = self.conn.simulation.findRoute(from_edge, to_edge, vType="ev").edges
        if not edges:
            return False
        try:
            self.conn.route.add(f"{vid}_route", list(edges))
            self.conn.vehicle.add(vid, f"{vid}_route", typeID=type_id, depart="now")
            self.conn.vehicle.setParameter(vid, "tripType", "av")
        except self._exc:
            return False
        return True

    def road(self, vid):
        return self.conn.vehicle.getRoadID(vid)

    def speed(self, vid):
        return self.conn.vehicle.getSpeed(vid)

    def driving_distance(self, vid, edge, pos):
        """Driving distance along the vehicle's route to (edge, pos); negative if unreachable."""
        return self.conn.vehicle.getDrivingDistance(vid, edge, pos)

    def destination(self, vid):
        return self.conn.vehicle.getRoute(vid)[-1]

    # --- routing and hubs ---------------------------------------------------
    def edge_travel_time(self, from_edge, to_edge):
        """Current-travel-time route cost, or None if unreachable."""
        route = self.conn.simulation.findRoute(from_edge, to_edge, vType="ev")
        return route.travelTime if route.edges else None

    def send_to_hub(self, vid, hub_id, hub_edge, destination):
        """Reroute via the hub's edge to the original destination and add a parking stop.
        Returns False if the vehicle cannot be rerouted right now (e.g. inside a junction)."""
        cur = self.road(vid)
        if cur.startswith(":"):
            return False
        hub = next(h for h in self.cfg["hubs"] if h["id"] == hub_id)
        if cur == hub_edge and self.conn.vehicle.getLanePosition(vid) > parking_area_start(self.cfg, hub) - 10.0:
            return False  # already at or past the parking entrance on this edge
        to_hub = self.conn.simulation.findRoute(cur, hub_edge, vType="ev").edges
        onward = self.conn.simulation.findRoute(hub_edge, destination, vType="ev").edges
        if not to_hub or not onward:
            return False
        try:
            self.conn.vehicle.setRoute(vid, list(to_hub) + list(onward[1:]))
            self.conn.vehicle.setParkingAreaStop(vid, parking_area_id(hub_id), duration=PARK_FOREVER_S)
        except self._exc:
            return False  # e.g. already past the parking area on the hub edge
        return True

    def cancel_stop(self, vid):
        """Remove the vehicle's next stop (a refused hub's parking stop)."""
        try:
            self.conn.vehicle.replaceStop(vid, 0, "")
        except self._exc:
            pass

    def parked_at(self, hub_id):
        return self.conn.parkingarea.getVehicleIDs(parking_area_id(hub_id))

    def resume(self, vid):
        self.conn.vehicle.resume(vid)
