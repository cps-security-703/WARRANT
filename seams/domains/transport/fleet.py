"""Per-vehicle state machine on the transport side: drive -> seek hub -> (queue) -> parked -> drive."""
from dataclasses import dataclass, field
from enum import Enum

from seams.domains.transport.routing import select_hub
from seams.scenarios.build_scenario import parking_area_end, parking_area_start


class VState(Enum):
    DRIVING = "driving"
    SEEKING = "seeking"   # rerouted to a hub, not yet parked (may be queued)
    PARKED = "parked"     # occupies a hub space, handed to the bridge
    ARRIVED = "arrived"   # trip finished, left the simulation


@dataclass
class Vehicle:
    vid: str
    trip_type: str
    destination: str
    capacity_wh: float
    wants_charge: bool = False  # seeks a hub at departure (a sampled real session), not on low SOC
    dwell_s: float = None       # the vehicle's own session, if it carries one
    request_kwh: float = None
    state: VState = VState.DRIVING
    hub: str = None
    reached: bool = False  # counted in lambda_h for the current visit (queued or parked)
    visits: int = 0
    excluded: set = field(default_factory=set)  # hubs that refused this vehicle on this visit
    stranded: bool = False
    depart_s: float = None   # first seen in the simulation
    arrive_s: float = None   # trip finished
    refusals: int = 0
    events: list = field(default_factory=list)


@dataclass
class TransportStep:
    newly_parked: list    # [(vid, hub)]
    arrivals: dict        # hub -> vehicles that reached the hub this period (attempts, pre-admission)
    queue: dict           # hub -> seeking vehicles halted near the hub
    spillback_m: dict     # hub -> how far back from the parking entrance the queue reaches
    refused: dict         # hub -> vehicles refused by hub admission this period
    stranded: list        # vehicles left with no admissible hub


class Fleet:
    def __init__(self, cfg, sumo, admission=None):
        self.cfg = cfg
        self.sumo = sumo
        self.admission = admission  # hub admission shield, or None
        self._refused = {}
        self._stranded = []
        self.hubs = {h["id"]: h for h in cfg["hubs"]}
        self.pa_end = {h: parking_area_end(cfg, hub) for h, hub in self.hubs.items()}
        self.pa_len = {h: parking_area_end(cfg, hub) - parking_area_start(cfg, hub) for h, hub in self.hubs.items()}
        self.vehicles = {}
        t = cfg.get("transport", {})
        dwell = [tt["dwell_s"] for tt in cfg["trip_types"].values()]
        self.mean_dwell_s = float(t.get("mean_dwell_s") or sum(dwell) / len(dwell))
        self.halt_speed = float(t.get("halt_speed_mps", 0.1))
        self.queue_horizon = float(t.get("queue_horizon_m", 1500.0))
        self.queue_gap = float(t.get("queue_gap_m", 20.0))
        self.geofence = float(t.get("arrival_geofence_m", 250.0))

    def _register_new(self, t_s):
        for vid in self.sumo.vehicle_ids():
            if vid not in self.vehicles:
                num = lambda key: float(self.sumo.param(vid, key) or "nan")
                dwell, request = num("dwellS"), num("requestKWh")
                self.vehicles[vid] = Vehicle(
                    vid=vid,
                    trip_type=self.sumo.param(vid, "tripType"),
                    destination=self.sumo.destination(vid),
                    capacity_wh=self.sumo.capacity_wh(vid),
                    wants_charge=self.sumo.param(vid, "wantsCharge") == "1",
                    dwell_s=None if dwell != dwell else dwell,
                    request_kwh=None if request != request else request,
                    depart_s=t_s,
                )

    def _handle_arrived(self, t_s):
        for vid in self.sumo.arrived:
            v = self.vehicles.get(vid)
            if v is not None:
                v.state = VState.ARRIVED
                v.arrive_s = t_s
                v.events.append((t_s, "arrived"))

    def _trigger_low_soc(self, t_s, views_fn, excluded):
        ev = self.cfg["ev"]
        for vid in sorted(self.vehicles):
            v = self.vehicles[vid]
            if v.state is not VState.DRIVING or v.visits >= ev["max_charge_visits"]:
                continue
            if not v.wants_charge and self.sumo.charge_wh(vid) / v.capacity_wh >= ev["low_soc_threshold"]:
                continue
            cur = self.sumo.road(vid)
            if cur.startswith(":"):
                continue  # inside a junction; retry next fine step
            times = {h: self.sumo.edge_travel_time(cur, self.hubs[h]["edge"]) for h in self.hubs}
            hub, _ = select_hub(times, views_fn(vid), self.mean_dwell_s, set(excluded) | v.excluded)
            if hub is None and v.excluded and not v.stranded:
                v.stranded = True  # every hub refused it on this visit
                v.events.append((t_s, "stranded"))
                self._stranded.append(vid)
            if hub and self.sumo.send_to_hub(vid, hub, self.hubs[hub]["edge"], v.destination):
                v.state, v.hub, v.reached = VState.SEEKING, hub, False
                v.events.append((t_s, f"seek:{hub}"))

    def _arrive(self, v, hub, arrivals, force=False):
        """The single arrival path: count the attempt, then ask hub admission. A refused vehicle
        drops its parking stop and re-enters hub selection with this hub excluded."""
        v.reached = True
        arrivals[hub] += 1
        if self.admission is None or self.admission.request(hub) or force:
            return True
        self._refused[hub] = self._refused.get(hub, 0) + 1
        v.refusals += 1
        self.sumo.cancel_stop(v.vid)
        v.excluded.add(hub)
        v.state, v.hub, v.reached = VState.DRIVING, None, False
        v.events.append((None, f"refused:{hub}"))
        return False

    def _queues(self, arrivals):
        """A hub's queue is the contiguous chain of halted seeking vehicles that starts at the
        parking entrance and continues back with at most queue_gap_m between vehicles.
        Not queued: a seeking vehicle halted inside the parking span (manoeuvring into a free
        space) or elsewhere on the road (e.g. yielding at a junction)."""
        halted = {h: [] for h in self.hubs}
        for vid in sorted(self.vehicles):
            v = self.vehicles[vid]
            if v.state is not VState.SEEKING:
                continue
            dist = self.sumo.driving_distance(vid, self.hubs[v.hub]["edge"], self.pa_end[v.hub])
            # arrival = crossing the hub's approach geofence (counted once per visit)
            if not v.reached and 0 <= dist <= self.geofence:
                if not self._arrive(v, v.hub, arrivals):
                    continue
            if self.sumo.speed(vid) <= self.halt_speed and self.pa_len[v.hub] - 1.0 <= dist <= self.queue_horizon:
                halted[v.hub].append((dist, vid))
        queue = {h: 0 for h in self.hubs}
        spill = {h: 0.0 for h in self.hubs}
        for h, cars in halted.items():
            reach = self.pa_len[h] + self.queue_gap
            for dist, vid in sorted(cars):
                if dist > reach:
                    break
                v = self.vehicles[vid]
                if not v.reached and not self._arrive(v, h, arrivals):
                    continue  # queued beyond the geofence and refused: leaves the queue
                queue[h] += 1
                spill[h] = dist
                reach = dist + self.queue_gap
        return queue, spill

    def step(self, t_s, views_fn, excluded=()):
        """Run after SUMO advanced one fine period. views_fn(vid) -> {hub: HubView} is what that
        vehicle's routing believes (per vehicle, so navigation spoofing can target a subset)."""
        self._handle_arrived(t_s)
        self._register_new(t_s)
        self._trigger_low_soc(t_s, views_fn, excluded)
        self._refused, self._stranded = {h: 0 for h in self.hubs}, []
        arrivals = {h: 0 for h in self.hubs}
        newly_parked = []
        for hub in sorted(self.hubs):
            for vid in self.sumo.parked_at(hub):
                v = self.vehicles[vid]
                if v.state is VState.SEEKING and v.hub == hub:
                    if not v.reached:  # parked within one step of crossing the geofence
                        self._arrive(v, hub, arrivals, force=True)
                    v.state = VState.PARKED
                    v.events.append((t_s, f"parked:{hub}"))
                    newly_parked.append((vid, hub))
        queue, spill = self._queues(arrivals)
        return TransportStep(newly_parked, arrivals, queue, spill, dict(self._refused), list(self._stranded))

    def release(self, vid, t_s):
        v = self.vehicles[vid]
        v.state, v.hub, v.reached = VState.DRIVING, None, False
        v.excluded, v.stranded = set(), False
        v.visits += 1
        v.events.append((t_s, "released"))
        self.sumo.resume(vid)
