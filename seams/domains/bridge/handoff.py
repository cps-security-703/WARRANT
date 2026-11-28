"""SUMO -> ACN-Sim bridge: a parked SUMO vehicle becomes an ACN-Sim session, and is handed
back when charging ends (fully served, or dwell budget reached).

Units: SUMO battery values are Wh, ACN-Sim energies are kWh, both clocks are fine periods.
"""
from dataclasses import dataclass


@dataclass
class Session:
    vehicle_id: str          # also the ACN-Sim session id, so logs join
    hub: str
    station_id: str
    trip_type: str
    arrival_k: int
    departure_k: int
    capacity_wh: float
    charge_at_plugin_wh: float
    requested_kwh: float
    delivered_kwh: float = 0.0
    released_k: int = None
    released_reason: str = None    # "completed" | "departure"
    charge_before_release_wh: float = None
    charge_after_release_wh: float = None

    @property
    def unserved_kwh(self):
        return max(0.0, self.requested_kwh - self.delivered_kwh)


class Bridge:
    def __init__(self, cfg, hubs, clock):
        self.cfg = cfg
        self.hubs = hubs             # hub_id -> AcnHub
        self.clock = clock
        self.occupancy = {h: {} for h in hubs}  # hub -> station_id -> vehicle_id
        self.active = {}             # vehicle_id -> Session
        self.finished = []
        # true: a served vehicle leaves at once (fast turnover); false: it stays plugged for its
        # whole dwell, as drivers in the session data do (idle occupancy is part of real demand)
        self.release_on_complete = bool(cfg["bridge"].get("release_on_complete", True))
        # dwell sensitivity (guideline: results for two dwell settings): every hub session's dwell
        # budget, sampled or per trip type, scaled by this factor
        self.dwell_scale = float(cfg["bridge"].get("dwell_scale", 1.0))

    def free_evses(self, hub):
        return len(self.hubs[hub].station_ids) - len(self.occupancy[hub])

    def plug_in(self, vid, hub_id, k, charge_wh, capacity_wh, trip_type, dwell_s=None, request_kwh=None):
        """Create the ACN-Sim session for a vehicle that occupied a hub space during fine
        period k. Charging starts at period k + 1: the vehicle was not there for all of k.
        dwell_s / request_kwh: the vehicle's own session (sampled from real session data);
        otherwise the trip type's defaults."""
        hub = self.hubs[hub_id]
        free = [s for s in hub.station_ids if s not in self.occupancy[hub_id]]
        if not free:
            # parkingArea capacity equals the EVSE count, so this is a broken invariant
            raise RuntimeError(f"{vid} parked at {hub_id} but no EVSE is free")
        station = free[0]
        ev_cfg = self.cfg["ev"]
        trip = self.cfg["trip_types"].get(trip_type, {})
        dwell_s = (dwell_s if dwell_s is not None else trip["dwell_s"]) * self.dwell_scale
        # a top-up sized to the session (or trip type), never beyond the target SOC
        requested_kwh = max(0.0, ev_cfg["target_soc"] * capacity_wh - charge_wh) / 1000.0
        request = request_kwh if request_kwh is not None else trip.get("request_kwh")
        if request is not None:
            requested_kwh = min(requested_kwh, float(request))
        arrival_k = k + 1
        session = Session(
            vehicle_id=vid, hub=hub_id, station_id=station, trip_type=trip_type,
            arrival_k=arrival_k, departure_k=arrival_k + self.clock.duration_to_fine(dwell_s),
            capacity_wh=capacity_wh, charge_at_plugin_wh=charge_wh, requested_kwh=requested_kwh,
        )
        evse_kw = hub.max_rate_a * hub.voltage_v / 1000.0
        hub.plug_in(
            session_id=vid, station_id=station, arrival_k=arrival_k, departure_k=session.departure_k,
            init_kwh=charge_wh / 1000.0, requested_kwh=requested_kwh,
            capacity_kwh=capacity_wh / 1000.0, max_power_kw=min(trip.get("max_kw", ev_cfg.get("max_kw", ev_cfg["obc_kw"])), evse_kw),
        )
        self.occupancy[hub_id][station] = vid
        self.active[vid] = session
        return session

    def due_releases(self):
        """Sessions to release after the ACN-Sim step that just ran (iteration already advanced)."""
        tol = self.cfg["bridge"]["release_tol_kwh"]
        due = []
        for vid in sorted(self.active):
            s = self.active[vid]
            hub = self.hubs[s.hub]
            if hub.iteration <= s.arrival_k:
                continue  # ACN-Sim has not processed the plug-in yet
            on_complete = self.cfg["trip_types"].get(s.trip_type, {}).get("release_on_complete", self.release_on_complete)
            if hub.remaining_kwh(vid) <= tol and on_complete:
                due.append((s, "completed"))
            elif hub.iteration >= s.departure_k:
                due.append((s, "departure"))
        return due

    def release(self, session, reason):
        hub = self.hubs[session.hub]
        ev = hub.unplug(session.vehicle_id)  # frees the EVSE now; ACN-Sim's own unplug at departure becomes a no-op
        session.delivered_kwh = float(ev.energy_delivered)
        session.released_k = hub.iteration
        session.released_reason = reason
        del self.occupancy[session.hub][session.station_id]
        del self.active[session.vehicle_id]
        self.finished.append(session)
        return session

    def new_charge_wh(self, session):
        return min(session.capacity_wh, session.charge_at_plugin_wh + session.delivered_kwh * 1000.0)
