"""ACN-Sim adapter: one charging network per hub, stepped one fine period at a time.

Verified against acnportal 0.3.3:
- Simulator.step() only advances while the event queue is non-empty, so it stalls between
  dynamic arrivals. step() below mirrors one iteration of Simulator.run() instead
  (events -> schedule -> pilots -> charge), in run()'s order.
- network.update_constraint() changes a limit mid-run; that is the derating path.
- An infeasible schedule is only warned about and still applied: ACN-Sim does not enforce
  its constraints physically. The external PCC limiter is the real protection.
"""
import warnings

import numpy as np
from acnportal import acnsim
from acnportal.acnsim.interface import Interface
from acnportal.acnsim.simulator import _increase_width
from acnportal.algorithms import (
    RoundRobin, SortedSchedulingAlgo, UncontrolledCharging,
    earliest_deadline_first, least_laxity_first,
)

AGGREGATE = "hub_aggregate"


def make_scheduler(name):
    if name == "EDF":
        return SortedSchedulingAlgo(earliest_deadline_first)
    if name == "LLF":
        return SortedSchedulingAlgo(least_laxity_first)
    if name == "RR":
        return RoundRobin(earliest_deadline_first)
    if name == "UNCONTROLLED":
        return UncontrolledCharging()
    raise ValueError(f"unknown scheduler {name!r}")


class AcnHub:
    def __init__(self, hub_id, n_evse, evse_cfg, period_min, scheduler=None):
        self.hub_id = hub_id
        self.voltage_v = float(evse_cfg["voltage_v"])
        self.max_rate_a = float(evse_cfg["max_rate_a"])
        self.station_ids = [f"{hub_id}-{i:02d}" for i in range(n_evse)]
        self.network = acnsim.ChargingNetwork(violation_tolerance=1e-4, relative_tolerance=1e-7)
        for sid in self.station_ids:
            self.network.register_evse(acnsim.EVSE(sid, max_rate=self.max_rate_a), self.voltage_v, 0)
        self.nameplate_limit_a = float(evse_cfg["hub_aggregate_limit_a"])
        self.network.add_constraint(acnsim.Current(self.station_ids), self.nameplate_limit_a, name=AGGREGATE)
        self.scheduler = scheduler or make_scheduler(evse_cfg["scheduler"])
        self.sim = acnsim.Simulator(
            self.network, self.scheduler, acnsim.EventQueue(),
            start=None, period=period_min, verbose=False,
        )
        self.evs = {}  # session_id -> acnsim.EV
        self.limit_a = self.nameplate_limit_a
        self.last_rates_a = np.zeros(n_evse)
        self.last_requested_kw = 0.0

    @property
    def iteration(self):
        return self.sim._iteration

    def plug_in(self, session_id, station_id, arrival_k, departure_k, init_kwh, requested_kwh,
                capacity_kwh, max_power_kw):
        if arrival_k < self.iteration:
            raise ValueError(f"plug-in at {arrival_k} is before ACN-Sim iteration {self.iteration}")
        battery = acnsim.Linear2StageBattery(capacity=capacity_kwh, init_charge=init_kwh, max_power=max_power_kw)
        ev = acnsim.EV(arrival_k, departure_k, requested_kwh, station_id, session_id, battery)
        self.sim.event_queue.add_event(acnsim.PluginEvent(arrival_k, ev))
        self.evs[session_id] = ev

    def unplug(self, session_id):
        ev = self.evs.pop(session_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.network.unplug(ev.station_id, session_id)
        return ev

    def set_scheduler(self, scheduler):
        """Swap the scheduling algorithm mid-run (compromised aggregator and its recovery)."""
        scheduler.register_interface(Interface(self.sim))
        self.scheduler = scheduler
        self.sim.scheduler = scheduler

    def set_aggregate_limit(self, amps):
        """Derating: rewrite this hub's aggregate current limit before the next step."""
        amps = float(min(self.nameplate_limit_a, max(0.0, amps)))
        if amps != self.limit_a:
            self.network.update_constraint(AGGREGATE, acnsim.Current(self.station_ids), amps)
            self.limit_a = amps

    def step(self, schedule_filters=()):
        """Advance one fine period. Each filter (schedule -> schedule) acts on the physical path
        between scheduler and chargers: the external PCC limiter, or an open breaker."""
        sim, k = self.sim, self.sim._iteration
        with warnings.catch_warnings():
            # stale UnplugEvents for sessions the bridge already released
            warnings.filterwarnings("ignore", message="Tried to remove EV")
            for event in sim.event_queue.get_current_events(k):
                sim.event_history.append(event)
                sim._process_event(event)
        schedule = self.scheduler.run()
        # power requested at the PCC limiter input, before any physical filter (pre-shield signal)
        self.last_requested_kw = sum(r[0] for r in schedule.values() if r) * self.voltage_v / 1000.0
        for schedule_filter in schedule_filters:
            schedule = schedule_filter(schedule)
        sim.pilot_signals = _increase_width(sim.pilot_signals, k + 2)
        sim.charging_rates = _increase_width(sim.charging_rates, k + 2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Invalid schedule")
            sim._update_schedules(schedule)
        sim.network.update_pilots(sim.pilot_signals, k, sim.period)
        sim._store_actual_charging_rates()
        sim.network.post_charging_update()
        sim._iteration = k + 1
        self.last_rates_a = sim.charging_rates[:, k].copy()
        return schedule

    def open_breaker(self, schedule):
        """Schedule filter for a tripped hub: nothing reaches the chargers, whatever the
        scheduler asked for."""
        return {sid: [0.0] * len(rates) for sid, rates in schedule.items()}

    def power_kw(self):
        """Hub power drawn in the last step."""
        return float(self.last_rates_a.sum() * self.voltage_v / 1000.0)

    def delivered_kwh(self, session_id):
        return float(self.evs[session_id].energy_delivered)

    def remaining_kwh(self, session_id):
        ev = self.evs[session_id]
        return float(ev.requested_energy - ev.energy_delivered)
