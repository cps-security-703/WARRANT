"""Home EV charging on the feeder, managed by the same aggregator cloud as the hubs.

Real aggregators and OEM clouds also control residential chargers on the feeder; that portfolio
is what makes a compromised aggregator a feeder-scale threat (load-altering attacks:
Mohsenian-Rad & Leon-Garcia 2011; Soltan et al., USENIX Security 2018). It is part of the
charging domain, not of the background load.

Model (grid.home_ev): `chargers` home L2 chargers, each attached to one feeder load element
with probability proportional to that element's nominal kW. Each charger sees
Poisson(sessions_per_charger_day) weekday sessions drawn from real residential sessions
(EV WATTS, Pacific, Single Family Residential): plug-in time, dwell, energy and the charger's
own average power. Honest charging starts at plug-in at that power until the energy is
delivered or the car leaves (the data's own timing already includes TOU-delayed starts).
Aggregator modes (attacks only): hold (all sessions paused) and release (a share at the
charger's max_kw, ignoring everything else).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from seams.scenarios.build_scenario import PROCESSED


@dataclass
class HomeSession:
    element: str
    arr_s: float
    dep_s: float
    kwh: float
    rate_kw: float
    delivered: float = 0.0

    def remaining(self):
        return max(0.0, self.kwh - self.delivered)


class HomeCharging:
    def __init__(self, cfg, elements, rng):
        """elements: {load element name: nominal kW} of the feeder's background loads."""
        h = cfg["grid"]["home_ev"]
        self.max_kw = float(h.get("max_kw", 7.2))
        self.dt_s = float(cfg["clock"]["fine_period_s"])
        names = sorted(elements)
        w = np.asarray([elements[n] for n in names], dtype=float)
        pool = pd.read_parquet(PROCESSED / "sessions.parquet")
        pool = pool[pool.profile == "home"].reset_index(drop=True)
        self.sessions = []
        self.charger_counts = {n: 0 for n in names}
        for _ in range(int(h["chargers"])):
            element = names[int(rng.choice(len(names), p=w / w.sum()))]
            self.charger_counts[element] += 1
            for j in rng.choice(len(pool), size=rng.poisson(float(h["sessions_per_charger_day"]))):
                r = pool.iloc[int(j)]
                self.sessions.append(HomeSession(element, float(r.arr_s), float(r.arr_s + r.dwell_s),
                                                 float(r.kwh), min(float(r.rate_kw), self.max_kw)))
        self.elements = names
        self.mode, self.fraction = "honest", 1.0
        self.kw = {n: 0.0 for n in names}
        self.requested_kw = 0.0

    def set_mode(self, mode, fraction=1.0):
        """honest | hold | release (compromised aggregator)."""
        self.mode, self.fraction = mode, float(fraction)

    def step(self, t_s, comply_kw=None, cap_kw=None):
        """Power per load element for the fine period starting at t_s [kW]; advances energy.
        comply_kw: the limit an honest aggregator keeps its own schedule to (its contracted
        P_bar; a compromised one ignores it). cap_kw: the trusted demand-response cap, which
        binds whatever the aggregator schedules. requested_kw is the aggregator's schedule
        (after its own compliance, before the trusted cap), as P_req is for a hub."""
        kw = {n: 0.0 for n in self.elements}
        active = [s for s in self.sessions if s.arr_s <= t_s < s.dep_s and s.remaining() > 1e-6]
        released = set()
        if self.mode == "release":
            n = int(np.ceil(self.fraction * len(active)))
            released = {id(s) for s in sorted(active, key=lambda s: (s.element, s.arr_s))[:n]}
        hours = self.dt_s / 3600.0
        want = []
        for s in active:
            if self.mode == "hold":
                p = 0.0
            elif id(s) in released:
                p = self.max_kw
            else:
                p = s.rate_kw
            want.append((s, min(p, s.remaining() / hours)))
        total = float(sum(p for _, p in want))
        if self.mode == "honest" and comply_kw is not None and total > comply_kw:
            want = [(s, p * max(0.0, comply_kw) / total) for s, p in want]  # smart charging
        self.requested_kw = float(sum(p for _, p in want))
        scale = 1.0
        if cap_kw is not None and self.requested_kw > cap_kw:
            scale = max(0.0, cap_kw) / self.requested_kw
        for s, p in want:
            p *= scale
            s.delivered += p * hours
            kw[s.element] += p
        self.kw = kw
        return kw

    def total_kw(self):
        return float(sum(self.kw.values()))

    def connected(self, t_s):
        return sum(1 for s in self.sessions if s.arr_s <= t_s < s.dep_s)
