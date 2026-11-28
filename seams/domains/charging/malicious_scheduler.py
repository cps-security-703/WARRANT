"""Compromised aggregator: requests the maximum rate on a fraction of connected EVSEs and
ignores derating and infrastructure limits.

It runs inside the charging domain, so ACN-Sim's own constraint check is compromised with it
(ACN-Sim only warns on an infeasible schedule and applies it anyway). Only the external PCC
limiter, outside the domain's software, stands between it and the feeder.
"""
import math

from acnportal.algorithms import BaseAlgorithm


class MaliciousScheduler(BaseAlgorithm):
    def __init__(self, fraction):
        super().__init__()
        self.fraction = float(fraction)
        self.max_recompute = 1

    def schedule(self, active_sessions):
        sessions = sorted(active_sessions, key=lambda s: s.station_id)
        n_max = math.ceil(self.fraction * len(sessions))
        out = {}
        for i, s in enumerate(sessions):
            if i < n_max:
                out[s.station_id] = [self.interface.max_pilot_signal(s.station_id)]
            else:
                out[s.station_id] = [0.0]
        return out
