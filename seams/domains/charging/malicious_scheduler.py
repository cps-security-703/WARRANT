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
