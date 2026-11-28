"""Reported grid telemetry, kept apart from the true state so FDI is representable.

V_rep_h = V_true_h(k - latency) + measurement error + falsification_h, rounded to the meter's
resolution. Only the grid controller (derating) and the contract monitors read reported
values; Phi_safe and metrics read the true state.

Measurement error (grid.telemetry, all optional, default an ideal meter):
  noise_sigma_pu  Gaussian error per report. Revenue-class voltage measurement (ANSI C12.20
                  class 0.2/0.5, i.e. +-0.2-0.5 % treated as ~3 sigma) gives 0.0007-0.0017 pu.
  latency_steps   reports are this many fine steps old (SCADA scan + backhaul; 0 at 60 s steps)
  quantum_pu      reporting resolution (0 = none)
"""
from collections import deque

import numpy as np


class Telemetry:
    def __init__(self, hubs, cfg=None, rng=None):
        t = (cfg or {}).get("grid", {}).get("telemetry", {}) or {}
        self.falsification = {h: 0.0 for h in hubs}     # pu offset; the FDI injection point
        self.falsification_mv = {h: 0.0 for h in hubs}  # the feeder RTU (stealthy FDI only)
        self.stuck = {}  # hub -> frozen LV reading (meter_stuck disturbance; None = freeze at next report)
        self.sigma = float(t.get("noise_sigma_pu", 0.0))
        self.quantum = float(t.get("quantum_pu", 0.0))
        self.latency = int(t.get("latency_steps", 0))
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.rng_mv = self.rng.spawn(1)[0]  # own stream: the LV reports draw exactly as before
        self._past = deque(maxlen=self.latency + 1)
        self._past_mv = deque(maxlen=self.latency + 1)

    def report(self, true_v):
        """Hub LV voltages as reported (the derating and contract input)."""
        out = self._report(true_v, self._past, self.falsification, self.rng)
        for h, v in self.stuck.items():
            if v is None:
                self.stuck[h] = out[h]
            out[h] = self.stuck[h]
        return out

    def report_mv(self, true_v):
        """MV voltages at the hubs' feeder buses as reported by the feeder RTU, an independent
        meter with the same error model (used by bad-data detection)."""
        return self._report(true_v, self._past_mv, self.falsification_mv, self.rng_mv)

    def _report(self, true_v, past, falsification, rng):
        past.append(dict(true_v))
        measured = past[0]  # oldest kept = latency steps ago (or the first, at start)
        out = {}
        for h in sorted(measured):
            v = measured[h]
            if self.sigma:
                v += self.sigma * rng.standard_normal()
            if self.quantum:
                v = round(v / self.quantum) * self.quantum
            out[h] = v + falsification[h]
        return out
