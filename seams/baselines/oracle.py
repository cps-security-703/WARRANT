"""Centralized oracle (upper bound on availability; not deployable across independent operators).

One controller sees all three domains' true state and dispatches charging and admission jointly:
  - routing sees true hub availability (spoofed routing inputs never reach it)
  - derating reads the true voltage (falsified telemetry never reaches it)
  - every fine step, each hub's (and the home portfolio's) power limit is the largest that keeps every hub's *true* voltage
    at or above v_min + margin under the quasi-steady-state sensitivity, capped at the
    transformer's 100 % loading power; it is enforced on the physical path (a malicious scheduler
    is clamped like any other)
It has no verification horizon and no contract conservatism: only the next step must be safe.
"""
from seams.contracts.params import Theta
from seams.cosim.config import resolve_path
from seams.shields.pcc_limiter import PccLimiter


class Oracle:
    def __init__(self, cfg, hubs, home=False):
        o = cfg["oracle"]
        theta = Theta.load(resolve_path(o["theta_file"]))
        eps = theta.eps  # P_load_max from calibration
        self.p_load_max = {h: float(eps["P_load_max"][h]) for h in hubs}
        if home:  # it also dispatches the home portfolio, up to its static range
            self.p_load_max["HOME"] = float(theta.P_bar["HOME"])
        self.v_floor = float(cfg["grid"]["phi_safe"]["v_min"]) + float(o["margin_pu"])
        self.every = max(1, round(float(o["sensitivity_every_s"]) / float(cfg["clock"]["fine_period_s"])))
        self.limit = dict(self.p_load_max)
        self.limiters = {h: PccLimiter(h, hub.voltage_v) for h, hub in hubs.items()}  # hubs only
        for h, lim in self.limiters.items():
            lim.set_limit(self.limit[h])
        self.Sq = None

    def update(self, k, grid, state, p_now):
        """After the grid solve of step k: limits for step k + 1 from the true state."""
        if self.Sq is None or k % self.every == 0:
            self.Sq = grid.sensitivity_qss(self.p_load_max)
        for h in self.limit:
            room = []
            for g in self.limiters:  # voltage is constrained at the hubs
                s = self.Sq[g][h]
                if s < 0:
                    room.append((state.v_lv[g] - self.v_floor) / -s)
            head = min(room) if room else float("inf")
            self.limit[h] = max(0.0, min(self.p_load_max[h], p_now[h] + head))
            if h in self.limiters:
                self.limiters[h].set_limit(self.limit[h])
