import math


class TransformerProtection:
    def __init__(self, cfg, fine_period_s):
        p = cfg["grid"]["protection"]
        self.enabled = bool(p.get("enabled", True))
        self.model = p.get("model", "threshold")
        self.dt = float(fine_period_s)
        self.offline_steps = max(1, round(p["offline_s"] / fine_period_s))
        self.offline_until = {}  # hub -> first fine step back online
        if self.model == "threshold":
            self.trip_pct = float(p["trip_loading_pct"])
            self.trip_steps = max(1, round(p["trip_after_s"] / fine_period_s))
            self.over = {}       # hub -> consecutive overloaded fine steps
        elif self.model == "c57_91":
            th = p.get("thermal", {})
            self.dTO_R = float(th.get("top_oil_rise_c", 55.0))
            self.dH_R = float(th.get("hotspot_gradient_c", 25.0))
            self.R = float(th.get("loss_ratio", 4.0))
            self.n = float(th.get("n", 0.8))
            self.m = float(th.get("m", 0.8))
            self.a_TO = 1.0 - math.exp(-self.dt / float(th.get("tau_top_oil_s", 7200.0)))
            self.a_H = 1.0 - math.exp(-self.dt / float(th.get("tau_winding_s", 300.0)))
            self.trip_c = float(p.get("trip_hotspot_c", 140.0))
            self.ambient_default = float(p.get("ambient_c", 30.0))
            self.dTO, self.dH = {}, {}  # hub -> current rises [C]
            self.hotspot = {}           # hub -> last hot-spot temperature [C]
        else:
            raise ValueError(f"unknown protection model {self.model!r}")

    def is_offline(self, hub, k):
        return k < self.offline_until.get(hub, -1)

    def _thermal(self, hub, pct, ambient_c):
        K = max(0.0, pct) / 100.0
        to_u = self.dTO_R * ((K * K * self.R + 1.0) / (self.R + 1.0)) ** self.n
        h_u = self.dH_R * K ** (2.0 * self.m)
        if hub not in self.dTO:  # start in steady state with the first load seen
            self.dTO[hub], self.dH[hub] = to_u, h_u
        self.dTO[hub] += (to_u - self.dTO[hub]) * self.a_TO
        self.dH[hub] += (h_u - self.dH[hub]) * self.a_H
        self.hotspot[hub] = ambient_c + self.dTO[hub] + self.dH[hub]
        return self.hotspot[hub]

    def update(self, k, loading, ambient_c=None):
        tripped = []
        if not self.enabled:
            return tripped
        for hub, pct in loading.items():
            if self.model == "c57_91":
                # the unit keeps heating or cooling while the breaker is open (it is unloaded)
                ambient = self.ambient_default if ambient_c is None else ambient_c
                hot = self._thermal(hub, 0.0 if self.is_offline(hub, k) else pct, ambient)
                if not self.is_offline(hub, k + 1) and hot > self.trip_c:
                    self.offline_until[hub] = k + 1 + self.offline_steps
                    tripped.append(hub)
                continue
            if self.is_offline(hub, k + 1):
                continue
            self.over[hub] = self.over.get(hub, 0) + 1 if pct > self.trip_pct else 0
            if self.over[hub] >= self.trip_steps:
                self.offline_until[hub] = k + 1 + self.offline_steps
                self.over[hub] = 0
                tripped.append(hub)
        return tripped
