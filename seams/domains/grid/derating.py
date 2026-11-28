"""Grid-side voltage-aware derating: d_h from the *reported* hub voltage.

Full allowance at or above v_knee, linear ramp down to p_min at v_lo, p_min below. Because it
reads reported values, a falsely low V_rep curtails needlessly and a falsely high one hides a
real undervoltage.
"""


class DeratingController:
    def __init__(self, cfg):
        d = cfg["grid"]["derating"]
        self.v_knee = float(d["v_knee"])
        self.v_lo = float(d["v_lo"])
        self.p_min_frac = float(d["p_min_frac"])
        self.p_full = {h["id"]: float(h["p_allow_kw"]) for h in cfg["hubs"]}

    def d_kw(self, hub, v_reported):
        full = self.p_full[hub]
        p_min = self.p_min_frac * full
        if v_reported >= self.v_knee:
            return full
        if v_reported <= self.v_lo:
            return p_min
        frac = (v_reported - self.v_lo) / (self.v_knee - self.v_lo)
        return p_min + frac * (full - p_min)

    def allowances(self, v_reported):
        return {h: self.d_kw(h, v) for h, v in v_reported.items()}
