class PccLimiter:
    def __init__(self, hub_id, voltage_v):
        self.hub_id = hub_id
        self.voltage_v = float(voltage_v)
        self.p_bar_kw = None   # contracted limit in force; None = not enforcing
        self.active = False    # clamped at the last step

    def set_limit(self, p_bar_kw):
        self.p_bar_kw = float(p_bar_kw)

    def __call__(self, schedule):
        self.active = False
        if self.p_bar_kw is None:
            return schedule
        total_kw = sum(r[0] for r in schedule.values() if r) * self.voltage_v / 1000.0
        if total_kw <= self.p_bar_kw:
            return schedule
        self.active = True
        scale = self.p_bar_kw / total_kw
        return {sid: [x * scale for x in rates] for sid, rates in schedule.items()}
