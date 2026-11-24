class PccMeter:
    def __init__(self, hubs):
        self.p_kw = {h: 0.0 for h in hubs}

    def measure(self, acn_hubs):
        self.p_kw = {h: hub.power_kw() for h, hub in acn_hubs.items()}
        return dict(self.p_kw)
