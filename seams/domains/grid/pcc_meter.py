"""Trusted PCC meter (part of the trusted computing base): hub power at the point of common
coupling. It measures what the chargers actually drew, never what the scheduler claims."""


class PccMeter:
    def __init__(self, hubs):
        self.p_kw = {h: 0.0 for h in hubs}

    def measure(self, acn_hubs):
        self.p_kw = {h: hub.power_kw() for h, hub in acn_hubs.items()}
        return dict(self.p_kw)
