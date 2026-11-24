import numpy as np

from seams.cosim.config import PROJECT_ROOT

LOAD_LA = PROJECT_ROOT / "seams" / "data" / "processed" / "load_la.parquet"


def candidate_days(b):
    import pandas as pd
    d = pd.read_parquet(LOAD_LA)
    row = d[d.month.isin(b.get("months", list(range(1, 13))))]
    if b.get("weekdays_only", True):
        row = row[row.weekday]
    if b.get("peak_quantile_max") is not None:  # typical days only: drop the hottest ones
        peak = row.filter(regex="^q[0-9]").max(axis=1)
        row = row[peak <= peak.quantile(float(b["peak_quantile_max"]))]
    return row


def envelope_top(b, n_sigma=4.0):
    if b.get("source", "hourly") != "nrel_la":
        top = max(b["hourly_mult"])
    else:
        top = float(candidate_days(b).filter(regex="^q[0-9]").to_numpy().max()) * float(b.get("peak_mult", 1.0))
    return top * (1.0 + n_sigma * float(b.get("noise_sigma", 0.0)))


def nrel_day(b, rng):
    import pandas as pd
    if b.get("day"):
        d = pd.read_parquet(LOAD_LA)
        row = d[d.date == str(b["day"])]
        if row.empty:
            raise ValueError(f"background.day {b['day']} not in {LOAD_LA.name}")
    else:
        row = candidate_days(b)
        row = row.iloc[[int(rng.integers(len(row)))]]
    temps = row.filter(regex="^t[0-9]").to_numpy()[0] if any(c.startswith("t0") for c in row.columns) else None
    return str(row.date.iloc[0]), row.filter(regex="^q[0-9]").to_numpy()[0] * float(b.get("peak_mult", 1.0)), temps


class BackgroundLoad:
    def __init__(self, cfg, rng):
        b = cfg["grid"]["background"]
        if b.get("source", "hourly") == "nrel_la":
            self.day, self.profile, self.temps = nrel_day(b, rng)
        else:
            self.day, self.profile, self.temps = None, np.asarray(b["hourly_mult"], dtype=float), None
            if len(self.profile) != 24:
                raise ValueError("background.hourly_mult needs 24 values")
        self.slot_s = 86400.0 / len(self.profile)
        self.sigma = float(b["noise_sigma"])
        self.phi = float(b["noise_ar1"])
        self.rng = rng
        self.noise = 0.0

    def shape(self, t_s):
        x = (t_s % 86400.0) / self.slot_s
        i = int(x)
        frac = x - i
        n = len(self.profile)
        return (1 - frac) * self.profile[i % n] + frac * self.profile[(i + 1) % n]

    def ambient_c(self, t_s):
        if self.temps is None:
            return None
        x = (t_s % 86400.0) / 3600.0
        i = int(x)
        return float((1 - (x - i)) * self.temps[i % 24] + (x - i) * self.temps[(i + 1) % 24])

    def step(self, t_s):
        innovation = self.rng.standard_normal() * self.sigma * np.sqrt(1 - self.phi ** 2)
        self.noise = self.phi * self.noise + innovation
        return self.shape(t_s) * (1.0 + self.noise)
