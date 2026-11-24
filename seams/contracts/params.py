import math
from dataclasses import dataclass, field
from fractions import Fraction

import yaml

from seams.cosim.config import resolve_path

DENOM = 10 ** 6


def outward(x, conservative="down"):
    scaled = Fraction(x) * DENOM
    n = math.floor(scaled) if conservative == "down" else math.ceil(scaled)
    return Fraction(n, DENOM)


@dataclass(frozen=True)
class Theta:
    version: str
    lam_bar: dict      # hub -> {window w: max arrivals over w periods}    (transport guarantee)
    P_bar: dict        # party -> max power [kW]: each hub at its PCC, and HOME (the aggregator's
                       # home portfolio on the feeder, AMI total)           (charging guarantee)
    V_lo: Fraction     # reported-voltage band [pu]                         (grid guarantee)
    V_hi: Fraction
    beta: dict         # hub -> {window w: max |physics residual| [pu]}     (grid assumption bound)
    # Assumption side of each seam (what the consumer assumes); defaults to the supplier's
    # guarantee. V1 checks guarantee => assumption, which matters once parties propose separately.
    lam_assumed: dict = None   # charging's assumption on arrivals
    P_assumed: dict = None     # grid's assumption on hub power
    V_lo_assumed: Fraction = None
    V_hi_assumed: Fraction = None
    # epsilon (calibrated, rounded up): D[h][k] drift envelope, lin[h], P_load_max[h]
    eps: dict = field(default=None, compare=False)
    meta: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, d):
        lam = {h: {int(w): outward(v, "down") for w, v in ws.items()} for h, ws in d["lam_bar"].items()}
        P = {h: outward(v, "down") for h, v in d["P_bar"].items()}
        # a guarantee's upper bound rounds down; an assumption's tolerance rounds up
        V_lo, V_hi = outward(d["V_lo"], "up"), outward(d["V_hi"], "down")
        eps = None
        if d.get("epsilon"):
            e = d["epsilon"]
            eps = {
                "D": {h: {int(k): outward(v, "up") for k, v in ks.items()} for h, ks in e["D"].items()},
                "lin": {h: outward(v, "up") for h, v in e["lin"].items()},
                "P_load_max": {h: outward(v, "down") for h, v in e["P_load_max"].items()},
                "cpl_V": {h: outward(v, "up") for h, v in e.get("cpl_V", {}).items()},
                "V_ceiling": {h: outward(v, "up") for h, v in e.get("V_ceiling", {}).items()},
                "gamma": {h: outward(v, "up") for h, v in e.get("gamma", {}).items()},
            }
        return cls(
            version=str(d["version"]), lam_bar=lam, P_bar=P, V_lo=V_lo, V_hi=V_hi,
            beta={h: {int(w): outward(b, "up") for w, b in ws.items()} for h, ws in d["beta"].items()},
            lam_assumed={h: {int(w): outward(v, "up") for w, v in ws.items()} for h, ws in d["lam_assumed"].items()}
            if d.get("lam_assumed") else lam,
            P_assumed={h: outward(v, "up") for h, v in d["P_assumed"].items()} if d.get("P_assumed") else P,
            V_lo_assumed=outward(d["V_lo_assumed"], "down") if "V_lo_assumed" in d else V_lo,
            V_hi_assumed=outward(d["V_hi_assumed"], "up") if "V_hi_assumed" in d else V_hi,
            eps=eps, meta=d.get("meta", {}),
        )

    def to_dict(self):
        fr = lambda x: f"{x.numerator}/{x.denominator}"
        out = {
            "version": self.version,
            "lam_bar": {h: {w: fr(v) for w, v in ws.items()} for h, ws in self.lam_bar.items()},
            "P_bar": {h: fr(v) for h, v in self.P_bar.items()},
            "V_lo": fr(self.V_lo), "V_hi": fr(self.V_hi),
            "beta": {h: {w: fr(v) for w, v in ws.items()} for h, ws in self.beta.items()},
            "lam_assumed": {h: {w: fr(v) for w, v in ws.items()} for h, ws in self.lam_assumed.items()},
            "P_assumed": {h: fr(v) for h, v in self.P_assumed.items()},
            "V_lo_assumed": fr(self.V_lo_assumed), "V_hi_assumed": fr(self.V_hi_assumed),
        }
        if self.eps:
            out["epsilon"] = {
                "D": {h: {k: fr(v) for k, v in ks.items()} for h, ks in self.eps["D"].items()},
                "lin": {h: fr(v) for h, v in self.eps["lin"].items()},
                "P_load_max": {h: fr(v) for h, v in self.eps["P_load_max"].items()},
                "cpl_V": {h: fr(v) for h, v in self.eps.get("cpl_V", {}).items()},
                "V_ceiling": {h: fr(v) for h, v in self.eps.get("V_ceiling", {}).items()},
                "gamma": {h: fr(v) for h, v in self.eps.get("gamma", {}).items()},
            }
        return out

    def replace(self, version, **changes):
        import dataclasses
        return dataclasses.replace(self, version=version, **changes)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def permissive(cls, hubs, windows, parties=None):
        big = 10 ** 6
        return cls.from_dict({
            "version": "calibration", "lam_bar": {h: {w: big for w in windows} for h in hubs},
            "P_bar": {j: big for j in (parties or hubs)},
            "V_lo": 0.0, "V_hi": 2.0, "beta": {h: {w: big for w in windows} for h in hubs},
        })

    def floats(self):
        return {
            "lam_bar": {h: {w: float(v) for w, v in ws.items()} for h, ws in self.lam_bar.items()},
            "P_bar": {h: float(v) for h, v in self.P_bar.items()},
            "V_lo": float(self.V_lo), "V_hi": float(self.V_hi),
            "beta": {h: {w: float(b) for w, b in ws.items()} for h, ws in self.beta.items()},
            # large-signal term of the physics guarantee [pu per kW of |dP|] (0 before Phase 4)
            "gamma": {h: float(v) for h, v in ((self.eps or {}).get("gamma") or {}).items()},
        }


def load_theta(cfg):
    c = cfg["contracts"]
    hubs = [h["id"] for h in cfg["hubs"]]
    parties = hubs + (["HOME"] if cfg.get("grid", {}).get("home_ev", {}).get("enabled") else [])
    path = resolve_path(c["theta_file"]) if c.get("theta_file") else None
    if c.get("force_calibration") or path is None or not path.exists():
        return Theta.permissive(hubs, c["windows"], parties)
    theta = Theta.load(path)
    scale = c.get("conservatism", 1.0)
    return scaled(theta, scale) if scale != 1.0 else theta


def scaled(theta, c):
    c = Fraction(str(c))
    lam = {h: {w: outward(v * c, "down") for w, v in ws.items()} for h, ws in theta.lam_bar.items()}
    P = {h: outward(v * c, "down") for h, v in theta.P_bar.items()}
    return theta.replace(f"{theta.version}-c{c}", lam_bar=lam, P_bar=P, lam_assumed=lam, P_assumed=P)
