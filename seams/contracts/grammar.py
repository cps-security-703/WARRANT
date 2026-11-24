from dataclasses import asdict, dataclass
from fractions import Fraction

import z3

from seams.contracts.verifier import Q


def dec(x):
    return x if isinstance(x, Fraction) else Fraction(str(x))


@dataclass(frozen=True)
class TimeWindowedCap:
    hub: str
    param: str          # "P_bar" | "lam_bar"
    value: float
    k_start: int
    k_end: int          # inclusive


@dataclass(frozen=True)
class HubExclusion:
    hub: str
    k_start: int
    k_end: int


@dataclass(frozen=True)
class CurtailmentWindow:
    hub: str
    k_start: int
    schedule: tuple     # P_bar for leads k_start, k_start + 1, ...


@dataclass(frozen=True)
class GridDistrust:
    hub: str
    eps_widen: float    # pu added to the drift bound at every lead


@dataclass(frozen=True)
class ValidityWindow:
    leads: int          # certified (and enforced) for leads 1..leads


TEMPLATES = {c.__name__: c for c in (TimeWindowedCap, HubExclusion, CurtailmentWindow, GridDistrust, ValidityWindow)}


class Schedule:
    def __init__(self, theta, K):
        self.K = K
        self.hubs = sorted(theta.P_bar)  # power parties: the hubs, and HOME if modeled
        self.P = {h: {k: Fraction(theta.P_bar[h]) for k in range(1, K + 1)} for h in self.hubs}
        # arrival quotas exist for hubs only (HOME has no parking entrance)
        self.L = {h: {k: Fraction(theta.lam_bar[h][1]) for k in range(1, K + 1)} for h in sorted(theta.lam_bar)}
        self.X = {h: Fraction(0) for h in self.hubs}
        self.valid = K   # leads the contract is certified and enforced for (ValidityWindow)

    def P_bar_k(self):
        return {(h, k): v for h, ks in self.P.items() for k, v in ks.items()}

    def extra(self):
        return {h: {k: self.X[h] for k in range(1, self.K + 1)} for h in self.hubs}

    def as_tuple(self):
        return (tuple((h, k, v) for h in self.hubs for k, v in sorted(self.P[h].items())),
                tuple((h, k, v) for h in sorted(self.L) for k, v in sorted(self.L[h].items())),
                tuple((h, self.X[h]) for h in self.hubs))


def _leads(t, K):
    return range(max(1, t.k_start), min(K, t.k_end) + 1)


def realize(templates, theta, K):
    s = Schedule(theta, K)
    for t in templates:
        if isinstance(t, TimeWindowedCap):
            table = s.P if t.param == "P_bar" else s.L
            for k in _leads(t, K):
                table[t.hub][k] = min(table[t.hub][k], dec(t.value))
        elif isinstance(t, HubExclusion):
            for k in _leads(t, K):
                s.L[t.hub][k] = Fraction(0)
        elif isinstance(t, CurtailmentWindow):
            for i, v in enumerate(t.schedule):
                k = t.k_start + i
                if 1 <= k <= K:
                    s.P[t.hub][k] = min(s.P[t.hub][k], dec(v))
        elif isinstance(t, GridDistrust):
            s.X[t.hub] = max(s.X[t.hub], dec(t.eps_widen))
        elif isinstance(t, ValidityWindow):
            s.valid = min(s.valid, int(t.leads))
        else:
            raise TypeError(f"not a template: {t!r}")
    return s


def encode(templates, theta, K):
    hubs, lam_hubs = sorted(theta.P_bar), sorted(theta.lam_bar)
    Pb = {h: {k: z3.Real(f"Pb_{h}_{k}") for k in range(1, K + 1)} for h in hubs}
    Lb = {h: {k: z3.Real(f"Lb_{h}_{k}") for k in range(1, K + 1)} for h in lam_hubs}
    Xd = {h: z3.Real(f"Xd_{h}") for h in hubs}
    bounds = {("P", h, k): [Fraction(theta.P_bar[h])] for h in hubs for k in range(1, K + 1)}
    bounds.update({("L", h, k): [Fraction(theta.lam_bar[h][1])] for h in lam_hubs for k in range(1, K + 1)})
    widen = {h: [Fraction(0)] for h in hubs}
    for t in templates:
        if isinstance(t, TimeWindowedCap):
            for k in _leads(t, K):
                bounds[("P" if t.param == "P_bar" else "L", t.hub, k)].append(dec(t.value))
        elif isinstance(t, HubExclusion):
            for k in _leads(t, K):
                bounds[("L", t.hub, k)].append(Fraction(0))
        elif isinstance(t, CurtailmentWindow):
            for i, v in enumerate(t.schedule):
                if 1 <= t.k_start + i <= K:
                    bounds[("P", t.hub, t.k_start + i)].append(dec(v))
        elif isinstance(t, GridDistrust):
            widen[t.hub].append(dec(t.eps_widen))
        # ValidityWindow: horizon only (realize sets Schedule.valid), nothing to encode
    cons = []
    for (kind, h, k), bs in bounds.items():
        var = Pb[h][k] if kind == "P" else Lb[h][k]
        cons += [var <= Q(b) for b in bs] + [z3.Or([var == Q(b) for b in bs])]
    for h, ws in widen.items():
        cons += [Xd[h] >= Q(w) for w in ws] + [z3.Or([Xd[h] == Q(w) for w in ws])]
    return (Pb, Lb, Xd), cons


def decode(model, variables, theta, K):
    Pb, Lb, Xd = variables
    s = Schedule(theta, K)
    val = lambda v: Fraction(str(model.eval(v, model_completion=True).as_fraction()))
    for h in s.hubs:
        for k in range(1, K + 1):
            s.P[h][k] = val(Pb[h][k])
            if h in s.L:
                s.L[h][k] = val(Lb[h][k])
        s.X[h] = val(Xd[h])
    return s


def round_trip(templates, theta, K):
    variables, cons = encode(templates, theta, K)
    solver = z3.Solver()
    solver.add(cons)
    if solver.check() != z3.sat:
        return False
    return decode(solver.model(), variables, theta, K).as_tuple() == realize(templates, theta, K).as_tuple()


# --- JSON proposals (Phase 8) ---------------------------------------------------
def to_json(t):
    return {"template": type(t).__name__, **{k: list(v) if isinstance(v, tuple) else v for k, v in asdict(t).items()}}


def from_json(d, hubs, K, lam_hubs=None, min_leads=1):
    if not isinstance(d, dict) or d.get("template") not in TEMPLATES:
        raise ValueError(f"unknown template: {d!r}")
    cls = TEMPLATES[d["template"]]
    fields = {k: v for k, v in d.items() if k != "template"}
    try:
        if cls is CurtailmentWindow:
            fields["schedule"] = tuple(float(x) for x in fields["schedule"])
        t = cls(**fields)
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError(f"malformed {d['template']}: {e}") from e
    if isinstance(t, ValidityWindow):
        if not isinstance(t.leads, int) or not (min_leads <= t.leads <= K):
            raise ValueError(f"validity window must be {min_leads}..{K} leads")
        return t
    if t.hub not in hubs:
        raise ValueError(f"unknown hub {t.hub!r}")
    on_arrivals = isinstance(t, HubExclusion) or (isinstance(t, TimeWindowedCap) and t.param == "lam_bar")
    if on_arrivals and t.hub not in (hubs if lam_hubs is None else lam_hubs):
        raise ValueError(f"{t.hub} has no arrival quota")
    if isinstance(t, TimeWindowedCap) and (t.param not in ("P_bar", "lam_bar") or t.value < 0):
        raise ValueError("bad cap")
    if hasattr(t, "k_start") and not (1 <= t.k_start <= K):
        raise ValueError("window outside the horizon")
    if isinstance(t, GridDistrust) and not (0 <= t.eps_widen <= 0.1):
        raise ValueError("eps_widen out of range")
    return t
