import time
from dataclasses import dataclass
from fractions import Fraction

import z3


def Q(x):
    x = Fraction(x)
    return z3.RealVal(f"{x.numerator}/{x.denominator}")


@dataclass
class Verdict:
    ok: bool
    check: str
    solve_ms: float
    counterexample: dict = None


def _cex(model, variables):
    out = {}
    for name, var in variables.items():
        val = model.eval(var, model_completion=True)
        out[name] = float(Fraction(str(val.as_fraction()))) if z3.is_rational_value(val) else str(val)
    return out


def v1_compatibility(theta, windows, horizon):
    t0 = time.perf_counter()
    K = int(horizon)
    for h in sorted(theta.lam_bar):
        # transport -> charging, windowed arrivals
        s = z3.Solver()
        lam = [z3.Real(f"lam_{h}_{k}") for k in range(K)]
        s.add(*[x >= 0 for x in lam])
        viol = []
        for w in windows:
            for k in range(w - 1, K):
                tot = z3.Sum(lam[k - w + 1:k + 1])
                s.add(tot <= Q(theta.lam_bar[h][w]))
                viol.append(tot > Q(theta.lam_assumed[h][w]))
        s.add(z3.Or(viol))
        if s.check() == z3.sat:
            return Verdict(False, f"V1:transport->charging:{h}", _ms(t0), _cex(s.model(), dict(enumerate(lam))))
        # charging -> grid, hub power
        s = z3.Solver()
        P = [z3.Real(f"P_{h}_{k}") for k in range(K)]
        s.add(*[p <= Q(theta.P_bar[h]) for p in P])
        s.add(z3.Or([p > Q(theta.P_assumed[h]) for p in P]))
        if s.check() == z3.sat:
            return Verdict(False, f"V1:charging->grid:{h}", _ms(t0), _cex(s.model(), dict(enumerate(P))))
    # grid -> charging, reported-voltage band; A_C at k+1 depends on G_G at k
    s = z3.Solver()
    V = [z3.Real(f"V_{k}") for k in range(K)]
    s.add(*[z3.And(v >= Q(theta.V_lo), v <= Q(theta.V_hi)) for v in V])
    s.add(z3.Or([z3.Or(V[k] < Q(theta.V_lo_assumed), V[k] > Q(theta.V_hi_assumed)) for k in range(K - 1)]))
    if s.check() == z3.sat:
        return Verdict(False, "V1:grid->charging", _ms(t0), _cex(s.model(), dict(enumerate(V))))
    return Verdict(True, "V1", _ms(t0))


def v2_refinement(theta, anchor, eps, phi, horizon, use_epsilon=True, P_bar_k=None, extra=None):
    t0 = time.perf_counter()
    hubs = sorted(anchor["V"])     # voltage-constrained parties
    parties = sorted(anchor["P"])  # power parties: the hubs, plus HOME (the home portfolio)
    over = [h for h in hubs if Fraction(eps["V_ceiling"][h]) > Fraction(phi["v_max"])]
    if use_epsilon and over:
        return Verdict(False, "V2:ceiling", _ms(t0), {f"V_ceiling_{h}": float(eps["V_ceiling"][h]) for h in over})
    s = z3.Solver()
    viol = []  # (label, expression, {name: var}) per possible violation
    v_min = Q(phi["v_min"])
    for k in range(1, int(horizon) + 1):
        P = {j: z3.Real(f"P_{j}_{k}") for j in parties}
        for j in parties:
            cap = (P_bar_k or {}).get((j, k), theta.P_bar[j])
            s.add(P[j] >= 0, P[j] <= Q(cap))
            if j in eps["P_load_max"]:  # hubs have a service transformer; HOME does not
                viol.append((f"loading:{j}:k{k}", P[j] > Q(eps["P_load_max"][j]), {f"P_{j}": P[j]}))
        for h in hubs:
            e = z3.Real(f"e_{h}_{k}")
            bound = (eps["D"][h][k] + eps["lin"][h]) if use_epsilon else Fraction(0)
            bound += Fraction((extra or {}).get(h, {}).get(k, 0))
            s.add(e >= -Q(bound), e <= 0)
            # measured floats convert to rationals exactly; no rounding needed
            V = Q(Fraction(anchor["V"][h])) + z3.Sum([
                Q(Fraction(anchor["S"][h][j])) * (P[j] - Q(Fraction(anchor["P"][j]))) for j in parties
            ]) + e
            viol.append((f"undervoltage:{h}:k{k}", V < v_min, {**{f"P_{j}": P[j] for j in parties}, "e": e, "V": V}))
    s.add(z3.Or([expr for _, expr, _ in viol]))
    res = s.check()
    if res == z3.unsat:
        return Verdict(True, "V2", _ms(t0))
    if res != z3.sat:
        return Verdict(False, "V2", _ms(t0), {"z3": str(res)})
    m = s.model()
    # report the first violated condition with its witness, not the whole model
    for label, expr, wit in viol:
        if z3.is_true(m.eval(expr, model_completion=True)):
            return Verdict(False, "V2", _ms(t0), {"violation": label, **_cex(m, wit)})
    return Verdict(False, "V2", _ms(t0), {"z3": "sat"})


def _ms(t0):
    return 1000.0 * (time.perf_counter() - t0)
