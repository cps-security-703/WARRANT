from dataclasses import dataclass
from fractions import Fraction

from seams.contracts import grammar
from seams.contracts.verifier import v1_compatibility, v2_refinement


@dataclass
class Context:
    kc: int
    base: object          # C^deg: Theta with degraded caps
    safe: object          # theta_safe
    anchor: dict          # {'V', 'P', 'S'} as for V2
    eps: dict
    phi: dict
    K: int
    windows: list
    extra: dict           # {hub: {k: pu}} distrust drift bound
    caps: dict            # {party: kW} degraded caps (C^deg's P_bar)
    floors: dict          # {party: kW} theta_safe's P_bar (the classical optimizer's floors)
    reason: str
    modes: tuple
    min_valid: int = 1    # shortest ValidityWindow an agent may propose


@dataclass
class GateResult:
    ok: bool
    check: str
    schedule: object = None
    cex: dict = None
    error: str = None


def lead_theta(base, sched, lead, version):
    P = {h: sched.P[h][lead] for h in sched.P}
    lam = {}
    for h, ws in base.lam_bar.items():
        q = sched.L[h][lead]
        lam[h] = {w: min(Fraction(v), q * w) for w, v in ws.items()}
    P_assumed = {h: max(sched.P[h].values()) for h in sched.P}
    return base.replace(version, P_bar=P, lam_bar=lam, lam_assumed=base.lam_assumed, P_assumed=P_assumed)


def shifted(sched, by, K):
    out = grammar.Schedule.__new__(grammar.Schedule)
    out.K, out.hubs = K, sched.hubs
    pick = lambda table: {h: {k: table[h][min(k + by, sched.K)] for k in range(1, K + 1)} for h in table}
    out.P, out.L, out.X = pick(sched.P), pick(sched.L), dict(sched.X)
    out.valid = max(0, getattr(sched, "valid", sched.K) - by)
    return out


def floored(sched, theta, base):
    out = shifted(sched, 0, sched.K)
    out.P = {h: {k: max(v, Fraction(theta.P_bar[h])) for k, v in ks.items()} for h, ks in sched.P.items()}
    out.L = {h: {k: max(v, Fraction(base.lam_bar[h][1])) for k, v in ks.items()} for h, ks in sched.L.items()}
    adds = any(sched.P[h][k] > Fraction(theta.P_bar[h]) for h in sched.P for k in range(1, out.valid + 1))
    return out if adds else None


def certify_schedule(ctx, sched, version="candidate"):
    Kv = getattr(sched, "valid", ctx.K)
    if Kv < 1:
        return GateResult(False, "expired", sched, None)
    theta = lead_theta(ctx.base, sched, 1, version)
    v1 = v1_compatibility(theta, ctx.windows, ctx.K)
    if not v1.ok:
        return GateResult(False, v1.check, sched, v1.counterexample)
    extra = {h: {k: Fraction(ctx.extra.get(h, {}).get(k, 0)) + sched.X.get(h, Fraction(0))
                 for k in range(1, ctx.K + 1)} for h in ctx.anchor["V"]}
    v2 = v2_refinement(theta, ctx.anchor, ctx.eps, ctx.phi, Kv, P_bar_k=sched.P_bar_k(), extra=extra)
    if not v2.ok:
        return GateResult(False, "V2", sched, v2.counterexample)
    if not v2_refinement(ctx.safe, ctx.anchor, ctx.eps, ctx.phi, Kv, extra=ctx.extra).ok:
        return GateResult(False, "exit-aware: theta_safe unreachable", sched, None)
    return GateResult(True, "certified", sched)


def parse(proposal, ctx):
    lam_hubs = sorted(ctx.base.lam_bar)
    return [grammar.from_json(t, sorted(ctx.base.P_bar), ctx.K, lam_hubs, ctx.min_valid) for t in proposal]


def certify(ctx, templates, version="candidate"):
    return certify_schedule(ctx, grammar.realize(templates, ctx.base, ctx.K), version)


def from_theta(theta, ctx):
    s = grammar.Schedule(ctx.base, ctx.K)
    for h in s.P:
        for k in s.P[h]:
            s.P[h][k] = Fraction(theta.P_bar[h])
    return s
