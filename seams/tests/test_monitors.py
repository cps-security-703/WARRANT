from fractions import Fraction

import pytest

from seams.contracts.monitors import SeamMonitor
from seams.contracts.params import Theta, outward
from seams.contracts.signals import PeriodSignals

HUBS = ["H1", "H2"]
REGS = ["r1"]
S = {"H1": {"H1": -2e-4, "H2": -0.7e-4}, "H2": {"H1": -0.7e-4, "H2": -2.5e-4}}
T = {"H1": {"r1": 3e-3}, "H2": {"r1": 3e-3}}
THETA = Theta.from_dict({
    "version": "test", "lam_bar": {h: {1: 6, 6: 20} for h in HUBS}, "P_bar": {"H1": 105, "H2": 105},
    "V_lo": 0.95, "V_hi": 1.05,
    "beta": {h: {1: 0.002, 6: 0.004} for h in HUBS},
})
TOL = {"kw": 0.01, "pu": 1e-6}


def sig(k, lam=(2, 2), P=(60.0, 60.0), V=None, tap=0.0, d=(105.0, 105.0), P_req=None):
    if V is None:
        V = tuple(1.0 + sum(S[h][j] * (P[i] - 60.0) for i, j in enumerate(HUBS)) + T[h]["r1"] * tap
                  for h in HUBS)
    P_req = P_req or P
    return PeriodSignals(k=k, lam=dict(zip(HUBS, lam)), P_req=dict(zip(HUBS, P_req)),
                         P=dict(zip(HUBS, P)), V_rep=dict(zip(HUBS, V)),
                         d_applied=dict(zip(HUBS, d)), taps=(tap,))


def run(trace):
    m = SeamMonitor(THETA, HUBS, REGS, windows=[1, 6], tol=TOL)
    out = []
    for s in trace:
        rows, blames, first, _ = m.update(s)
        m.publish(S, T)
        out.append((blames, first))
    return out


def blamed_domains(out):
    return {d for blames, _ in out for d, _ in blames}


def test_nominal_no_blame():
    trace = [sig(k, P=(60 + 10 * (k % 3), 50 + 5 * (k % 4)), tap=float(k // 4)) for k in range(30)]
    assert blamed_domains(run(trace)) == set(), "load swings and tap moves explained by S and T"


def test_navigation_blames_transport_only():
    trace = [sig(k) for k in range(10)] + [sig(k, lam=(12, 2)) for k in range(10, 14)]
    out = run(trace)
    assert blamed_domains(out) == {"transport"}
    firsts = [f for _, f in out if f]
    assert firsts[0] == [("transport", "H1")]


def test_fdi_blames_grid():
    trace = [sig(k) for k in range(10)]
    trace += [sig(10, V=(1.0 - 0.01, 1.0))]  # reported V drops with no power change
    out = run(trace)
    assert out[-1][1] == [("grid", "H1")]


def test_aggregator_blames_charging_and_excuses_grid():
    trace = [sig(k) for k in range(10)]
    big = sig(10, P=(133.0, 60.0))
    v = big.V_rep["H1"] - 0.08  # push the honestly reported voltage below the band too
    trace += [sig(10, P=(133.0, 60.0), V=(v, big.V_rep["H2"]))]
    out = run(trace)
    assert ("charging", "H1") in out[-1][0]
    assert all(d != "grid" for d, _ in out[-1][0])


def test_charging_must_follow_derating():
    trace = [sig(k) for k in range(10)] + [sig(10, d=(40.0, 105.0), P_req=(60.0, 60.0))]
    assert run(trace)[-1][0] == [("charging", "H1")]


def test_slow_ramp_caught_by_window_only():
    trace = [sig(k) for k in range(10)]
    trace += [sig(10 + i, V=(1.0 - 0.0015 * (i + 1), 1.0)) for i in range(8)]
    out = run(trace)
    assert blamed_domains(out[:10]) == set()
    assert "grid" in blamed_domains(out[10:])


def test_first_blame_once_per_incident():
    trace = [sig(k) for k in range(10)] + [sig(k, lam=(12, 2)) for k in range(10, 13)]
    trace += [sig(k) for k in range(13, 20)] + [sig(20, lam=(12, 2))]
    out = run(trace)
    firsts = [f for _, f in out if f]
    assert len(firsts) == 2
    assert out[14][0], "still blamed: the 6-period window holds 36 arrivals > 20"


def test_sustained_surge_caught_by_window_only():
    trace = [sig(k) for k in range(10)] + [sig(k, lam=(5, 2)) for k in range(10, 16)]
    out = run(trace)
    # period 11: window = 4*2 + 2*5 = 18 <= 20; period 12: 3*2 + 3*5 = 21 > 20
    assert blamed_domains(out[:12]) == set()
    assert out[12][1] == [("transport", "H1")]


def test_outward_rounding_is_conservative():
    assert outward(0.1 + 0.2, "down") <= Fraction(3, 10) <= outward(0.1 + 0.2, "up")
    assert THETA.V_lo >= Fraction(95, 100) and THETA.V_hi <= Fraction(105, 100)
