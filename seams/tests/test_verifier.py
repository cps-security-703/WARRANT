import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from seams.contracts.params import Theta
from seams.contracts.verifier import v1_compatibility, v2_refinement

HUBS = ["H1", "H2", "H3"]
PHI = {"v_min": 0.95, "v_max": 1.05}
K = 6
BASE = {
    "version": "t", "lam_bar": {h: {1: 17, 6: 29} for h in HUBS}, "P_bar": {h: 105 for h in HUBS},
    "V_lo": 0.95, "V_hi": 1.05, "beta": {h: {1: 0.007, 6: 0.016} for h in HUBS},
    "epsilon": {
        "D": {h: {k: 0.005 + 0.001 * k for k in range(1, K + 1)} for h in HUBS},
        "lin": {h: 0.002 for h in HUBS}, "P_load_max": {h: 109.5 for h in HUBS},
        "V_ceiling": {h: 1.045 for h in HUBS},
    },
}
S = {"H1": {"H1": -2.2e-4, "H2": -0.7e-4, "H3": -0.7e-4},
     "H2": {"H1": -0.8e-4, "H2": -2.6e-4, "H3": -1.2e-4},
     "H3": {"H1": -0.8e-4, "H2": -1.2e-4, "H3": -2.6e-4}}


def theta(**over):
    d = {**BASE, **over}
    return Theta.from_dict(d)


def test_v1_ok_for_matching_contracts():
    assert v1_compatibility(theta(), [1, 6], K).ok


@pytest.mark.parametrize("over, seam", [
    ({"lam_assumed": {h: {1: 17, 6: 25} for h in HUBS}}, "transport->charging"),
    ({"P_assumed": {h: 100 for h in HUBS}}, "charging->grid"),
    ({"V_lo_assumed": 0.96}, "grid->charging"),
])
def test_v1_detects_incompatibility_with_counterexample(over, seam):
    v = v1_compatibility(theta(**over), [1, 6], K)
    assert not v.ok and seam in v.check and v.counterexample


def test_v1_windowed_guarantee_can_imply_a_looser_single_window_assumption():
    t = theta(lam_assumed={h: {1: 29, 6: 29} for h in HUBS})
    assert v1_compatibility(t, [1, 6], K).ok


def brute_force_v2(t, anchor, use_eps=True):
    if use_eps and any(t.eps["V_ceiling"][h] > Fraction(PHI["v_max"]) for h in HUBS):
        return False
    if any(t.P_bar[j] > t.eps["P_load_max"][j] for j in HUBS):
        return False
    for k in range(1, K + 1):
        for h in HUBS:
            low = Fraction(anchor["V"][h])
            for j in HUBS:
                s = Fraction(anchor["S"][h][j])
                low += min(s * (0 - Fraction(anchor["P"][j])), s * (t.P_bar[j] - Fraction(anchor["P"][j])))
            if use_eps:
                low -= t.eps["D"][h][k] + t.eps["lin"][h]
            if low < Fraction(PHI["v_min"]):
                return False
    return True


@pytest.mark.parametrize("use_eps", [True, False])
def test_v2_matches_brute_force_on_random_anchors(use_eps):
    rng = np.random.default_rng(7)
    t = theta()
    agree, n_cert = 0, 0
    for _ in range(60):
        anchor = {"V": {h: float(rng.uniform(0.96, 1.04)) for h in HUBS},
                  "P": {h: float(rng.uniform(0, 105)) for h in HUBS}, "S": S}
        z = v2_refinement(t, anchor, t.eps, PHI, K, use_eps).ok
        assert z == brute_force_v2(t, anchor, use_eps)
        n_cert += z
    assert 0 < n_cert < 60, "the sample must exercise both verdicts"


def test_v2_counterexample_names_the_violation():
    t = theta()
    anchor = {"V": {h: 0.955 for h in HUBS}, "P": {h: 50.0 for h in HUBS}, "S": S}
    v = v2_refinement(t, anchor, t.eps, PHI, K)
    assert not v.ok and v.counterexample["violation"].startswith("undervoltage")
    assert v.counterexample["V"] < 0.95


def test_v2_ceiling_and_loading():
    t = theta()
    anchor = {"V": {h: 1.02 for h in HUBS}, "P": {h: 0.0 for h in HUBS}, "S": S}
    bad_ceiling = dataclasses.replace(t, eps={**t.eps, "V_ceiling": {h: Fraction(106, 100) for h in HUBS}})
    assert v2_refinement(bad_ceiling, anchor, bad_ceiling.eps, PHI, K).check == "V2:ceiling"
    # 115 kW everywhere still leaves V >= 1.02 - 0.0414 - 0.013 > 0.95: only loading can fail
    over_rated = t.replace("big", P_bar={h: Fraction(115) for h in HUBS})
    v = v2_refinement(over_rated, anchor, over_rated.eps, PHI, K)
    assert not v.ok and v.counterexample["violation"].startswith("loading")


def test_theta_exact_round_trip():
    t = theta()
    assert Theta.from_dict(t.to_dict()) == t
