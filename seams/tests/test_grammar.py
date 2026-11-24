from fractions import Fraction

import pytest

from seams.contracts import grammar as G
from seams.contracts.params import Theta
from seams.contracts.verifier import v2_refinement
from seams.tests.tests_support import BASE_THETA, HUBS, PHI, S

K = 6
THETA = Theta.from_dict(BASE_THETA)

SINGLE = [
    G.TimeWindowedCap("H1", "P_bar", 60.0, 2, 4),
    G.TimeWindowedCap("H2", "lam_bar", 5, 1, 6),
    G.HubExclusion("H3", 3, 5),
    G.CurtailmentWindow("H1", 2, (80.0, 70.0, 90.0)),
    G.GridDistrust("H2", 0.012),
]


@pytest.mark.parametrize("template", SINGLE, ids=lambda t: type(t).__name__)
def test_each_template_round_trips(template):
    assert G.round_trip([template], THETA, K)


def test_combined_templates_round_trip_and_tighten_by_min():
    combo = SINGLE + [G.TimeWindowedCap("H1", "P_bar", 75.0, 1, 6), G.GridDistrust("H2", 0.02)]
    assert G.round_trip(combo, THETA, K)
    s = G.realize(combo, THETA, K)
    assert s.P["H1"][3] == Fraction(60)       # min(60 window, 70 schedule, 75 window)
    assert s.P["H1"][5] == Fraction(75)
    assert s.L["H3"][4] == 0 and s.L["H3"][2] == THETA.lam_bar["H3"][1]
    assert s.X["H2"] == Fraction(2, 100)


def test_window_clipped_to_horizon():
    t = G.TimeWindowedCap("H1", "P_bar", 50.0, 5, 99)
    assert G.round_trip([t], THETA, K)
    assert G.realize([t], THETA, K).P["H1"][6] == 50


@pytest.mark.parametrize("bad", [
    {"template": "Nope", "hub": "H1"},
    {"template": "TimeWindowedCap", "hub": "H9", "param": "P_bar", "value": 1, "k_start": 1, "k_end": 2},
    {"template": "TimeWindowedCap", "hub": "H1", "param": "voltage", "value": 1, "k_start": 1, "k_end": 2},
    {"template": "HubExclusion", "hub": "H1", "k_start": 0, "k_end": 2},
    {"template": "GridDistrust", "hub": "H1", "eps_widen": 0.5},
    {"template": "CurtailmentWindow", "hub": "H1", "k_start": 1},
    "not a dict",
])
def test_malformed_proposals_rejected(bad):
    with pytest.raises(ValueError):
        G.from_json(bad, HUBS, K)


@pytest.mark.parametrize("template", SINGLE, ids=lambda t: type(t).__name__)
def test_json_round_trip(template):
    assert G.from_json(G.to_json(template), HUBS, K) == template


def test_schedule_feeds_v2():
    # flat 105 kW: worst V = 0.965 - 4.6e-4 * 25 - 0.010 < 0.95; capped at 40 kW it certifies
    anchor = {"V": {h: 0.965 for h in HUBS}, "P": {h: 80.0 for h in HUBS}, "S": S}
    assert not v2_refinement(THETA, anchor, THETA.eps, PHI, K).ok
    caps = [G.TimeWindowedCap(h, "P_bar", 40.0, 1, K) for h in HUBS]
    s = G.realize(caps, THETA, K)
    assert v2_refinement(THETA, anchor, THETA.eps, PHI, K, P_bar_k=s.P_bar_k(), extra=s.extra()).ok
    widened = G.realize(caps + [G.GridDistrust("H1", 0.05)], THETA, K)
    assert not v2_refinement(THETA, anchor, THETA.eps, PHI, K, P_bar_k=widened.P_bar_k(), extra=widened.extra()).ok
