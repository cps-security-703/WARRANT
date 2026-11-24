from fractions import Fraction

from seams.contracts.params import Theta
from seams.contracts.renegotiate import Renegotiator, optimize_P_bar
from seams.contracts.signals import PeriodSignals
from seams.contracts.verifier import v2_refinement
from seams.tests.tests_support import BASE_THETA, HUBS, PHI, S

K = 6
STATIC = Theta.from_dict(BASE_THETA)
SAFE = Theta.from_dict({**BASE_THETA, "version": "safe", "P_bar": {h: 26.25 for h in HUBS},
                        "meta": {"offline_certified": True}})
CFG = {
    "renegotiation": {"exit_quiet_periods": 3, "charging_cap_frac": 0.9, "time_budget_ms": 5000,
                      "renegotiate_on_decertification": True},
    "grid": {"phi_safe": PHI}, "contracts": {"windows": [1, 6]}, "verifier": {"horizon_periods": K},
}


def anchor(v, p=80.0):
    return {"V": {h: v for h in HUBS}, "P": {h: p for h in HUBS}, "S": S}


def test_optimizer_output_is_certified_and_maximal():
    a = anchor(0.965)  # uncertifiable at 105 kW (see test_grammar.test_schedule_feeds_v2)
    caps = {h: Fraction(105) for h in HUBS}
    floors = {h: Fraction(0) for h in HUBS}
    pb, _ = optimize_P_bar(STATIC, a, STATIC.eps, PHI, K, caps, floors, {}, {})
    assert pb is not None and all(pb[h] < 105 for h in HUBS)
    cand = STATIC.replace("c", P_bar=pb)
    assert v2_refinement(cand, a, STATIC.eps, PHI, K).ok
    bumped = STATIC.replace("c+", P_bar={h: pb[h] + (1 if h == "H1" else 0) for h in HUBS})
    assert not v2_refinement(bumped, a, STATIC.eps, PHI, K).ok, "a larger allowance would not certify"


def test_optimizer_infeasible_below_floor():
    pb, _ = optimize_P_bar(STATIC, anchor(0.94), STATIC.eps, PHI, K,
                           {h: Fraction(105) for h in HUBS}, {h: Fraction(50) for h in HUBS}, {}, {})
    assert pb is None


def sig(k, v=0.99, p=80.0):
    return PeriodSignals(k=k, lam={h: 1 for h in HUBS}, P_req={h: p for h in HUBS}, P={h: p for h in HUBS},
                         V_rep={h: v for h in HUBS}, d_applied={h: 105.0 for h in HUBS}, taps=(0.0,))


def run(steps):
    r = Renegotiator(CFG, STATIC, SAFE)
    hist, out = {}, []
    for k, (s, blames) in enumerate(steps):
        hist[k] = (s, S)
        decision, events = r.step(k, s, S, blames, hist)
        out.append((decision, events))
    return r, out


def test_static_kept_when_certified():
    _, out = run([(sig(k, v=1.02, p=40.0), []) for k in range(5)])
    assert all(d.status == "static" and d.theta is STATIC for d, _ in out)


def test_decertification_renegotiates_then_relaxes():
    steps = [(sig(0, v=1.02, p=40.0), []), (sig(1, v=0.965), []), (sig(2, v=1.02, p=40.0), [])]
    _, out = run(steps)
    assert [d.status for d, _ in out] == ["static", "renegotiated", "static"]
    assert out[1][0].certified and all(out[1][0].theta.P_bar[h] < 105 for h in HUBS)


def test_infeasible_falls_back_to_theta_safe():
    # even at the theta_safe floor: 0.92 + 4.6e-4 * (100 - 26.25) - 0.010 < 0.95
    _, out = run([(sig(0, v=0.92, p=100.0), [])])
    d = out[0][0]
    assert d.status == "fallback" and d.theta is SAFE and d.certified  # offline certificate


def test_charging_blame_caps_and_exits_after_quiet_periods():
    steps = [(sig(0, v=1.02, p=40.0), [("charging", "H1")])] + [(sig(k, v=1.02, p=40.0), []) for k in range(1, 5)]
    r, out = run(steps)
    assert out[0][0].theta.P_bar["H1"] <= Fraction(945, 10)
    assert ("mode_enter:charging", "H1") in out[0][1]
    exits = [k for k, (_, ev) in enumerate(out) if ("mode_exit:charging", "H1") in ev]
    assert exits == [3] and out[4][0].status == "static"


def test_grid_distrust_uses_trusted_estimate_and_exits_when_consistent():
    # nominal history, then reported voltage falsified 0.04 low with the power unchanged
    steps = [(sig(k, v=1.0, p=60.0), []) for k in range(10)]
    steps += [(sig(10, v=0.96, p=60.0), [("grid", "H1")])]
    steps += [(sig(k, v=0.96 if k < 14 else 1.0, p=60.0), []) for k in range(11, 20)]
    r, out = run(steps)
    assert ("mode_enter:grid", "H1") in out[10][1]
    # while falsified, V2 anchors on V_est (1.0), not the falsified 0.96: no needless cut
    assert out[11][0].status == "renegotiated" and out[11][0].theta.P_bar["H1"] == 105
    exit_k = [k for k, (_, ev) in enumerate(out) if ("mode_exit:grid", "H1") in ev]
    assert exit_k and exit_k[0] >= 14 + CFG["renegotiation"]["exit_quiet_periods"] - 1


def test_anchor_reentry_keeps_the_last_clean_anchor():
    # a falsification inside the anchor estimator's tolerance (0.008 < D(K) + lin = 0.010) is
    # re-trusted while still active; the re-entry must not re-anchor on the falsified readings
    steps = [(sig(k, v=1.0, p=60.0), []) for k in range(10)]
    steps += [(sig(10, v=0.992, p=60.0), [("grid", "H1")])]
    steps += [(sig(k, v=0.992, p=60.0), []) for k in range(11, 20)]
    steps += [(sig(20, v=0.992, p=60.0), [("grid", "H1")])]
    r, out = run(steps)
    entries = [k for k, (_, ev) in enumerate(out) if ("mode_enter:grid", "H1") in ev]
    assert entries == [10, 20], "re-trusted while falsified, then caught again"
    assert r.modes[("grid", "H1")]["V0"] == 1.0, "re-anchored inside the attack"


def mv_sig(k, v, v_mv=1.0, p=60.0):
    s = sig(k, v=v, p=p)
    s.V_mv_rep = {h: v_mv for h in HUBS}
    return s


def test_mv_estimator_does_not_retrust_an_active_falsification(tmp_path):
    import json
    bdd = tmp_path / "bdd.json"
    bdd.write_text(json.dumps({"hubs": {h: {"a": 0.0, "b": 1.0, "c": 0.0, "est_err_max": 0.005} for h in HUBS}}))
    cfg = {**CFG, "renegotiation": {**CFG["renegotiation"], "distrust_estimator": "mv"}, "ids": {"bdd_file": str(bdd)}}
    r = Renegotiator(cfg, STATIC, SAFE)
    steps = [(mv_sig(k, 1.0), []) for k in range(10)] + [(mv_sig(10, 0.96), [("grid", "H1")])]
    steps += [(mv_sig(k, 0.96 if k < 50 else 1.0), []) for k in range(11, 60)]
    hist, exits = {}, []
    for k, (s, blames) in enumerate(steps):
        hist[k] = (s, S)
        d, ev = r.step(k, s, S, blames, hist)
        if k == 11:
            assert d.theta.P_bar["H1"] == 105, "V2 anchors on the MV estimate, not the falsified 0.96"
        exits += [k for e in ev if e == ("mode_exit:grid", "H1")]
    # no drift bound that grows past the falsification: re-trusted only once it stopped
    assert exits == [50 + CFG["renegotiation"]["exit_quiet_periods"] - 1]
    assert r.v_est("H1", {h: 60.0 for h in HUBS}, {h: 0.99 for h in HUBS}) == 0.99


def test_blind_tightening_degrades_every_domain_at_every_hub():
    cfg = {**CFG, "renegotiation": {**CFG["renegotiation"], "attribution": "blind"}}
    r = Renegotiator(cfg, STATIC, SAFE)
    s = sig(0, v=1.02, p=40.0)
    decision, events = r.step(0, s, S, [("charging", "H1")], {0: (s, S)})
    assert set(r.modes) == {(d, h) for d in ("transport", "charging", "grid") for h in HUBS}
    assert all(decision.theta.P_bar[h] <= Fraction(945, 10) for h in HUBS)


def test_both_estimator_needs_both_to_exit_and_estimates_from_the_anchor(tmp_path):
    # two-meter falsification: V_rep and the MV meter agree (the MV estimate is poisoned), so the
    # MV test alone would re-trust at once; the anchor test keeps distrust until the attack ends
    import json
    bdd = tmp_path / "bdd.json"
    bdd.write_text(json.dumps({"hubs": {h: {"a": 0.0, "b": 1.0, "c": 0.0, "est_err_max": 0.005} for h in HUBS}}))
    cfg = {**CFG, "renegotiation": {**CFG["renegotiation"], "distrust_estimator": "both"}, "ids": {"bdd_file": str(bdd)}}
    r = Renegotiator(cfg, STATIC, SAFE)
    steps = [(mv_sig(k, 1.0), []) for k in range(10)] + [(mv_sig(10, 0.96, v_mv=0.96), [("grid", "H1")])]
    steps += [(mv_sig(k, 0.96 if k < 30 else 1.0, v_mv=0.96 if k < 30 else 1.0), []) for k in range(11, 40)]
    hist, exits = {}, []
    for k, (s, blames) in enumerate(steps):
        hist[k] = (s, S)
        d, ev = r.step(k, s, S, blames, hist)
        if k == 11:
            assert abs(r.v_est("H1", s.P, s.V_mv_rep) - 1.0) < 1e-9, "estimates from the anchor, not the MV meter"
        exits += [k for e in ev if e == ("mode_exit:grid", "H1")]
    assert exits == [30 + CFG["renegotiation"]["exit_quiet_periods"] - 1]
