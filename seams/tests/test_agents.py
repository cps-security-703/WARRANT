import json
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from seams.agents import gate
from seams.agents.llm import LLM, CacheMiss
from seams.agents.operator_agent import OperatorAgent
from seams.agents.protocol import Negotiation
from seams.contracts import grammar as G
from seams.contracts.params import Theta
from seams.tests.tests_support import BASE_THETA, HUBS, PHI, S

K = 6
THETA = Theta.from_dict(BASE_THETA)
WINDOWS = sorted(THETA.lam_bar["H1"])
SAFE = THETA.replace("safe", P_bar={h: Fraction(2) for h in HUBS}, P_assumed={h: Fraction(2) for h in HUBS})
CFG = {"agents": {"max_candidates": 2, "forecast_periods": 3, "deadline_s": 120, "revise_rounds": 1},
       "clock": {"contract_period_s": 120.0}}


def ctx(V=0.965, P=80.0, min_valid=2):
    anchor = {"V": {h: V for h in HUBS}, "P": {h: P for h in HUBS}, "S": S}
    return gate.Context(kc=10, base=THETA, safe=SAFE, anchor=anchor, eps=THETA.eps, phi=PHI, K=K, windows=WINDOWS,
                        extra={}, caps=dict(THETA.P_bar), floors=dict(SAFE.P_bar), reason="test", modes=(),
                        min_valid=min_valid)


# --- grammar: ValidityWindow ------------------------------------------------------
def test_validity_window_shortens_horizon_only():
    s = G.realize([G.TimeWindowedCap("H1", "P_bar", 60.0, 1, K), G.ValidityWindow(3)], THETA, K)
    assert s.valid == 3 and s.P["H1"][K] == 60
    assert G.round_trip([G.ValidityWindow(3)], THETA, K)
    assert G.from_json(G.to_json(G.ValidityWindow(4)), HUBS, K, min_leads=2) == G.ValidityWindow(4)


@pytest.mark.parametrize("leads", [1, K + 1, 2.5])
def test_validity_window_bounds(leads):
    with pytest.raises(ValueError):
        G.from_json({"template": "ValidityWindow", "leads": leads}, HUBS, K, min_leads=2)


# --- gate --------------------------------------------------------------------------
def test_gate_is_monotone_in_caps():
    c = ctx()
    caps = [G.TimeWindowedCap(h, "P_bar", 40.0, 1, K) for h in HUBS]
    assert not gate.certify(c, []).ok              # flat 105 kW fails at V = 0.965
    assert gate.certify(c, caps).ok
    assert gate.certify(c, caps + [G.CurtailmentWindow("H2", 1, (10.0, 0.0))]).ok


def test_short_validity_certifies_where_full_horizon_cannot():
    # anchored at 0 kW and V = 0.9595: the bound D(k) + lin = 0.0075 + 0.0005 k leaves 0.95 at
    # k = 2 but not at k = K = 6, whatever the caps
    c = ctx(V=0.9595, P=0.0)
    zero = [G.TimeWindowedCap(h, "P_bar", 0.0, 1, K) for h in HUBS]
    full, short = gate.certify(c, zero), gate.certify(c, zero + [G.ValidityWindow(2)])
    assert not full.ok and short.ok


def test_expired_plan_fails_and_shift_keeps_remaining_validity():
    s = G.realize([G.ValidityWindow(3), G.CurtailmentWindow("H1", 1, (10.0, 20.0, 30.0))], THETA, K)
    later = gate.shifted(s, 2, K)
    assert later.valid == 1 and later.P["H1"][1] == 30
    assert gate.certify_schedule(ctx(), gate.shifted(s, 3, K)).check == "expired"


def test_lead_theta_carries_quota_and_caps():
    s = G.realize([G.HubExclusion("H2", 1, 2), G.TimeWindowedCap("H1", "P_bar", 50.0, 1, 1)], THETA, K)
    t1 = gate.lead_theta(THETA, s, 1, "x")
    assert t1.P_bar["H1"] == 50 and t1.lam_bar["H2"][1] == 0 and t1.P_assumed["H1"] == THETA.P_bar["H1"]


# --- LLM layer: replay cache ---------------------------------------------------------
class FakeClient:
    def __init__(self, reply):
        self.reply, self.n = reply, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, messages, **kw):
        self.n += 1
        text = self.reply(messages[-1]["content"], kw["response_format"]["json_schema"]["name"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


BACKBONE = {"model": "fake", "temperature": 0.0, "price_in_per_mtok": 1.0, "price_out_per_mtok": 1.0}


def test_replay_cache_records_then_replays(tmp_path):
    client = FakeClient(lambda u, n: json.dumps({"x": 1}))
    rec = LLM(BACKBONE, tmp_path / "c.sqlite", "record", client)
    assert rec.call("s", "u", "n", {"type": "object"})[0] == {"x": 1}
    assert rec.call("s", "u", "n", {"type": "object"})[0] == {"x": 1} and client.n == 1
    rep = LLM(BACKBONE, tmp_path / "c.sqlite", "replay", FakeClient(lambda u, n: "unused"))
    assert rep.call("s", "u", "n", {"type": "object"})[0] == {"x": 1}
    with pytest.raises(CacheMiss):
        rep.call("s", "other", "n", {"type": "object"})


def test_malformed_reply_is_retried_then_dropped(tmp_path):
    llm = LLM(BACKBONE, tmp_path / "c.sqlite", "record", FakeClient(lambda u, n: "not json"))
    obj, _, err = llm.call("s", "u", "n", {"type": "object"})
    assert obj is None and "JSONDecodeError" in err


# --- one negotiation with scripted agents ------------------------------------------------
def _cap(h, v):
    return {"template": "TimeWindowedCap", "hub": h, "param": "P_bar", "value": v, "k_start": 1, "k_end": K,
            "schedule": None, "eps_widen": None, "leads": None}


def _history():
    sig = SimpleNamespace(lam={h: 2 for h in HUBS}, P_req={h: 30.0 for h in HUBS}, P={h: 30.0 for h in HUBS})
    return {k: (sig, None) for k in range(3)}


def _negotiate(tmp_path, reply):
    llm = LLM(BACKBONE, tmp_path / "c.sqlite", "record", FakeClient(reply))
    neg = Negotiation([OperatorAgent(r, llm, CFG) for r in ("fleet", "charging", "utility")], CFG)
    default = THETA.replace("d", P_bar={h: Fraction(20) for h in HUBS})
    return neg.run(ctx(), _history(), default, "theta_safe")


def test_unanimous_certified_candidate_is_agreed(tmp_path):
    def reply(user, name):
        if name == "proposal":
            if "charging operator" in user or "Your private view: {\"forecast_demand_kw\"" in user:
                return json.dumps({"candidates": [{"rationale": "", "templates": [_cap(h, 40.0) for h in HUBS]},
                                                  {"rationale": "", "templates": [_cap("H1", 104.0)]}]})
            return json.dumps({"candidates": []})
        ids = re.findall(r"^([FCU]\d\d) \(from", user, re.M)
        return json.dumps({"rationale": "", "accept": ids[:1], "ranking": ids[:1]})
    sched, rec = _negotiate(tmp_path, reply)
    assert rec["agreement"] and rec["n_certified"] == 1 and rec["n_candidates"] == 2
    assert sched.P["H2"][3] == 40


def test_no_agreement_without_unanimity(tmp_path):
    def reply(user, name):
        if name == "proposal":
            return json.dumps({"candidates": [{"rationale": "", "templates": [_cap(h, 40.0) for h in HUBS]}]})
        if "unserved charging energy" in user or "forecast_demand_kw" in user:
            return json.dumps({"rationale": "", "accept": [], "ranking": []})
        ids = re.findall(r"^([FCU]\d\d) \(from", user, re.M)
        return json.dumps({"rationale": "", "accept": ids, "ranking": ids})
    sched, rec = _negotiate(tmp_path, reply)
    assert sched is None and not rec["agreement"] and rec["rounds"] == 2


def test_infeasible_state_skips_the_agents(tmp_path):
    client = FakeClient(lambda u, n: pytest.fail("no LLM call expected"))
    llm = LLM(BACKBONE, tmp_path / "c.sqlite", "record", client)
    neg = Negotiation([OperatorAgent(r, llm, CFG) for r in ("fleet", "charging", "utility")], CFG)
    sched, rec = neg.run(ctx(V=0.90), _history(), SAFE, "theta_safe")
    assert sched is None and rec["skipped"] == "infeasible" and client.n == 0


def test_anyof_enum_rewrite_keeps_the_constraint():
    from seams.agents.llm import nullable_enums_as_anyof
    from seams.agents.operator_agent import TEMPLATE_SCHEMA
    p = nullable_enums_as_anyof(TEMPLATE_SCHEMA)["properties"]["param"]
    assert p == {"anyOf": [{"type": "string", "enum": ["P_bar", "lam_bar"]}, {"type": "null"}]}
    assert TEMPLATE_SCHEMA["properties"]["param"]["enum"] == ["P_bar", "lam_bar", None]   # canonical untouched


# --- floor protocol ---------------------------------------------------------------------
def test_floored_plan_never_below_the_classical_contract():
    s = G.realize([G.CurtailmentWindow("H1", 1, (90.0, 10.0)), G.HubExclusion("H2", 1, 2), G.ValidityWindow(3)], THETA, K)
    classical = THETA.replace("c", P_bar={h: Fraction(50) for h in THETA.P_bar})
    f = gate.floored(s, classical, THETA)
    assert f.valid == 3 and f.P["H1"][1] == 90 and f.P["H1"][2] == 50 and f.P["H3"][1] == 105
    assert f.L["H2"][1] == THETA.lam_bar["H2"][1]          # quotas back to the base contract
    assert s.P["H1"][2] == 10                               # the plan itself is untouched


def test_floored_is_none_when_classical_dominates():
    s = G.realize([G.TimeWindowedCap(h, "P_bar", 40.0, 1, K) for h in HUBS], THETA, K)
    classical = THETA.replace("c", P_bar={h: Fraction(60) for h in THETA.P_bar})
    assert gate.floored(s, classical, THETA) is None


# --- RQ5: Byzantine negotiator ------------------------------------------------------------
def _neg(tmp_path, reply, byz=None, stall_limit=None):
    cfg = {"agents": dict(CFG["agents"], stall_limit=stall_limit), "clock": CFG["clock"]}
    client = FakeClient(reply)
    llm = LLM(BACKBONE, tmp_path / "c.sqlite", "record", client)
    agents = [OperatorAgent(r, llm, cfg, byz if byz and r == byz["role"] else None) for r in ("fleet", "charging", "utility")]
    return Negotiation(agents, cfg), client


def _accept_all(user, name):
    if name == "proposal":
        return json.dumps({"candidates": [{"rationale": "", "templates": [_cap(h, 40.0) for h in HUBS]}]})
    ids = re.findall(r"^([FCU]\d\d) \(from", user, re.M)
    return json.dumps({"rationale": "", "accept": ids, "ranking": ids})


def test_stalling_agent_blocks_agreement_without_llm_calls(tmp_path):
    neg, client = _neg(tmp_path, _accept_all, {"role": "charging", "strategy": "stall"})
    default = THETA.replace("d", P_bar={h: Fraction(20) for h in HUBS})
    sched, rec = neg.run(ctx(), _history(), default, "theta_safe")
    assert sched is None and not rec["agreement"]
    assert not any(c["source"] == "charging" for c in rec["candidates"])
    # 2 honest agents x (propose, rank, revise, rank) = 8 calls; none for the staller
    assert client.n == 8


def test_stall_limit_excludes_the_staller(tmp_path):
    neg, _ = _neg(tmp_path, _accept_all, {"role": "charging", "strategy": "stall"}, stall_limit=1)
    default = THETA.replace("d", P_bar={h: Fraction(20) for h in HUBS})
    first, rec1 = neg.run(ctx(), _history(), default, "theta_safe")
    assert first is None and {"event": "excluded", "role": "charging", "after_stalls": 1} in rec1["log"]
    second, rec2 = neg.run(ctx(), _history(), default, "theta_safe")
    assert second is not None and rec2["agreement"] and rec2["excluded"] == ["charging"]  # the honest two agree


def test_misreport_scales_only_the_false_type():
    cfg = CFG
    honest = OperatorAgent("charging", None, cfg)
    liar = OperatorAgent("charging", None, cfg, {"strategy": "misreport", "misreport_scale": 3.0})
    h, l = honest.view(ctx(), _history()), liar.view(ctx(), _history())
    assert all(abs(l["forecast_demand_kw"][j] - 3 * h["forecast_demand_kw"][j]) < 1e-9 for j in h["forecast_demand_kw"])
    assert h["forecast_demand_kw"] == dict(_history()[2][0].P_req)     # honest view untouched
    util_h = OperatorAgent("utility", None, cfg).view(ctx(), _history())
    util_l = OperatorAgent("utility", None, cfg, {"strategy": "misreport"}).view(ctx(), _history())
    assert util_l["certified_margin_at_base_caps_mpu"] != util_h["certified_margin_at_base_caps_mpu"]


def test_directive_only_in_the_compromised_prompt():
    honest = OperatorAgent("charging", None, CFG).system(ctx())
    shifted = OperatorAgent("charging", None, CFG, {"strategy": "cost_shift"}).system(ctx())
    assert "HIDDEN DIRECTIVE" not in honest and shifted.startswith(honest) and "HOME" in shifted[len(honest):]
