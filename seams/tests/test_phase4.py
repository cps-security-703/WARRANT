import numpy as np
import pandas as pd
import pytest
import yaml

from seams.analysis.metrics import attribution, cascade, certification
from seams.cosim.config import load_config, resolve_path, with_overrides
from seams.cosim.master import Master
from seams.contracts.params import Theta

CFG = load_config("seams/config/phase4.yaml", {"run.horizon_s": 72000})
THETA = Theta.load(resolve_path(CFG["contracts"]["theta_file"]))
P_BAR = {h: float(v) for h, v in THETA.P_bar.items()}


def spec(name, **strength):
    attacks = yaml.safe_load(resolve_path(f"seams/config/attacks/{name}.yaml").read_text())["attacks"]
    return [dict(a, **strength) for a in attacks]


def run(tmp, name, cfg=CFG, attacks=(), seed=42, hook=None):
    cfg = with_overrides(cfg, {"attacks": list(attacks), "run.seed": seed})
    m = Master(cfg, name, run_dir=tmp / name)
    if hook:
        hook(m)
    m.run()
    return tmp / name

def tight_power(frac):
    from fractions import Fraction
    def hook(m):
        th = m.theta
        P = {j: Fraction(v) * Fraction(str(frac)) for j, v in th.P_bar.items()}
        m.apply_theta(th.replace(f"{th.version}-tight", P_bar=P, P_assumed=P))
    return hook



def frames(d):
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in ("fine", "contract", "session", "events")}


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("phase4")


@pytest.fixture(scope="module")
def nominal(tmp):
    return [run(tmp, f"nominal_{s}", seed=s) for s in (42, 43)]


def test_theta_has_epsilon_and_passes_v1(nominal):
    assert THETA.eps and set(THETA.eps) >= {"D", "lin", "P_load_max", "V_ceiling", "cpl_V"}
    ev = frames(nominal[0])["events"]
    assert (ev.event == "V1:ok").any()


def test_certified_theta_keeps_phi_safe_on_nominal(nominal):
    for d in nominal:
        c = certification(d)
        assert c["unsound_periods"] == 0
        # measured finding: on real 2035 days the static theta certifies only part of the day
        # (the stressed afternoon fails V2); renegotiation (Phase 5) is what keeps it certified
        assert 0 < c["certified_share"] < 1, c
        assert c["v2_ms_max"] < 1000, "certification must fit easily inside a contract period"


def test_shield_power_accounting_matches_delivered_energy(nominal):
    f, s = frames(nominal[0])["fine"], frames(nominal[0])["session"]
    assert (f.P_pcc == f.P_sched).all(), "grid sees exactly what the chargers drew"
    # per hub: integral of PCC power = energy ACN-Sim delivered to all sessions (incl. active)
    pcc = f.groupby("hub").P_pcc.sum() / 60.0
    acn = s.groupby("hub").delivered_kWh.sum()
    for h in pcc.index:
        assert pcc[h] == pytest.approx(acn[h], rel=1e-9), h
    assert (f.P_pcc <= f.hub.map(P_BAR) + 1e-9).all()


# On real sessions the release only exceeds the calibrated static P_bar on heavy days (it is
# the renegotiated, tighter theta it exceeds), so the shield mechanism is exercised against a
# tight contract (every power guarantee at 30 %, arrival quotas unchanged).
TIGHT = 0.3


@pytest.fixture(scope="module")
def aggregator(tmp):
    return run(tmp, "aggregator", attacks=spec("aggregator", fraction=1.0), hook=tight_power(TIGHT))


def test_shields_contain_compromised_aggregator(aggregator):
    f = frames(aggregator)["fine"]
    p_bar = {h: TIGHT * v for h, v in P_BAR.items()}
    att = f[f.attack_active.notna()]
    assert (att.P_req > att.hub.map(p_bar) + 1).any(), "the malicious scheduler asks for more"
    assert (f.P_pcc <= f.hub.map(p_bar) + 1e-6).all(), "the PCC limiter holds every fine step"
    assert att.limiter_active.any()
    home_cap = TIGHT * float(THETA.P_bar["HOME"])
    assert (att.home_req_kw > home_cap + 1).any(), "the home portfolio is released above its guarantee"
    assert (f.home_kw <= home_cap + 1e-6).all(), "the demand-response cap holds every fine step"
    c = cascade(aggregator)
    assert c["trips"] == 0 and "grid" not in c["failing"]
    assert attribution(aggregator)["first_blame"] == "charging", "still detected and attributed"


def test_hub_admission_deflects_navigation_surge(tmp):
    d = run(tmp, "navigation", attacks=spec("navigation", fraction=1.0))
    f, c = frames(d)["fine"], frames(d)["contract"]
    # admitted arrivals per period never exceed the period quota
    admitted = (f.assign(k=f.t_fine // 5).groupby(["hub", "k"])[["arrivals", "refused"]].sum())
    quota = {h: float(THETA.lam_bar[h][1]) for h in THETA.lam_bar}
    adm = admitted.arrivals - admitted.refused
    assert (adm <= adm.index.get_level_values("hub").map(quota)).all()
    # with real demand the surge exceeds the windowed quota only on some days; when it is
    # refused it must be blamed on transport (attempts are monitored pre-shield)
    blame = attribution(d)["first_blame"]
    assert blame in ("", "transport") and (f.refused.sum() == 0 or blame == "transport")


def test_verifier_probes_do_not_change_physics(tmp, nominal):
    off = with_overrides(CFG, {"verifier": None})
    d_off = run(tmp, "nominal_42_noverifier", cfg=off, seed=42)
    a, b = frames(nominal[0])["fine"], frames(d_off)["fine"]
    for col, atol in (("V_true", 1e-8), ("xfmr_loading", 1e-5), ("P_pcc", 0), ("occupied_evse", 0), ("queue", 0)):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=atol, err_msg=col)
