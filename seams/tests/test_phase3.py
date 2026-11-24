import numpy as np
import pandas as pd
import pytest
import yaml

from seams.analysis.metrics import attribution, cascade
from seams.cosim.config import load_config, resolve_path, with_overrides
from seams.cosim.master import Master

CFG = load_config("seams/config/phase3.yaml", {"run.horizon_s": 72000})
STRONG = {
    "fdi": {"bias_pu": -0.04},
    "aggregator": {"fraction": 1.0},
    "navigation": {"fraction": 1.0},
}


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



@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("phase3")


def test_theta_static_exists_and_is_calibrated():
    theta = yaml.safe_load(resolve_path(CFG["contracts"]["theta_file"]).read_text())
    assert theta["meta"]["calibration_seeds"] == CFG["contracts"]["calibration"]["seeds"]
    assert not set(theta["meta"]["calibration_seeds"]) & {42, 43, 44}, "calibration seeds are disjoint"


# Where each attack violates a contract on real data: FDI at the 2035 default; the aggregator's
# release exceeds a tight theta (the static one only on heavy days; renegotiated ones often). Navigation at full strength builds a real queue but only sometimes
# exceeds the windowed arrival guarantee, so for it only "never misattributed" is asserted.
AT = {"fdi": None, "aggregator": 0.3}  # aggregator: power guarantees at 30 % (see test_phase4)


@pytest.mark.parametrize("scenario", sorted(STRONG))
@pytest.mark.parametrize("seed", [42, 43])
def test_correct_first_blame_above_detection_strength(tmp, scenario, seed):
    hook = tight_power(AT[scenario]) if AT.get(scenario) else None
    d = run(tmp, f"{scenario}_{seed}", attacks=spec(scenario, **STRONG[scenario]), seed=seed, hook=hook)
    a, c = attribution(d), cascade(d)
    assert a["false_incidents"] == 0
    if scenario not in AT:
        assert a["first_blame"] in ("", c["origin"]), (a["first_blame"], c["origin"])
        return
    assert a["attribution_ok"], (a["first_blame"], c["origin"])
    assert a["first_blame"] == c["origin"]
    assert a["tau_detect_s"] > 0


def test_multi_fault_first_blame_is_a_true_origin(tmp):
    a = attribution(run(tmp, "navfdi", attacks=spec("navigation_fdi")))
    assert set(filter(None, a["first_blame"].split(","))) <= {"grid", "transport"}


@pytest.fixture(scope="module")
def nominal(tmp):
    return [run(tmp, f"nominal_{s}", seed=s) for s in (42, 43, 44)]


def test_false_attribution_rate_within_bound(nominal):
    per_day = np.mean([attribution(d)["false_incidents"] for d in nominal])
    assert per_day <= CFG["contracts"]["max_false_incidents_per_day"]


def test_beta_validated_on_nominal(nominal):
    c = pd.concat([pd.read_parquet(d / "contract.parquet") for d in nominal])
    for w in CFG["contracts"]["windows"]:
        rate = (c[f"rho_G_G_phys_{w}"] < 0).mean()
        assert rate <= 0.005, f"w={w}: {rate:.4f} of nominal periods violate the physics term"


def test_monitoring_does_not_change_physics(tmp, nominal):
    off = with_overrides(CFG, {"contracts.monitor": False})
    d_off = run(tmp, "nominal_42_off", cfg=off, seed=42)
    a = pd.read_parquet(nominal[0] / "fine.parquet")
    b = pd.read_parquet(d_off / "fine.parquet")
    # power-flow tolerance 1e-9: voltages agree to ~1e-10 pu, loading (in %) to ~1e-7
    for col, atol in (("V_true", 1e-8), ("xfmr_loading", 1e-5), ("P_pcc", 0), ("occupied_evse", 0), ("queue", 0)):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=atol, err_msg=col)
