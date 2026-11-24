import pandas as pd
import pytest
import yaml

from seams.analysis.metrics import attribution, cascade, certification, containment, renegotiation
from seams.cosim.config import load_config, resolve_path, with_overrides
from seams.cosim.master import Master

CFG = load_config("seams/config/phase5.yaml", {"run.horizon_s": 72000})
BASE = load_config("seams/config/phase2.yaml", {"run.horizon_s": 72000})
STRONG = {"fdi": {"bias_pu": -0.04}, "aggregator": {"fraction": 1.0}, "navigation": {"fraction": 1.0}}


def spec(name, **strength):
    attacks = yaml.safe_load(resolve_path(f"seams/config/attacks/{name}.yaml").read_text())["attacks"]
    return [dict(a, **strength) if i == 0 else a for i, a in enumerate(attacks)]


def run(tmp, name, cfg, attacks=(), seed=42):
    Master(with_overrides(cfg, {"attacks": list(attacks), "run.seed": seed}), name, run_dir=tmp / name).run()
    return tmp / name


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("phase5")


def enforced_thetas_certified(d):
    c = pd.read_parquet(d / "contract.parquet")
    return bool(c.certified.all()) and certification(d)["unsound_periods"] == 0


@pytest.mark.parametrize("seed", [42, 43])
def test_fdi_contained_within_window(tmp, seed):
    d = run(tmp, f"fdi_{seed}", CFG, spec("fdi", **STRONG["fdi"]), seed)
    b = run(tmp, f"fdi_base_{seed}", BASE, spec("fdi", **STRONG["fdi"]), seed)
    w, c = containment(d, b), cascade(d)
    assert w["baseline_depth"] == 2
    assert w["window_margin_s"] > 0, w
    # contained through the first hour: no charging failure before onset + 1 h
    assert pd.isna(c["first_charging_s"]) or c["first_charging_s"] > 3600, c
    assert attribution(d)["first_blame"] == "grid"
    ev = pd.read_parquet(d / "events.parquet")
    assert (ev.event == "mode_enter:grid").any(), "derating moved to the trusted estimate"
    assert enforced_thetas_certified(d)


@pytest.mark.parametrize("scenario", ["aggregator", "navigation"])
def test_other_scenarios_depth_one(tmp, scenario):
    d = run(tmp, scenario, CFG, spec(scenario, **STRONG[scenario]))
    c = cascade(d)
    assert c["depth"] == 1, c
    # at 2035 neither violates a guarantee on every day (real demand); never misattributed
    assert attribution(d)["first_blame"] in ("", c["origin"])
    assert enforced_thetas_certified(d)


def test_multi_fault_does_not_cross(tmp):
    d = run(tmp, "navfdi", CFG, spec("navigation_fdi"))
    c = cascade(d)
    assert set(filter(None, c["failing"].split(","))) <= set(c["origin"].split(","))
    assert enforced_thetas_certified(d)


def test_nominal_day(tmp):
    d = run(tmp, "nominal", CFG)
    a, r = attribution(d), renegotiation(d)
    assert a["false_incidents"] <= CFG["contracts"]["max_false_incidents_per_day"]
    assert enforced_thetas_certified(d)
    assert r["periods_fallback"] == 0
    assert r["solve_ms_max"] < CFG["renegotiation"]["time_budget_ms"]
