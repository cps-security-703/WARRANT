import pandas as pd
import pytest

from seams.analysis.metrics import _sustained, cascade
from seams.cosim.config import load_config, resolve_path, with_overrides
from seams.cosim.master import Master
from seams.domains.grid.protection import TransformerProtection

BASE = load_config("seams/config/phase2.yaml", {"run.horizon_s": 72000})  # to 20:00


def attack(name, **strength):
    import yaml
    specs = yaml.safe_load(resolve_path(f"seams/config/attacks/{name}.yaml").read_text())["attacks"]
    return [dict(spec, **strength) for spec in specs]


def run(tmp, name, attacks=None, seed=42, **over):
    cfg = with_overrides(BASE, {"attacks": attacks or [], "run.seed": seed, **over})
    Master(cfg, name, run_dir=tmp / name).run()
    return tmp / name


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("phase2")


@pytest.fixture(scope="module")
def nominal(tmp):
    return run(tmp, "nominal")


def test_nominal_protection_only_has_no_domain_failure(nominal):
    c = cascade(nominal)
    assert not (c["fail_transport"] or c["fail_charging"] or c["fail_grid"])
    ev = pd.read_parquet(nominal / "events.parquet")
    assert not ev.event.isin(["trip", "ids_alarm"]).any(), "no trips or IDS false alarms on a nominal day"


@pytest.mark.parametrize("seed", [42, 43])
def test_fdi_crosses_above_threshold(tmp, seed):
    strong = cascade(run(tmp, f"fdi_strong_{seed}", attack("fdi", bias_pu=-0.04), seed))
    weak = cascade(run(tmp, f"fdi_weak_{seed}", attack("fdi", bias_pu=-0.005), seed))
    assert strong["origin"] == "grid" and strong["depth"] >= 2 and "charging" in strong["failing"]
    assert strong["tau_cross_s"] > 0
    assert strong["tau_detect_ids_s"] >= 0, "the retrained IDS sees falsified grid voltage"
    assert weak["depth"] == 1


def test_aggregator_load_step_grows_with_electrification(tmp, nominal):
    rel = "1110 <= t_fine < 1170"
    def dip(base, hit):
        lo = lambda d: pd.read_parquet(d / "fine.parquet").query(rel).V_true.min()
        return lo(base) - lo(hit)
    over45 = {"scenario.electrification": "2045"}
    nom45 = run(tmp, "nominal_2045", **over45)
    agg45 = run(tmp, "agg_2045", attack("aggregator", fraction=1.0), **over45)
    agg35 = run(tmp, "agg_2035", attack("aggregator", fraction=1.0))
    home = lambda d: pd.read_parquet(d / "fine.parquet").query(rel).home_kw.max()
    assert home(agg45) > 1.3 * home(nom45), "held demand is released at once"
    assert dip(nom45, agg45) > max(dip(nominal, agg35), 0.005)
    c = cascade(agg45)
    assert c["origin"] == "charging" and "transport" not in c["failing"]


def test_navigation_surges_but_is_contained_by_protection(tmp, nominal):
    nav = run(tmp, "nav_strong", attack("navigation", fraction=1.0))
    c = cascade(nav)
    win = lambda d: pd.read_parquet(d / "fine.parquet").query("450 <= t_fine < 570 and hub == 'H1'")  # 07:30-09:30
    assert win(nav).arrivals.sum() > 1.5 * win(nominal).arrivals.sum(), "surge at the target hub"
    assert win(nav).spillback_m.max() > win(nominal).spillback_m.max()
    # measured finding: the surge stays in transport in the protection-only baseline
    assert c["depth"] == 1 and c["failing"] in ("", "transport")
    assert pd.isna(c["tau_detect_ids_s"]), "charging-side IDS is blind to navigation manipulation"


def test_protection_trips_after_sustained_overload():
    cfg = {"grid": {"protection": {"trip_loading_pct": 120, "trip_after_s": 300, "offline_s": 1800}}}
    p = TransformerProtection(cfg, 60.0)
    tripped = [p.update(k, {"H1": 125.0}) for k in range(5)]
    assert tripped == [[], [], [], [], ["H1"]]
    assert p.is_offline("H1", 5) and p.is_offline("H1", 34) and not p.is_offline("H1", 35)
    p2 = TransformerProtection(cfg, 60.0)
    assert not any(p2.update(k, {"H1": 125.0 if k % 2 else 100.0}) for k in range(20)), "must be sustained"


def test_c57_91_protection_trips_on_hot_spot_not_on_brief_overload():
    th = {"model": "c57_91", "trip_hotspot_c": 140.0, "offline_s": 1800, "ambient_c": 30.0}
    warm = lambda p: [p.update(k, {"H1": 80.0}, 30.0) for k in range(240)]  # settled at 80 %
    p = TransformerProtection({"grid": {"protection": th}}, 60.0)
    warm(p)
    assert not any(p.update(240 + k, {"H1": 130.0}, 30.0) for k in range(10)), "10 min at 130 % is ridden through"
    p = TransformerProtection({"grid": {"protection": th}}, 60.0)
    warm(p)
    trips = [k for k in range(600) if p.update(240 + k, {"H1": 150.0}, 40.0)]
    assert trips and 10 < trips[0] < 300, "a sustained heavy overload heats the winding to the trip"
    assert p.hotspot["H1"] < 140.0, "it cools while the breaker is open"


def test_sustained_helper():
    s = pd.Series([True, True, False, True, True, True])
    assert _sustained(s, 3).tolist() == [False, False, False, False, False, True]
