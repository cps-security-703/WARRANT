import pandas as pd
import pytest

from seams.cosim.config import load_config, with_overrides
from seams.cosim.master import Master

BASE = load_config("seams/config/phase0.yaml")

# commuter dwell 2 h < time to reach target SOC (~5 h at 7.68 kW) -> leaves under-charged
DEPARTURE = {}
# 6 h dwell and an 8 h horizon -> reaches target SOC and is released as completed
COMPLETED = {"trip_types.commuter.dwell_s": 21600, "run.horizon_s": 28800}


def run(tmp_path, name, overrides):
    cfg = with_overrides(BASE, overrides)
    run_dir = Master(cfg, name, run_dir=tmp_path / name).run()
    return {t: pd.read_parquet(run_dir / f"{t}.parquet") for t in ("fine", "session", "events")}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase0")
    return {
        "departure": run(tmp, "departure", DEPARTURE),
        "completed": run(tmp, "completed", COMPLETED),
    }


@pytest.mark.parametrize("path", ["departure", "completed"])
def test_release_path(runs, path):
    s = runs[path]["session"]
    assert len(s) == 1
    assert s.released_reason.iloc[0] == path
    assert s.vehicle_arrived.iloc[0], "vehicle should resume and finish its trip"


@pytest.mark.parametrize("path", ["departure", "completed"])
def test_energy_conservation(runs, path):
    s = runs[path]["session"].iloc[0]
    sumo_gain_kwh = (s.charge_after_release_wh - s.charge_at_plugin_wh) / 1000.0
    assert sumo_gain_kwh == pytest.approx(s.delivered_kWh, rel=0.01)
    # SUMO must neither charge nor drain the battery while ACN-Sim owns the vehicle
    assert s.charge_before_release_wh == pytest.approx(s.charge_at_plugin_wh, abs=1e-6)
    # energy delivered equals the integral of the logged hub power
    fine = runs[path]["fine"]
    integrated_kwh = fine.loc[fine.hub == s.hub, "P_sched"].sum() / 60.0
    assert integrated_kwh == pytest.approx(s.delivered_kWh, rel=1e-9)


@pytest.mark.parametrize("path", ["departure", "completed"])
def test_no_orphan_and_capacity(runs, path):
    s, ev, fine = runs[path]["session"], runs[path]["events"], runs[path]["fine"]
    plugins = ev[ev.event == "plugin"].vehicle_id.tolist()
    releases = ev[ev.event.str.startswith("release")].vehicle_id.tolist()
    assert sorted(plugins) == sorted(releases), "every plug-in has exactly one release"
    assert (s.released_k <= s.departure).all(), "no vehicle waits past its departure"
    n_evse = {h["id"]: h["n_evse"] for h in BASE["hubs"]}
    assert (fine.occupied_evse <= fine.hub.map(n_evse)).all()


def test_completed_reaches_target(runs):
    s = runs["completed"]["session"].iloc[0]
    assert s.unserved_kWh <= BASE["bridge"]["release_tol_kwh"] + 1e-9


def test_departure_logs_unserved(runs):
    s = runs["departure"]["session"].iloc[0]
    assert s.unserved_kWh > 1.0
    assert s.released_k == s.departure


def test_bit_identical_replay(tmp_path):
    a = run(tmp_path, "a", {})
    b = run(tmp_path, "b", {})
    for table in a:
        pd.testing.assert_frame_equal(a[table].drop(columns="run_id"), b[table].drop(columns="run_id"))
