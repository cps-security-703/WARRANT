import dataclasses

import pandas as pd
import pytest

from seams.cosim.config import load_config, with_overrides
from seams.cosim.master import Master

BASE = load_config("seams/config/phase1.yaml")
N_EVSE = {h["id"]: h["n_evse"] for h in BASE["hubs"]}
HALF_DAY = {"run.horizon_s": 43200}


def run(tmp, name, overrides=None, setup=None):
    master = Master(with_overrides(BASE, overrides or {}), name, run_dir=tmp / name)
    if setup:
        setup(master)
    run_dir = master.run()
    return {t: pd.read_parquet(run_dir / f"{t}.parquet") for t in ("fine", "contract", "session", "events")}


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("phase1")


@pytest.fixture(scope="module")
def nominal(tmp):
    return run(tmp, "nominal")


@pytest.fixture(scope="module")
def half_day(tmp):
    return run(tmp, "half", HALF_DAY)


def test_nominal_day_keeps_phi_safe(nominal):
    f = nominal["fine"]
    assert f.converged.all()
    assert (f.margin_v >= 0).all(), f"min voltage margin {f.margin_v.min():.4f}"
    assert (f.margin_loading >= 0).all(), f"max loading {f.xfmr_loading.max():.1f}%"


def test_bridge_invariants_full_fleet(nominal):
    s, f, ev = nominal["session"], nominal["fine"], nominal["events"]
    done = s[s.released_reason != "active_at_end"]
    gain_kwh = (done.charge_after_release_wh - done.charge_at_plugin_wh) / 1000.0
    assert ((gain_kwh - done.delivered_kWh).abs() <= 0.01 * done.delivered_kWh.clip(lower=1e-9) + 1e-9).all()
    assert (done.charge_before_release_wh == done.charge_at_plugin_wh).all()
    assert (done.released_k <= done.departure).all()
    assert (f.occupied_evse <= f.hub.map(N_EVSE)).all()
    released = ev[ev.event.str.startswith("release")].vehicle_id
    assert released.is_unique and set(released) == set(done.vehicle_id)
    # drivers stay their real dwell (bridge.release_on_complete: false): every release is a departure
    expected = {"completed", "departure"} if BASE["bridge"].get("release_on_complete", True) else {"departure"}
    assert set(done.released_reason) <= expected and "departure" in set(done.released_reason)


def test_full_hub_produces_spillback(nominal):
    f = nominal["fine"]
    full = f.occupied_evse == f.hub.map(N_EVSE)
    for hub in N_EVSE:
        at = f[f.hub == hub]
        assert at.spillback_m.max() > 0, f"{hub} never spilled back"
    # queues form when the hub is full; a space freed this minute is taken next minute,
    # so allow a one-port lag
    nearly_full = f.occupied_evse >= f.hub.map(N_EVSE) - 1
    assert nearly_full[f.queue > 0].mean() > 0.95
    assert full[f.queue > 0].mean() > 0.7


def test_derating_changes_schedules(tmp, half_day):
    def falsify(m):
        m.telemetry.falsification["H1"] = -0.05
    derated = run(tmp, "derated", HALF_DAY, falsify)
    base = half_day["fine"].groupby("hub")[["P_pcc", "d_h"]].mean()
    test = derated["fine"].groupby("hub")[["P_pcc", "d_h"]].mean()
    assert test.loc["H1", "d_h"] < 0.8 * base.loc["H1", "d_h"]
    # real sessions need little energy, so demand often sits below the derated cap: less, not 20 % less
    assert test.loc["H1", "P_pcc"] < base.loc["H1", "P_pcc"]
    assert (derated["fine"].query("hub == 'H1'").P_pcc <= derated["fine"].query("hub == 'H1'").d_h.shift().fillna(1e9) + 1e-6).all()


def test_unavailable_hub_changes_routing(tmp, half_day):
    def hide_h2(m):
        m.view_hooks.append(lambda views, vid: {
            h: dataclasses.replace(v, availability=-100) if h == "H2" else v for h, v in views.items()
        })
    hidden = run(tmp, "hidden", HALF_DAY, hide_h2)
    base_n = (half_day["session"].hub == "H2").sum()
    test_n = (hidden["session"].hub == "H2").sum()
    assert test_n < 0.5 * base_n, (base_n, test_n)


def test_bit_identical_replay(tmp):
    short = {"run.horizon_s": 21600}
    a, b = run(tmp, "rep_a", short), run(tmp, "rep_b", short)
    for table in a:
        pd.testing.assert_frame_equal(a[table].drop(columns="run_id"), b[table].drop(columns="run_id"))
