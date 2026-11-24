import itertools

import numpy as np
import pandas as pd
from PIL import Image

from seams.analysis import make_figures as MF

STRENGTHS = {"navigation": [0.25, 0.5, 0.75, 1.0], "fdi": [-0.005, -0.01, -0.02, -0.04],
             "aggregator": [0.25, 0.5, 0.75, 1.0]}
ORIGIN = {"navigation": "transport", "fdi": "grid", "aggregator": "charging"}


def fake(conditions, scheds, seeds, extra_axes=None):
    rng = np.random.default_rng(0)
    rows = []
    axes = extra_axes or {}
    for cond, sched, seed, *ax in itertools.product(conditions, scheds, seeds, *axes.values()):
        axv = dict(zip(axes, ax))
        rows.append(dict(condition=cond, scheduler=sched, seed=seed, scenario="nominal", strength=None,
                         false_incidents=int(rng.integers(0, 2)), total_unserved_kWh=780 + rng.normal(0, 10),
                         refused_vehicles=0, trip_delay_s=0.0, trip_delay_n=0, **axv))
        for s, levels in STRENGTHS.items():
            for lv in levels:
                cross = cond == "protection_only" and abs(lv) >= abs(levels[1])
                rows.append(dict(
                    condition=cond, scheduler=sched, seed=seed, scenario=s, strength=lv, origin=ORIGIN[s],
                    failing="charging" if cross and s != "navigation" else "", depth=2 if cross else 1,
                    tau_cross_s=420.0 if cross else np.nan, tau_detect_s=240.0 if cond != "protection_only" else np.nan,
                    tau_detect_ids_s=330.0 if s == "fdi" else np.nan, tau_reneg_s=0.0,
                    attribution_ok=cond != "protection_only", total_unserved_kWh=800 + 50 * abs(lv) + rng.normal(0, 10),
                    refused_vehicles=int(10 * abs(lv)), trip_delay_s=1000.0, trip_delay_n=1200, **axv))
    return pd.DataFrame(rows)


def test_make_figures_one_command(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    fake(["protection_only", "static_contracts", "contracts"], ["EDF", "LLF", "RR"], [1, 2, 3]).to_parquet(res / "rq1_summary.parquet")
    r2 = pd.concat([fake(["contracts"], ["EDF"], [1, 2], {"c": [0.6, 1.0, 1.25]}),
                    fake(["protection_only", "oracle"], ["EDF"], [1, 2])])
    r2.to_parquet(res / "rq2_summary.parquet")
    d = fake(["contracts"], ["EDF"], [1, 2, 3], {"delta": [60, 120, 300]})
    d = d[(d.scenario == "nominal") | (d.strength.abs().isin([0.04, 1.0]))]
    d.to_parquet(res / "delta_summary.parquet")
    made = MF.main(["--results", str(res), "--out", str(tmp_path / "fig")])
    names = {p.stem for p in made}
    for s in ("navigation", "fdi", "aggregator"):
        assert {f"rq1_cascade_depth_{s}", f"rq1_containment_by_scheduler_{s}", f"rq2_frontier_{s}_total_unserved_kWh"} <= names
    assert {"rq1_detection_time", "rq1_attribution_accuracy", "rq1_false_attribution", "rq1_containment_window_fdi",
            "rq2_nominal_cost_total_unserved_kWh", "delta_containment", "delta_window_margin_fdi"} <= names
    for p in made:
        assert p.suffix == ".png"
        assert Image.open(p).size == (1000, 800), p.name  # 5 x 4 in at 200 dpi, one axes


def test_pdf_switch(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    fake(["protection_only", "contracts"], ["EDF"], [1]).to_parquet(res / "rq1_summary.parquet")
    made = MF.main(["--results", str(res), "--out", str(tmp_path / "fig"), "--pdf"])
    assert any(p.suffix == ".pdf" for p in made)
