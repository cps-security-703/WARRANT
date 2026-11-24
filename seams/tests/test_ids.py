import json

import numpy as np
import pytest

from seams.baselines.bdd import BadDataDetector, CompositeIDS
from seams.baselines.ids import SEQ_LEN, SessionWindows, TransformerSessionIDS, make_ids
from seams.cosim.config import load_config, resolve_path

CFG = load_config("seams/config/phase2.yaml")


def test_retrained_bundle_metadata():
    meta = json.loads(resolve_path(CFG["ids"]["retrained_path"]).with_suffix(".json").read_text())
    assert meta["train_seeds"] + [meta["test_seed"]] == CFG["ids"]["training"]["seeds"]
    assert not set(meta["train_seeds"]) & set(range(42, 52)), "no evaluation or development seeds"
    assert meta["roc_auc"] > 0.95 and 0.10 <= meta["threshold"] <= 0.99


def transformer_ids():
    ids = make_ids(CFG)
    assert isinstance(ids, CompositeIDS)  # transformer+bdd
    return next(d for d in ids.detectors if isinstance(d, TransformerSessionIDS))


def test_detector_uses_bundle_threshold_and_scores_windows():
    ids = transformer_ids()
    meta = json.loads(resolve_path(CFG["ids"]["retrained_path"]).with_suffix(".json").read_text())
    assert ids.threshold == pytest.approx(meta["threshold"])
    x = np.zeros((2, SEQ_LEN * 14), dtype=np.float32)
    p = ids.model.predict_proba(ids.scaler.transform(x))[:, 1]
    assert p.shape == (2,) and np.all((0 <= p) & (p <= 1))


def test_recorder_and_detector_share_feature_construction():
    rec, det = make_ids({**CFG, "ids": {**CFG["ids"], "kind": "recorder"}}), transformer_ids()
    assert type(rec.sessions) is type(det.sessions) is SessionWindows


def test_bdd_flags_lv_falsification_not_consistent_state():
    bdd = BadDataDetector(CFG)
    ctx = {}
    for h, p in bdd.params.items():  # a state exactly on the model: J = 0
        v_mv, kw = 1.0, 50.0
        ctx[h] = {"v_rep": p["a"] + p["b"] * v_mv + p["c"] * kw, "v_mv_rep": v_mv, "p": kw}
    for _ in range(bdd.persist):
        assert not any(a for a, _ in bdd.observe(0.0, ctx).values())
    ctx["H1"] = dict(ctx["H1"], v_rep=ctx["H1"]["v_rep"] - 0.04)  # FDI on the LV report only
    alarms = [bdd.observe(0.0, ctx) for _ in range(bdd.persist)]
    assert alarms[-1]["H1"][0] and not alarms[0]["H1"][0], "alarm only after `persist` steps"
