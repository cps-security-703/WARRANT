"""Retrain SPEAR's Transformer IDS on this testbed, with SPEAR's own recipe.

    python -m seams.experiments.train_ids seams/config/phase2.yaml

SPEAR's pre-trained IDS does not transfer here (it alarms on ~99 % of nominal windows: our
sessions are out of its ACN-Data distribution). So the baseline IDS is the same architecture and
recipe, trained in-domain:

  benign windows   this testbed's nominal days, per-session 10 x 14 windows (SessionWindows),
                   calibration seeds only: all but the last for training, the last held out
  attack windows   SPEAR's six synthetic FDI injections (_inject_fdi_14d) applied to benign
                   windows from the last half of the window on, exactly as SPEAR builds them;
                   never our scripted attacks, so the IDS is not trained on what it is scored on
  model            SPEAR's train_transformer_ids (TransformerIDS, AdamW, 80 epochs, cosine)
  scaler           StandardScaler fit on the training split
  threshold        SPEAR's rule: Youden J on the held-out split, clipped to [0.10, 0.99]
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from seams.cosim.config import load_config, resolve_path


def _record(job):
    from seams.cosim.master import Master
    cfg_path, seed, run_dir = job
    cfg = load_config(cfg_path, {"run.seed": seed, "attacks": [], "ids.kind": "recorder"})
    m = Master(cfg, f"ids_s{seed}", run_dir=run_dir)
    m.run()
    return seed, np.stack([x for _, _, _, x in m.ids.records]).astype(np.float32)


def spear_dataset(pool, n, attack_ratio, rng, C):
    """SPEAR's construction (compare_ids_models_update.build_dataset) over our window pool."""
    seq, feat = C.SEQ_LEN, C.FEATURE_DIM
    n_attack = int(n * attack_ratio)
    X, y = [], []
    for i in range(n):
        w = pool[rng.randint(len(pool))].reshape(seq, feat).copy()
        if i >= n - n_attack:
            atype = int(rng.choice(6))
            inject_at = rng.randint(seq // 2, seq)
            for t in range(inject_at, seq):
                step_frac = (t - inject_at) / max(seq - inject_at - 1, 1)
                w[t] = C._inject_fdi_14d(w[t], atype, step_frac=step_frac)
            y.append(1)
        else:
            y.append(0)
        X.append(w)
    X, y = np.stack(X).astype(np.float32), np.array(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    t = cfg["ids"]["training"]
    seeds = list(t["seeds"])
    out_root = resolve_path(cfg["run"]["out_dir"]) / "ids_training"
    jobs = [(args.config, s, out_root / f"s{s}") for s in seeds]
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
        pools = dict(pool.map(_record, jobs))

    from seams.baselines.ids import import_spear
    import_spear(cfg)
    import compare_ids_models_update as C
    import torch
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(int(t["random_state"]))
    rng = np.random.RandomState(int(t["random_state"]))
    train_pool = np.concatenate([pools[s] for s in seeds[:-1]])
    test_pool = pools[seeds[-1]]
    n_test = int(t["n_samples"] * t["test_frac"])
    X_tr, y_tr = spear_dataset(train_pool, t["n_samples"] - n_test, t["attack_ratio"], rng, C)
    X_ts, y_ts = spear_dataset(test_pool, n_test, t["attack_ratio"], rng, C)

    scaler = StandardScaler()
    Xf_tr = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
    Xf_ts = scaler.transform(X_ts.reshape(len(X_ts), -1))
    model = C.train_transformer_ids(Xf_tr.reshape(X_tr.shape), y_tr, epochs=int(t["epochs"]))
    wrapper = C.TransformerIDSWrapper(model)
    proba = wrapper.predict_proba(Xf_ts)[:, 1]
    fpr_c, tpr_c, thr_c = roc_curve(y_ts, proba)
    best = int(np.argmax(tpr_c - fpr_c))
    thr = float(np.clip(thr_c[best] if best < len(thr_c) else 0.5, 0.10, 0.99))
    pred = proba >= thr
    meta = {
        "model_name": "Transformer IDS (testbed-retrained, SPEAR recipe)",
        "threshold": thr,
        "roc_auc": float(auc(fpr_c, tpr_c)),
        "recall": float(pred[y_ts == 1].mean()),
        "fpr": float(pred[y_ts == 0].mean()),
        "train_seeds": seeds[:-1], "test_seed": seeds[-1],
        "train_windows_pool": int(len(train_pool)), "n_train": int(len(X_tr)), "n_test": int(len(X_ts)),
        "epochs": int(t["epochs"]), "feature_dim": int(C.FEATURE_DIM), "seq_len": int(C.SEQ_LEN),
    }
    path = resolve_path(cfg["ids"]["retrained_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": wrapper, "scaler": scaler, **meta}, f, protocol=4)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    for d in out_root.glob("*"):
        for f in d.rglob("*"):
            if f.is_file() or f.is_symlink():
                f.unlink()
    print(yaml.safe_dump(meta, sort_keys=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
