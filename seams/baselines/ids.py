"""IDS for the protection-only baseline. An alarm counts as detection only: it triggers no
containment action, which is the gap contracts fill.

SpearIDS: SPEAR's champion detector (Transformer IDS, SPEAR/models/best_ids_model.pkl: AUC 0.989,
FPR 0.13 % per window on SPEAR's test set). It scores one charging session at a time over 10
consecutive 1-minute samples of a 14-feature vector, as in SPEAR's training
(compare_ids_models_update._evcs_features_14d over ACNDataLoader.sample_benign_window):

   0 soc               energy delivered / (60 kWh * 0.8)        (SPEAR's derive_soc)
   1 voltage / 240     charger voltage                          per-unit adapted, see below
   2 current / 32      charger current [A]
   3 power / 7.68      charger power                            per-unit adapted
   4 temperature / 50  25 + 10 * power/7.68 (degC)              derived as in training
   5 demand_factor     power/7.68
   6 load_factor       power/7.68
   7 grid_voltage      reported hub voltage V_rep [pu]          (training: always 1.0)
   8 frequency / 60    60 Hz
   9 queue / 10        vehicles queued at the hub               (training: always 3)
  10 utilization       power/7.68
  11 urgency           1 + 0.5 (1 - soc)
  12 time of day / 24
  13 system id / 10    hub index 1..3

Per-unit adaptation: our chargers are 208 V / 32 A / 6.66 kW; SPEAR's training data is 240 V /
32 A / 7.68 kW. Voltage and power enter as per-unit of the charger's own rating mapped onto the
training rating, so a charger at its rating looks like a training charger at its rating.
Features 7 and 9 carry what a charging-side IDS would actually see, although training only saw
constants there; any resulting false alarms are measured, and the threshold can be recalibrated
on calibration seeds (never the model). A hub alarms when any of its sessions does.

PlaceholderIDS: a charging-side rule (hub power above its own limit, or an EVSE above rating).
"""
import os
import pickle
import sys
from collections import deque

import numpy as np

from seams.cosim.config import resolve_path

TRAIN_V, TRAIN_P_KW = 240.0, 7.68
SEQ_LEN = 10


def import_spear(cfg):
    """Make SPEAR's modules importable and its pickled classes resolvable."""
    spear_dir = str(resolve_path(cfg["ids"].get("spear_dir", "SPEAR")))
    if spear_dir not in sys.path:
        sys.path.insert(0, spear_dir)
    import __main__
    import ids_neural_models as m
    for cls in ("TransformerIDS", "TransformerIDSWrapper", "AutoencoderIDS", "LSTMIDSBenchmark"):
        if not hasattr(__main__, cls):  # bundles were pickled from a __main__ script
            setattr(__main__, cls, getattr(m, cls))
    return m


class SessionWindows:
    """Per-session 10 x 14 feature windows, built every fine step (shared by the detector and
    the training-data recorder, so both see identical windows)."""

    def __init__(self, cfg):
        self.evse_v = float(cfg["evse"]["voltage_v"])
        self.p_rated = self.evse_v * float(cfg["evse"]["max_rate_a"]) / 1000.0
        self.hub_index = {h["id"]: i + 1 for i, h in enumerate(cfg["hubs"])}
        self.windows = {}  # session id -> deque of feature vectors

    def _features(self, ev, rate_a, v_pu, v_rep, queue, hour, hub):
        p_kw = rate_a * self.evse_v / 1000.0
        p = min(TRAIN_P_KW, TRAIN_P_KW * p_kw / self.p_rated)  # per-unit adapted power [kW]
        soc = float(np.clip(ev.energy_delivered / (60.0 * 0.8), 0.0, 1.0))
        lf = p / TRAIN_P_KW
        return np.array([
            soc, np.clip(TRAIN_V * v_pu, 200.0, 270.0) / 240.0, min(rate_a, 32.0) / 32.0, p / 7.68,
            np.clip(25.0 + lf * 10.0, 20.0, 45.0) / 50.0, lf, lf, v_rep, 1.0, queue / 10.0, min(lf, 1.0),
            np.clip(1.0 + (1.0 - soc) * 0.5, 0.5, 2.0), hour / 24.0, self.hub_index[hub] / 10.0,
        ], dtype=np.float32)

    def update(self, t_s, ctx):
        """ctx: hub -> {'hub': AcnHub, 'v_true', 'v_rep', 'queue'}.
        Returns [(hub, session id, flat 140-vector)] for every session with a full window."""
        hour = (t_s / 3600.0) % 24.0
        live, full = set(), []
        for h, c in ctx.items():
            hub = c["hub"]
            for sid, ev in hub.evs.items():
                if ev.station_id not in hub.network.station_ids or hub.network.get_ev(ev.station_id) is not ev:
                    continue  # plug-in not processed yet
                live.add(sid)
                rate = float(hub.last_rates_a[hub.station_ids.index(ev.station_id)])
                win = self.windows.setdefault(sid, deque(maxlen=SEQ_LEN))
                win.append(self._features(ev, rate, c["v_true"], c["v_rep"], c["queue"], hour, h))
                if len(win) == SEQ_LEN:
                    full.append((h, sid, np.concatenate(win)))
        for sid in [s for s in self.windows if s not in live]:
            del self.windows[sid]  # session ended
        return full


class SpearIDS:
    """SPEAR's Transformer IDS. model_path selects the bundle: SPEAR's pre-trained one
    (ACN-Data; does not transfer to this testbed) or the testbed-retrained one
    (seams.experiments.train_ids: same architecture and recipe, calibration seeds only)."""
    name = "spear_transformer"

    def __init__(self, cfg):
        c = cfg["ids"]
        import_spear(cfg)
        with open(resolve_path(c["model_path"]), "rb") as f:
            bundle = pickle.load(f)
        self.model, self.scaler = bundle["model"], bundle["scaler"]
        self.threshold = float(c["threshold"]) if c.get("threshold") is not None else float(bundle["threshold"])
        self.model_name = bundle.get("model_name")
        self.sessions = SessionWindows(cfg)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    def observe(self, t_s, ctx):
        """Returns hub -> (alarm, max session score)."""
        full = self.sessions.update(t_s, ctx)
        out = {h: (False, 0.0) for h in ctx}
        if full:
            proba = self.model.predict_proba(self.scaler.transform(np.stack([x for _, _, x in full])))[:, 1]
            for (h, _, _), p in zip(full, proba):
                if p > out[h][1]:
                    out[h] = (bool(p >= self.threshold), float(p))
        return out


class WindowRecorder:
    """Training-data mode: records every full session window, raises no alarm."""
    name = "recorder"

    def __init__(self, cfg):
        self.sessions = SessionWindows(cfg)
        self.records = []  # (t_s, hub, session id, flat vector)

    def observe(self, t_s, ctx):
        for h, sid, x in self.sessions.update(t_s, ctx):
            self.records.append((t_s, h, sid, x))
        return {h: (False, 0.0) for h in ctx}


class PlaceholderIDS:
    name = "placeholder"

    def __init__(self, cfg):
        self.tol = float(cfg["ids"].get("rate_tol", 0.02))

    def observe(self, t_s, ctx):
        out = {}
        for h, c in ctx.items():
            hub = c["hub"]
            limit_kw = hub.limit_a * hub.voltage_v / 1000.0
            over_limit = hub.power_kw() / max(limit_kw, 1e-9) - 1.0
            over_rating = (float(hub.last_rates_a.max()) if len(hub.last_rates_a) else 0.0) / hub.max_rate_a - 1.0
            score = max(over_limit, over_rating)
            out[h] = (score > self.tol, score)
        return out


def make_ids(cfg):
    """ids.kind: one detector, or several joined with '+' (e.g. spear+bdd)."""
    kind = cfg.get("ids", {}).get("kind")
    if kind is None:
        return None
    if "+" in kind:
        from seams.baselines.bdd import CompositeIDS
        return CompositeIDS([make_ids({**cfg, "ids": {**cfg["ids"], "kind": k}}) for k in kind.split("+")])
    if kind == "bdd":
        from seams.baselines.bdd import BadDataDetector
        return BadDataDetector(cfg)
    if kind == "placeholder":
        return PlaceholderIDS(cfg)
    if kind == "spear":
        return SpearIDS(cfg)
    if kind == "recorder":
        return WindowRecorder(cfg)
    raise ValueError(f"unknown IDS {kind!r}")
