import os
import pickle
import sys
from collections import deque

import numpy as np

from seams.cosim.config import resolve_path

TRAIN_V, TRAIN_P_KW = 240.0, 7.68
SEQ_LEN = 10


def import_adaptive_rl(cfg):
    adaptive_rl_dir = str(resolve_path(cfg["ids"].get("adaptive_rl_dir", "adaptive_rl")))
    if adaptive_rl_dir not in sys.path:
        sys.path.insert(0, adaptive_rl_dir)
    import __main__
    import ids_neural_models as m
    for cls in ("TransformerIDS", "TransformerIDSWrapper", "AutoencoderIDS", "LSTMIDSBenchmark"):
        if not hasattr(__main__, cls):  # bundles were pickled from a __main__ script
            setattr(__main__, cls, getattr(m, cls))
    return m


class SessionWindows:
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


class TransformerSessionIDS:
    name = "transformer"

    def __init__(self, cfg):
        c = cfg["ids"]
        import_adaptive_rl(cfg)
        with open(resolve_path(c["model_path"]), "rb") as f:
            bundle = pickle.load(f)
        self.model, self.scaler = bundle["model"], bundle["scaler"]
        self.threshold = float(c["threshold"]) if c.get("threshold") is not None else float(bundle["threshold"])
        self.model_name = bundle.get("model_name")
        self.sessions = SessionWindows(cfg)
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    def observe(self, t_s, ctx):
        full = self.sessions.update(t_s, ctx)
        out = {h: (False, 0.0) for h in ctx}
        if full:
            proba = self.model.predict_proba(self.scaler.transform(np.stack([x for _, _, x in full])))[:, 1]
            for (h, _, _), p in zip(full, proba):
                if p > out[h][1]:
                    out[h] = (bool(p >= self.threshold), float(p))
        return out


class WindowRecorder:
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
    if kind == "transformer":
        return TransformerSessionIDS(cfg)
    if kind == "recorder":
        return WindowRecorder(cfg)
    raise ValueError(f"unknown IDS {kind!r}")
