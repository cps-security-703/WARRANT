"""Gymnasium environments for the two-layer attacker, retargeted to the seams.

DQN layer (ArmEnv): one decision per episode, the discrete arm (target, onset slot, duration);
its reward is the return of a whole episode played by the current (frozen) SAC policy. The
grey-box attacker does not know the day, so its observation is constant: a bandit.
SAC layer (AttackEnv): per decision step, the continuous strength (surface-specific, two dims),
conditioned on the arm; arms are drawn from the DQN layer's current preferences (arm_probs file).
Each env instance owns one seed's SnapshotWorker (seams.attacks.adaptive.episode).
"""
import json
import os
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np

from seams.cosim.config import load_config

N_HUBS = 3


def arms(rl):
    return [dict(target=t, onset_s=o, duration_s=d)
            for t, o, d in product(rl["targets"][rl["surface"]], sorted(rl["onsets_s"]), sorted(rl["durations_s"]))]


def arm_features(rl, arm):
    tg = rl["targets"][rl["surface"]]
    f = [1.0 if arm["target"] == t else 0.0 for t in tg]
    f.append((arm["onset_s"] % 86400) / 86400)
    f.append(arm["duration_s"] / max(rl["durations_s"]))
    return f


def obs_dim(rl):
    cfg = load_config(rl["config"], rl.get("overrides") or {})
    home = bool(cfg["grid"].get("home_ev", {}).get("enabled"))
    n = len(cfg["hubs"])
    grey = {"fdi": 3, "aggregator": 4 * n + (2 if home else 0) + 2, "navigation": 2 * n + 2}[rl["surface"]]
    white = (n + (1 if home else 0) + 3) if rl["box"] == "white" else 0
    return 3 + grey + white + len(rl["targets"][rl["surface"]]) + 2


class AttackEnv(gym.Env):
    def __init__(self, rl, seed, run_root, arm_probs_file=None, fixed_arm=None):
        self.rl, self.seed_, self.run_root = rl, seed, run_root
        self.arms = arms(rl)
        self.arm_probs_file, self.fixed_arm = arm_probs_file, fixed_arm
        self.worker = None
        d = obs_dim(rl)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (d,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.arm = None

    def _worker(self):
        if self.worker is None:
            from seams.attacks.adaptive.episode import SnapshotWorker
            self.worker = SnapshotWorker(self.rl, self.seed_, self.run_root)
        return self.worker

    def _probs(self):
        if self.arm_probs_file and os.path.exists(self.arm_probs_file):
            p = np.asarray(json.loads(Path(self.arm_probs_file).read_text()), dtype=float)
            if len(p) == len(self.arms) and p.sum() > 0:
                return p / p.sum()
        return np.full(len(self.arms), 1.0 / len(self.arms))

    def _wrap(self, o):
        return np.concatenate([o, np.asarray(arm_features(self.rl, self.arm), np.float32)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "arm" in options:
            self.arm = options["arm"]
        elif self.fixed_arm is not None:
            self.arm = self.fixed_arm
        else:
            self.arm = self.arms[self.rng.choice(len(self.arms), p=self._probs())]
        return self._wrap(self._worker().reset(self.arm)), {"arm": self.arm}

    def step(self, action):
        o, r, done, info = self._worker().step(action)
        if done:
            info = dict(info, arm=self.arm)
        return self._wrap(o), r, done, False, info

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


class ArmEnv(gym.Env):
    """One-step bandit over arms; the episode is played by the SAC policy saved at sac_path."""

    def __init__(self, rl, seed, run_root, sac_path):
        self.inner = AttackEnv(rl, seed, run_root)
        self.n = len(self.inner.arms)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(self.n)
        self.sac_path, self._sac, self._mtime = sac_path, None, None

    def _policy(self):
        mtime = os.path.getmtime(self.sac_path) if os.path.exists(self.sac_path) else None
        if mtime is not None and mtime != self._mtime:
            import torch
            from stable_baselines3 import SAC
            torch.set_num_threads(1)
            self._sac, self._mtime = SAC.load(self.sac_path, device="cpu"), mtime
        return self._sac

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.ones(1, np.float32), {}

    def step(self, action):
        arm = self.inner.arms[int(action)]
        o, _ = self.inner.reset(options={"arm": arm})
        pol, total, done, info = self._policy(), 0.0, False, {}
        while not done:
            a = pol.predict(o, deterministic=True)[0] if pol is not None else self.inner.action_space.sample()
            o, r, done, _, info = self.inner.step(a)
            total += r
        return np.ones(1, np.float32), float(total), True, False, dict(info, arm=arm)

    def close(self):
        self.inner.close()
