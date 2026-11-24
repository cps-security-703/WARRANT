import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from seams.cosim.config import load_config
from seams.cosim.master import Master

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
CFG = "seams/config/phase5.yaml"
SHORT = {"run.seed": 2008, "run.horizon_s": 10800, "attacks": [], "sumo.libsumo": True}


def _frames(d):
    return {t: pd.read_parquet(d / f"{t}.parquet").drop(columns=["run_id", "v2_ms"], errors="ignore")
            for t in ("contract", "session", "trips")}


def _isolated(fn, *args):
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        return pool.apply(fn, args)


def _reference(run_dir):
    Master(load_config(CFG, SHORT), "ref", run_dir=run_dir).run()


def _forked(tmp_path):
    m = Master(load_config(CFG, SHORT), "fork", run_dir=tmp_path / "parent")
    m.start()
    m.advance(60)  # 1 h
    pid = os.fork()
    if pid == 0:
        try:
            m.logger.run_dir = tmp_path / "child"
            m.logger.run_dir.mkdir()
            m.advance(m.clock.fine_steps(10800) - m.k)
            m.sumo.close()
            m.finish()
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    m.advance(m.clock.fine_steps(10800) - m.k)
    m.sumo.close()
    m.finish()


def test_forked_cosimulation_continues_identically(tmp_path):
    _isolated(_reference, tmp_path / "ref")
    _isolated(_forked, tmp_path)
    a = _frames(tmp_path / "ref")
    for d in ("parent", "child"):
        b = _frames(tmp_path / d)
        for t in a:
            pd.testing.assert_frame_equal(a[t], b[t], obj=f"{d}/{t}")


def _zero_episode(tmp_path, surface):
    from seams.attacks.adaptive.episode import SnapshotWorker
    rl = yaml.safe_load(open("seams/config/phase7.yaml"))["rl"]
    rl.update(surface=surface, box="grey", mode="within", onsets_s=[1800], durations_s=[600], tail_s=600,
              overrides={"run.horizon_s": 7200})
    w = SnapshotWorker(rl, 2008, str(tmp_path))
    try:
        w.reset(dict(target="H1", onset_s=1800, duration_s=600))
        # fdi: bias 0; aggregator: hold off, fraction from a0=-1 -> 0 (the honest scheduler is
        # replaced, but at night no EV is connected); navigation: fraction 0
        zero = {"fdi": [0.0, 0.0], "aggregator": [-1.0, -1.0], "navigation": [-1.0, -1.0]}[surface]
        done = False
        while not done:
            _, r, done, info = w.step(np.asarray(zero))
        return info
    finally:
        w.close()


@pytest.mark.parametrize("surface", ["fdi", "aggregator", "navigation"])
def test_zero_strength_episode_does_no_damage(tmp_path, surface):
    info = _isolated(_zero_episode, str(tmp_path), surface)
    assert not info["detected"]
    assert abs(info["damage"]) < 1e-9
