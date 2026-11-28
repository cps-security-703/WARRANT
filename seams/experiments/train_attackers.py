"""Phase 7 (RQ3): train SPEAR's two-layer attacker per surface, then evaluate the frozen policies
on held-out seeds.

    python -m seams.experiments.train_attackers seams/config/phase7.yaml \
        --surface fdi --box grey --mode within [--set verifier.use_epsilon=false] [--tag noeps]

Training alternates `rounds` times: SAC on episodes whose arms are drawn from the DQN layer's
current preferences, then DQN with that SAC policy frozen. One env per training seed (a
SnapshotWorker each). Output in ~/seams_runs/rq3/<name>/: sac.zip, dqn.zip, arm_probs.json,
learning curves (sac_monitor.csv, dqn_monitor.csv), eval/ runs and eval.parquet.

Evaluation: per held-out seed, the DQN's greedy arm played by the deterministic SAC policy,
continued to the end of the day, next to its no-attack twin from the same snapshot; the
campaign metrics (seams.experiments.campaign.run_metrics) on both.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # os.fork in env workers (macOS)


def _env_fn(kind, rl, seed, run_root, path):
    def make():
        import torch
        torch.set_num_threads(1)
        from seams.attacks.adaptive.env import ArmEnv, AttackEnv
        return AttackEnv(rl, seed, run_root, arm_probs_file=path) if kind == "sac" else ArmEnv(rl, seed, run_root, path)
    return make


def _vec(kind, rl, out, path, monitor):
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    fns = [_env_fn(kind, rl, s, str(out / "work"), path) for s in rl["train_seeds"]]
    env = SubprocVecEnv(fns, start_method="spawn")
    return VecMonitor(env, filename=str(monitor),
                      info_keywords=("damage", "E", "Q", "R", "violations", "consumed", "detected", "ids"))


def arm_probs(dqn, n, temperature, floor):
    import torch
    with torch.no_grad():
        q = dqn.q_net(torch.ones(1, 1)).numpy()[0]
    z = (q - q.max()) / max(temperature, 1e-6)
    p = np.exp(z) / np.exp(z).sum()
    return ((1 - floor) * p + floor / n).tolist(), q.tolist()


def train(rl, out):
    from stable_baselines3 import DQN, SAC
    from seams.attacks.adaptive.env import arms
    tr = rl["training"]
    n_arms = len(arms(rl))
    probs_file, sac_path, dqn_path = out / "arm_probs.json", out / "sac.zip", out / "dqn.zip"
    sac = dqn = None
    done = len(list(out.glob("q_r*.json")))  # rounds already trained (--resume-from)
    for rnd in range(done, int(tr["rounds"])):
        env = _vec("sac", rl, out, str(probs_file), out / f"sac_monitor_r{rnd}")
        if sac is None and done:
            # the replay buffer is not kept between runs: SAC refills it (learning_starts) first
            sac = SAC.load(sac_path, env, device="cpu")
        elif sac is None:
            sac = SAC("MlpPolicy", env, learning_starts=500, batch_size=256, gradient_steps=-1,
                      train_freq=1, seed=0, device="cpu", verbose=0)
        else:
            sac.set_env(env)
        sac.learn(int(tr["sac_steps"]), reset_num_timesteps=False)
        sac.save(sac_path)
        env.close()
        env = _vec("dqn", rl, out, str(sac_path), out / f"dqn_monitor_r{rnd}")
        if dqn is None and done:
            dqn = DQN.load(dqn_path, env, device="cpu")
        elif dqn is None:
            dqn = DQN("MlpPolicy", env, learning_starts=n_arms, buffer_size=10000, batch_size=64,
                      train_freq=1, gradient_steps=4, target_update_interval=50,
                      exploration_fraction=0.6, exploration_final_eps=0.1, learning_rate=1e-3,
                      policy_kwargs=dict(net_arch=[64]), seed=0, device="cpu", verbose=0)
        else:
            dqn.set_env(env)
        dqn.learn(int(tr["dqn_episodes"]), reset_num_timesteps=False)
        dqn.save(dqn_path)
        env.close()
        p, q = arm_probs(dqn, n_arms, tr["arm_temperature"], tr["arm_floor"])
        probs_file.write_text(json.dumps(p))
        (out / f"q_r{rnd}.json").write_text(json.dumps(q))
        print(f"round {rnd}: best arm {arms(rl)[int(np.argmax(q))]} q={max(q):.2f}", flush=True)
    return sac_path, dqn_path


def _eval_one(job):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from seams.attacks.adaptive.env import AttackEnv
    rl, seed, arm, sac_path, out = job
    rl = dict(rl, onsets_s=[arm["onset_s"]], continue_to_end=True)
    env = AttackEnv(rl, seed, str(out / "work"), fixed_arm=arm)
    pol = SAC.load(sac_path, device="cpu")
    o, _ = env.reset()
    done, ret, info, actions = False, 0.0, {}, []
    while not done:
        a = pol.predict(o, deterministic=True)[0]
        actions.append(a.tolist())
        o, r, done, _, info = env.step(a)
        ret += r
    att, nom = out / "eval" / f"attack_s{seed}", out / "eval" / f"nominal_s{seed}"
    env.worker.write(att)
    env.worker.nominal(arm["onset_s"], nom)
    env.close()
    row = {"seed": seed, **{f"arm_{k}": v for k, v in arm.items()}, "return": ret,
           **{f"ep_{k}": v for k, v in info.items() if k != "arm"}, "actions": json.dumps(actions)}
    return row, str(att), str(nom)


def evaluate(rl, out, sac_path, dqn_path, workers):
    from stable_baselines3 import DQN
    from seams.attacks.adaptive.env import arms
    dqn = DQN.load(dqn_path, device="cpu")
    q = arm_probs(dqn, len(arms(rl)), 1.0, 0.0)[1]
    arm = arms(rl)[int(np.argmax(q))]
    jobs = [(rl, s, arm, str(sac_path), out) for s in rl["eval_seeds"]]
    # one job per process: a process that ran libsumo cannot read parquet (see sumo_env)
    import multiprocessing as mp
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        results = list(pool.map(_eval_one, jobs))
    from seams.experiments.campaign import run_metrics
    rows = []
    for row, att, nom in results:
        for tag, d in (("att", att), ("nom", nom)):
            row.update({f"{tag}_{k}": v for k, v in run_metrics(d).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "eval.parquet", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--surface", required=True, choices=["fdi", "aggregator", "navigation"])
    ap.add_argument("--box", default="grey", choices=["grey", "white"])
    ap.add_argument("--mode", default="within", choices=["within", "inter"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="~/seams_runs/rq3")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--rl", action="append", default=[], metavar="KEY=VALUE",
                    help="override the rl block, dotted (e.g. reward.detection_penalty=0)")
    ap.add_argument("--resume-from", default=None, metavar="NAME",
                    help="continue training that attacker (under --out) to training.rounds, in this run's directory")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args(argv)
    rl = yaml.safe_load(Path(args.config).read_text())["rl"]
    for key, v in (s.split("=", 1) for s in args.rl):
        *path, last = key.split(".")
        node = rl
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(v)
    overrides = dict(rl.get("overrides") or {})
    overrides.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    rl.update(surface=args.surface, box=args.box, mode=args.mode, overrides=overrides)
    name = f"{args.mode}_{args.surface}_{args.box}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out).expanduser() / name
    out.mkdir(parents=True, exist_ok=True)
    if args.resume_from and not args.eval_only:
        import shutil
        src = Path(args.out).expanduser() / args.resume_from
        for f in [*src.glob("*.zip"), *src.glob("*.json"), *src.glob("*.monitor.csv")]:
            shutil.copy2(f, out / f.name)
    (out / "rl.yaml").write_text(yaml.safe_dump(rl, sort_keys=False))
    if args.eval_only:
        sac_path, dqn_path = out / "sac.zip", out / "dqn.zip"
    else:
        sac_path, dqn_path = train(rl, out)
    df = evaluate(rl, out, sac_path, dqn_path, args.workers)
    cols = [c for c in ("ep_detected", "ep_damage", "ep_violations", "att_total_unserved_kWh",
                        "nom_total_unserved_kWh", "att_first_blame") if c in df]
    print(df[["seed", "arm_target", "arm_onset_s", "arm_duration_s"] + cols].to_string(), flush=True)


if __name__ == "__main__":
    main()
