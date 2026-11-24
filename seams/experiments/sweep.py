import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import yaml

from seams.cosim.config import load_config, resolve_path


def _one(job):
    from seams.analysis.metrics import attribution, availability_cost, cascade
    from seams.cosim.master import Master

    cfg_path, overrides, run_id, run_dir = job
    cfg = load_config(cfg_path, overrides)
    t0 = time.perf_counter()
    Master(cfg, run_id, run_dir=run_dir).run()
    return {"run_id": run_id, "wall_s": time.perf_counter() - t0,
            **{k: v for k, v in overrides.items() if k != "attacks"},
            **cascade(run_dir), **availability_cost(run_dir), **attribution(run_dir)}


def parse_param(spec):
    key, values = spec.split("=", 1)
    return key, [yaml.safe_load(v) for v in values.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--attack")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=V1,V2,...")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    base = {}
    if args.attack:
        base["attacks"] = yaml.safe_load(resolve_path(args.attack).read_text())["attacks"]
    base.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    params = [parse_param(p) for p in args.param]
    seeds = [int(s) for s in args.seeds.split(",")]

    out_root = resolve_path(load_config(args.config)["run"]["out_dir"]) / args.name
    jobs = []
    for combo in itertools.product(*[vals for _, vals in params]):
        for seed in seeds:
            overrides = dict(base, **{"run.seed": seed}, **{k: v for (k, _), v in zip(params, combo)})
            tag = "_".join(f"{k.split('.')[-1]}{v}" for (k, _), v in zip(params, combo))
            run_id = f"{tag or 'base'}_s{seed}"
            jobs.append((args.config, overrides, run_id, out_root / run_id))

    print(f"{len(jobs)} runs on {args.workers} workers -> {out_root}")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    df = pd.DataFrame(rows)
    df.to_parquet(out_root / "summary.parquet", index=False)
    print(df.drop(columns=[c for c in df.columns if c.startswith("first_")]).to_string())


if __name__ == "__main__":
    main()
