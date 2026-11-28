"""Experiment campaigns (Phase 6+): one YAML spec -> every run, resumable, one metrics row each.

    python -m seams.experiments.campaign seams/experiments/campaigns/rq1.yaml [--workers 10] [--dry-run]

Spec:
  name, out_dir (run data; outside Dropbox), results_dir (summary parquet; in the project)
  seeds:       evaluation seeds
  conditions:  {name: {config, set: {dotted: value}, scenarios: [names] (optional: only these)}}
  axes:        [{name, key, values}]   cartesian, applied to every condition except those listed
               in the axis' `skip`
  schedulers:  [EDF, LLF, RR]           (evse.scheduler)
  scenarios:   {name: {attack: file | null, param, values, set: {dotted: value}}}   null attack = nominal day
  keep_fine:   seeds whose fine.parquet is kept (example trajectories); others drop it

Each run writes metrics.json (cascade, attribution, renegotiation, certification, availability,
transport costs). A pairing pass then adds trip delay against the nominal run of the same
condition, axes, scheduler and seed, and writes <results_dir>/<name>_summary.parquet.
"""
import argparse
import itertools
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seams.cosim.config import load_config, resolve_path


def jobs_from_spec(spec):
    out_root = Path(spec["out_dir"]).expanduser() / spec["name"]
    axes = spec.get("axes", [])
    for cname, cond in spec["conditions"].items():
        cond_axes = [a for a in axes if cname not in a.get("skip", [])]
        for combo in itertools.product(*[a["values"] for a in cond_axes]):
            axis_vals = {a["name"]: v for a, v in zip(cond_axes, combo)}
            for sched in spec.get("schedulers", ["EDF"]):
                for sname, scen in spec["scenarios"].items():
                    if "scenarios" in cond and sname not in cond["scenarios"]:
                        continue
                    for strength in scen.get("values", [None]):
                        for seed in spec["seeds"]:
                            over = dict(cond.get("set", {}))
                            over.update({a["key"]: v for a, v in zip(cond_axes, combo)})
                            over["evse.scheduler"] = sched
                            over["run.seed"] = seed
                            attacks = []
                            if scen.get("attack"):
                                attacks = yaml.safe_load(resolve_path(scen["attack"]).read_text())["attacks"]
                                if strength is not None:  # attacks.<i>.<key>, or attacks.*.<key> for every spec
                                    i, key = scen["param"].split(".", 2)[1:]
                                    for j in (range(len(attacks)) if i == "*" else [int(i)]):
                                        attacks[j] = dict(attacks[j], **{key: strength})
                            over["attacks"] = attacks
                            over.update(scen.get("set", {}))  # scenario-level config (e.g. a demand surge)
                            tag = "_".join([cname, *(f"{k}{v}" for k, v in axis_vals.items()), sched, sname,
                                            *([f"x{strength}"] if strength is not None else []), f"s{seed}"])
                            yield {
                                "run_id": tag, "run_dir": str(out_root / tag), "config": cond["config"],
                                "overrides": over, "keep_fine": seed in spec.get("keep_fine", []),
                                "key": {"condition": cname, **axis_vals, "scheduler": sched, "scenario": sname,
                                        "strength": strength, "seed": seed},
                            }


def run_metrics(run_dir):
    from seams.analysis import metrics as M
    out = {}
    for fn in (M.cascade, M.availability_cost, M.attribution, M.renegotiation, M.certification):
        try:
            out.update(fn(run_dir))
        except FileNotFoundError:
            pass
    trips = pd.read_parquet(Path(run_dir) / "trips.parquet")
    out.update(
        refused_vehicles=int(trips.refusals.sum()), stranded_vehicles=int(trips.stranded.sum()),
        vehicles_arrived=int(trips.arrive_s.notna().sum()),
    )
    return out


class RunFailed(RuntimeError):
    """A run that raised; picklable (some library exceptions are not, and break the pool)."""


def _run(job):
    run_dir = Path(job["run_dir"])
    done = run_dir / "metrics.json"
    if done.exists():
        return json.loads(done.read_text())
    try:
        return _execute(job, run_dir, done)
    except Exception as e:
        shutil.rmtree(run_dir, ignore_errors=True)  # no partial logs for a later pass to append to
        raise RunFailed(f"{job['run_id']}: {type(e).__name__}: {e}") from None


def _execute(job, run_dir, done):
    from seams.cosim.master import Master
    cfg = load_config(job["config"], job["overrides"])
    t0 = time.perf_counter()
    Master(cfg, job["run_id"], run_dir=run_dir).run()
    row = {"run_id": job["run_id"], **job["key"], "wall_s": time.perf_counter() - t0, **run_metrics(run_dir)}
    row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
    if not job["keep_fine"]:
        (run_dir / "fine.parquet").unlink(missing_ok=True)
    for sub in (run_dir / "sumo").glob("*"):  # generated SUMO inputs are reproducible from config
        sub.unlink()
    done.write_text(json.dumps(row, default=str))
    return row


def trip_delay(df, out_root):
    """Per attacked run: extra trip time of vehicles that arrived in both it and its nominal twin."""
    keys = [c for c in df.columns if c not in ("scenario", "strength") and c in df.attrs.get("key_cols", [])]
    nominal = df[df.scenario == "nominal"].set_index(keys)
    delays = []
    for _, r in df.iterrows():
        if r.scenario == "nominal":
            delays.append((0.0, 0))
            continue
        try:
            twin = nominal.loc[tuple(r[k] for k in keys)]
        except KeyError:
            delays.append((np.nan, 0))
            continue
        a = pd.read_parquet(out_root / r.run_id / "trips.parquet").set_index("vehicle_id")
        b = pd.read_parquet(out_root / twin.run_id / "trips.parquet").set_index("vehicle_id")
        both = a.merge(b, left_index=True, right_index=True, suffixes=("_a", "_n")).dropna(
            subset=["arrive_s_a", "arrive_s_n"])
        extra = (both.arrive_s_a - both.depart_s_a) - (both.arrive_s_n - both.depart_s_n)
        delays.append((float(extra.sum()), int(len(both))))
    df["trip_delay_s"] = [d for d, _ in delays]
    df["trip_delay_n"] = [n for _, n in delays]
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-condition", default=None, metavar="A,B",
                    help="run only these conditions (the summary then covers only them; a later full pass reuses every finished run)")
    args = ap.parse_args(argv)
    spec = yaml.safe_load(resolve_path(args.spec).read_text())
    if args.only_condition:
        keep = args.only_condition.split(",")
        spec["conditions"] = {c: v for c, v in spec["conditions"].items() if c in keep}
    jobs = list(jobs_from_spec(spec))
    out_root = Path(spec["out_dir"]).expanduser() / spec["name"]
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"{spec['name']}: {len(jobs)} runs, {len(todo)} to do, {args.workers} workers -> {out_root}")
    if args.dry_run:
        return
    rows, failed, t0 = [], [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except RunFailed as e:
                failed.append(str(e))
                print(f"  FAILED {e}", flush=True)
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    if failed:
        # a summary must cover every run of the spec; re-run the campaign (finished runs are reused)
        raise SystemExit(f"{len(failed)} runs failed, no summary written; first: {failed[0]}")
    df = pd.DataFrame(rows)
    df.attrs["key_cols"] = [c for c in jobs[0]["key"] if c not in ("scenario", "strength")]
    df = trip_delay(df, out_root)
    results = resolve_path(spec["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results / f"{spec['name']}_summary.parquet", index=False)
    print(f"-> {results / (spec['name'] + '_summary.parquet')}")


if __name__ == "__main__":
    main()
