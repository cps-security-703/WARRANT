import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from seams.cosim.config import PROJECT_ROOT

STAGES = ("tests", "data", "calibrate", "ids", "campaigns", "rq3", "agents", "figures")
DEFAULT = tuple(s for s in STAGES if s != "tests")
RUNS = Path("~/seams_runs").expanduser()

# (feeder, electrification, Delta [s]) of every contract file a campaign loads
CALIBRATIONS = [("ieee34", e, 120) for e in ("today", "2035", "2045")] + \
               [("ieee34", "2035", d) for d in (60, 300)] + \
               [("ieee123", e, 120) for e in ("today", "2035", "2045")] + \
               [("ieee123", "2035", 120, "dcfc_depot")]  # fast-charging hubs + robotaxi depot at H2

CAMPAIGNS = ("rq1", "rq2", "delta", "electrification", "feeder123", "dwell", "blind", "fdi_morning",
             "detect_respond", "disturbances", "two_meter", "anchor_screen", "both_dev", "both_eval", "cost_decomp", "horizon30", "depot", "depot_aux2kW")
AGENT_CAMPAIGNS = ("rq4", "rq4_conservatism", "rq5")

# RQ3 attackers: (train_attackers arguments, output name). Order matters: --resume-from needs its source
_E45, _NOEPS = ["--set", "scenario.electrification=2045"], ["--set", "verifier.use_epsilon=false"]
_STRESS = _E45 + ["--rl", "reward.detection_penalty=0", "--rl", "reward.terminate_on_detection=false",
                  "--rl", "surfaces.fdi.bias_max_pu=0.2"]
_R5 = ["--rl", "training.rounds=5"]
ATTACKERS = [
    *[(["--surface", s, "--box", b, "--mode", "within"], f"within_{s}_{b}")
      for b in ("grey", "white") for s in ("fdi", "aggregator", "navigation")],
    *[(["--surface", s, "--mode", "inter"], f"inter_{s}_grey") for s in ("aggregator", "fdi", "navigation")],
    *[(["--surface", s, "--mode", "inter", *_NOEPS, "--tag", "noeps"], f"inter_{s}_grey_noeps")
      for s in ("aggregator", "fdi", "navigation")],
    *[(["--surface", s, "--mode", "inter", *_E45, "--tag", "2045"], f"inter_{s}_grey_2045") for s in ("aggregator", "fdi")],
    *[(["--surface", s, "--mode", "inter", *_E45, *_NOEPS, "--tag", "2045_noeps"], f"inter_{s}_grey_2045_noeps")
      for s in ("aggregator", "fdi")],
    # navigation at 2045: 5 rounds, as its final 2035 attacker (inter_navigation_grey_r5)
    (["--surface", "navigation", "--mode", "inter", *_E45, *_R5, "--tag", "2045"], "inter_navigation_grey_2045"),
    (["--surface", "navigation", "--mode", "inter", *_E45, *_NOEPS, *_R5, "--tag", "2045_noeps"],
     "inter_navigation_grey_2045_noeps"),
    (["--surface", "fdi", "--mode", "inter", *_STRESS, "--tag", "2045_stress"], "inter_fdi_grey_2045_stress"),
    (["--surface", "fdi", "--mode", "inter", *_STRESS, *_NOEPS, "--tag", "2045_stress_noeps"],
     "inter_fdi_grey_2045_stress_noeps"),
    (["--surface", "aggregator", "--mode", "within", *_R5, "--resume-from", "within_aggregator_grey", "--tag", "r5"],
     "within_aggregator_grey_r5"),
    (["--surface", "navigation", "--mode", "within", *_R5, "--resume-from", "within_navigation_grey", "--tag", "r5"],
     "within_navigation_grey_r5"),
    (["--surface", "navigation", "--box", "white", "--mode", "within", *_R5, "--resume-from",
      "within_navigation_white", "--tag", "r5"], "within_navigation_white_r5"),
    (["--surface", "navigation", "--mode", "inter", *_R5, "--resume-from", "inter_navigation_grey", "--tag", "r5"],
     "inter_navigation_grey_r5"),
    (["--surface", "navigation", "--mode", "inter", *_NOEPS, *_R5, "--resume-from", "inter_navigation_grey_noeps",
      "--tag", "noeps_r5"], "inter_navigation_grey_noeps_r5"),
]


class Runner:
    def __init__(self, dry_run):
        self.dry_run = dry_run

    def __call__(self, *args):
        cmd = [sys.executable, "-m", *map(str, args)]
        print("  $", " ".join(cmd[2:]), flush=True)
        if self.dry_run:
            return
        t = time.time()
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"  done in {time.time() - t:.0f} s", flush=True)


def skip(path, force):
    if Path(path).exists() and not force:
        print(f"  skip: {Path(path).name} exists")
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all", help=f"comma list of {','.join(STAGES)}; 'all' = all but tests")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="rebuild data, contracts, IDS and attackers that exist")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)
    stages = DEFAULT if args.stages == "all" else tuple(args.stages.split(","))
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; known: {STAGES}")
    run, w = Runner(args.dry_run), ["--workers", args.workers]
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    t0 = time.time()
    for stage in (s for s in STAGES if s in stages):
        print(f"=== {stage} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        if stage == "tests":
            run("pytest", "-q", "-p", "no:cacheprovider", "seams/tests")
        elif stage == "data":
            if not skip(PROJECT_ROOT / "seams/data/processed/summary.json", args.force):
                run("seams.data.prepare")
        elif stage == "calibrate":
            for feeder, elec, delta, *kind in CALIBRATIONS:
                kind = kind[0] if kind else "l2"
                token = feeder if kind == "l2" else f"{feeder}-{kind}"
                theta = PROJECT_ROOT / f"seams/config/contracts_static_{token}_{elec}_{delta}s.yaml"
                if not skip(theta, args.force):
                    run("seams.experiments.calibrate", "seams/config/phase5.yaml", *w,
                        "--set", f"scenario.feeder={feeder}", "--set", f"scenario.electrification={elec}",
                        "--set", f"clock.contract_period_s={delta}", "--set", f"scenario.hub_kind={kind}")
        elif stage == "ids":
            if not skip(PROJECT_ROOT / "seams/models/ids_transformer_testbed.pkl", args.force):
                run("seams.experiments.train_ids", "seams/config/phase2.yaml", *w)
        elif stage in ("campaigns", "agents"):
            if stage == "agents" and not (PROJECT_ROOT / ".env").exists() and not os.environ.get("OPENROUTER_API_KEY"):
                print("  warning: no OPENROUTER_API_KEY; only calls in the replay cache will succeed")
            for name in CAMPAIGNS if stage == "campaigns" else AGENT_CAMPAIGNS:
                run("seams.experiments.campaign", f"seams/experiments/campaigns/{name}.yaml", *w)
            if stage == "campaigns" and not skip(PROJECT_ROOT / "seams/results/scaling.csv", args.force):
                run("seams.experiments.scaling")  # E5: certification time vs hubs (solver only)
        elif stage == "rq3":
            for cli, name in ATTACKERS:
                if not skip(RUNS / "rq3" / name / "eval.parquet", args.force):
                    run("seams.experiments.train_attackers", "seams/config/phase7.yaml", *cli, *w)
        elif stage == "figures":
            run("seams.analysis.make_figures", *([] if args.no_pdf else ["--pdf"]))
    print(f"=== pipeline done in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
