import argparse
import time

import yaml

from seams.cosim.config import load_config, resolve_path
from seams.cosim.master import Master


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--attack", help="YAML file with an `attacks:` list (replaces the config's)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, value parsed as YAML (e.g. evse.scheduler=LLF)")
    args = ap.parse_args(argv)

    overrides = {}
    if args.attack:
        overrides["attacks"] = yaml.safe_load(resolve_path(args.attack).read_text())["attacks"]
    overrides.update({k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)})
    if args.seed is not None:
        overrides["run.seed"] = args.seed
    if args.gui:
        overrides["sumo.gui"] = True
    cfg = load_config(args.config, overrides)
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-s{cfg['run']['seed']}"

    t0 = time.perf_counter()
    run_dir = Master(cfg, run_id).run()
    print(f"run {run_id} done in {time.perf_counter() - t0:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
