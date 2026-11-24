import argparse
from pathlib import Path

import pandas as pd

from seams.analysis.figures import new_figure, save
from seams.analysis.rq3 import _bars

KEY = ["condition", "scenario", "strength"]
GROUPS = {("protection_only", 1.0): ("tab:gray", "Prot., 1×"), ("protection_only", 0.5): ("silver", "Prot., 0.5×"),
          ("contracts", 1.0): ("tab:blue", "WARRANT, 1×"), ("contracts", 0.5): ("lightskyblue", "WARRANT, 0.5×")}
SCENARIO_LABEL = {("navigation", 1.0): "Navigation", ("aggregator", 1.0): "Aggregator",
                  ("fdi", -0.02): "Grid telemetry −0.02", ("fdi", -0.04): "Grid telemetry −0.04"}


def combined(results):
    d = pd.read_parquet(results / "dwell_summary.parquet").assign(dwell=0.5)
    r = pd.read_parquet(results / "rq2_summary.parquet").query("scheduler == 'EDF'")
    r = r[(r.condition == "protection_only") | ((r.condition == "contracts") & (r.c == 1.0))].assign(dwell=1.0)
    r = r[r.set_index(KEY).index.isin(d.set_index(KEY).index)]
    a = pd.concat([d, r], ignore_index=True)
    return a.assign(crossed=a.tau_cross_s.notna().astype(float))


def table(a):
    return a.groupby(KEY + ["dwell"]).agg(
        runs=("seed", "size"), crossed=("crossed", "mean"), tau_cross_median_s=("tau_cross_s", "median"),
        total_unserved_kWh=("total_unserved_kWh", "mean"), hub_unserved_kWh=("unserved_kWh", "mean"),
        periods_fallback=("periods_fallback", "mean"), phi_safe_violation_steps=("phi_safe_violation_steps", "sum"),
        out_of_scope_violations=("out_of_scope_violations", "sum"), unsound_periods=("unsound_periods", "sum")).reset_index()


def figures(a, out, pdf):
    a = a[a.scenario != "nominal"]
    labels = [s for s in SCENARIO_LABEL if s in set(zip(a.scenario, a.strength))]
    groups = list(GROUPS)
    made = []
    for col, ylabel, name in (("crossed", "Share of days crossing a seam", "dwell_crossing"),
                              ("total_unserved_kWh", "Unserved energy [kWh]", "dwell_unserved")):
        fig, ax = new_figure(ylabel=ylabel)
        _bars(ax, labels, groups, GROUPS, lambda s, g: a[(a.scenario == s[0]) & (a.strength == s[1]) &
                                                         (a.condition == g[0]) & (a.dwell == g[1])][col].astype(float))
        ax.set_xticks(range(len(labels)), [SCENARIO_LABEL[s] for s in labels], rotation=30)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
        ax.legend(fontsize=12, ncol=2, loc="upper center", columnspacing=0.6, handlelength=1.0)
        made += save(fig, out, name, pdf)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results = Path(args.results)
    a = combined(results)
    t = table(a)
    t.to_csv(results / "dwell_table.csv", index=False)
    made = figures(a, results / "figures", args.pdf or None)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(2).to_string(index=False))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
