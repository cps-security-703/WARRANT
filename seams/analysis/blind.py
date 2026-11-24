import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seams.analysis.figures import new_figure, save
from seams.analysis.make_figures import ci
from seams.analysis.rq3 import _bars

KEY = ["condition", "scenario", "strength"]
GROUPS = {"protection_only": ("tab:gray", "Protection only"), "contracts": ("tab:blue", "WARRANT"),
          "blind": ("tab:orange", "Blind tightening")}
DELTA_STYLE = {"aggregator": ("tab:purple", "s", "Aggregator"), "fdi": ("tab:red", "o", "Grid telemetry"),
               "navigation": ("tab:green", "^", "Navigation")}
SCENARIO_LABEL = {("nominal", 0.0): "Nominal", ("navigation", 1.0): "Navigation", ("aggregator", 1.0): "Aggregator",
                  ("fdi", -0.02): "Grid telemetry −0.02", ("fdi", -0.04): "Grid telemetry −0.04"}


def combined(results):
    b = pd.read_parquet(results / "blind_summary.parquet")
    r = pd.read_parquet(results / "rq2_summary.parquet").query("scheduler == 'EDF'")
    r = r[(r.condition == "protection_only") | ((r.condition == "contracts") & (r.c == 1.0))]
    a = pd.concat([b, r], ignore_index=True)
    a["strength"] = a.strength.fillna(0.0)
    return a.assign(crossed=a.tau_cross_s.notna().astype(float))


def table(a):
    return a.groupby(KEY).agg(
        runs=("seed", "size"), crossed=("crossed", "mean"), total_unserved_kWh=("total_unserved_kWh", "mean"),
        periods_renegotiated=("periods_renegotiated", "mean"), periods_fallback=("periods_fallback", "mean"),
        false_incidents=("false_incidents", "mean"), phi_safe_unsound=("unsound_periods", "sum"),
        out_of_scope_violations=("out_of_scope_violations", "sum")).reset_index()


def paired(a):
    p = a[a.condition.isin(["blind", "contracts"])].pivot_table(
        index=["scenario", "strength", "seed"], columns="condition", values="total_unserved_kWh").dropna()
    d = (p["blind"] - p["contracts"]).rename("delta")
    return d.groupby(level=["scenario", "strength"]).agg(
        mean_delta_kWh="mean", worse=lambda x: int((x > 1e-6).sum()), better=lambda x: int((x < -1e-6).sum()),
        days="size").reset_index()


def _deltas(a):
    p = a[a.condition.isin(["blind", "contracts"])].pivot_table(
        index=["scenario", "strength", "seed"], columns="condition", values="total_unserved_kWh").dropna()
    return (p["blind"] - p["contracts"]).rename("delta").reset_index()


def figures(a, out, pdf):
    labels = [s for s in SCENARIO_LABEL if s in set(zip(a.scenario, a.strength))]
    fig, ax = new_figure(ylabel="Unserved energy [kWh]")
    _bars(ax, labels, list(GROUPS), GROUPS, lambda s, g: a[(a.scenario == s[0]) & (a.strength == s[1]) &
                                                           (a.condition == g)]["total_unserved_kWh"])
    ax.set_xticks(range(len(labels)), [SCENARIO_LABEL[s] for s in labels], rotation=30)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax.legend(fontsize=12, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.0)
    made = save(fig, out, "blind_unserved", pdf)

    # Paired cost of blind tightening over attribution, by attack strength (weakest to strongest).
    d = _deltas(a)
    fig, ax = new_figure(xlabel="Attack strength level", ylabel="Blind − WARRANT [kWh]")
    for scen, (color, marker, name) in DELTA_STYLE.items():
        ds = d[d.scenario == scen]
        levels = sorted(ds.strength.unique(), key=abs)
        stats = np.array([ci(ds[ds.strength == lv].delta) for lv in levels])
        x = np.arange(1, len(levels) + 1)
        ax.errorbar(x, stats[:, 0], yerr=stats[:, 1:].T, color=color, marker=marker, capsize=4, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=12, loc="upper left")
    made += save(fig, out, "blind_delta", pdf)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results = Path(args.results)
    a = combined(results)
    t, p = table(a), paired(a)
    t.to_csv(results / "blind_table.csv", index=False)
    p.to_csv(results / "blind_paired.csv", index=False)
    made = figures(a, results / "figures", args.pdf or None)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(2).to_string(index=False))
        print(p.round(2).to_string(index=False))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
