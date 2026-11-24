import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seams.analysis.figures import new_figure, save
from seams.analysis.make_figures import line_by
from seams.analysis.rq3 import _bars

KEY = ["condition", "scenario", "strength"]
CONDS = ["protection_only", "static_contracts", "blind", "contracts"]
GROUPS = {"protection_only": ("tab:red", "Protection only"), "static_contracts": ("tab:orange", "Static contracts"),
          "blind": ("tab:olive", "Blind tightening"), "contracts": ("tab:blue", "WARRANT")}
STYLE = {"protection_only": ("tab:red", "o"), "static_contracts": ("tab:orange", "s"),
         "blind": ("tab:olive", "D"), "contracts": ("tab:blue", "^")}
LABEL = {c: name for c, (_, name) in GROUPS.items()}
# strongest setting of each attack origin, and its tick label
STRONGEST = {("fdi", -0.04): "Grid\ntelemetry", ("aggregator", 1.0): "Aggregator", ("dispatch", 1.0): "Dispatcher"}
COSTED = {("nominal", 0.0): "Attack-\nfree", **STRONGEST}
DELTA = {("fdi", -0.04): "Grid\ntelemetry", ("aggregator", 1.0): "Aggregator",
         ("dispatch", 0.75): "Dispatcher\n75 %", ("dispatch", 1.0): "Dispatcher\n100 %"}


def load(results, name="depot"):
    a = pd.read_parquet(results / f"{name}_summary.parquet").query("scheduler == 'EDF'").copy()
    a["strength"] = a.strength.fillna(0.0)
    return a.assign(crossed=a.tau_cross_s.notna().astype(float), depth=a.depth.fillna(1.0))


def table(a):
    return a.groupby(KEY).agg(
        runs=("seed", "size"), crossed=("crossed", "mean"), depth=("depth", "mean"),
        unserved_kWh=("unserved_kWh", "mean"), total_unserved_kWh=("total_unserved_kWh", "mean"),
        tau_cross_min=("tau_cross_s", lambda x: x.median() / 60), tau_blame_min=("tau_blame_s", lambda x: x.median() / 60),
        attribution_ok=("attribution_ok", "mean"), periods_fallback=("periods_fallback", "mean"),
        false_incidents=("false_incidents", "mean"), phi_safe_unsound=("unsound_periods", "sum"),
        out_of_scope_violations=("out_of_scope_violations", "sum")).reset_index()


def _deltas(a):
    p = a[a.condition.isin(["blind", "contracts"])].pivot_table(
        index=["scenario", "strength", "seed"], columns="condition", values="total_unserved_kWh").dropna()
    return (p["blind"] - p["contracts"]).rename("delta").reset_index()


def paired(a):
    from scipy.stats import wilcoxon
    rows = []
    for (s, st), d in _deltas(a).groupby(["scenario", "strength"]):
        nz = d.delta[d.delta.abs() > 1e-6]
        rows.append({"scenario": s, "strength": st, "mean_delta_kWh": d.delta.mean(), "worse": int((nz > 0).sum()),
                     "better": int((nz < 0).sum()), "days": len(d),
                     "p": wilcoxon(nz).pvalue if len(nz) > 1 else np.nan})
    return pd.DataFrame(rows)


def _cell(a, key, cond, col):
    return a[(a.scenario == key[0]) & (a.strength == key[1]) & (a.condition == cond)][col].astype(float)


def figures(a, out, pdf):
    made = []
    # crossings by attack origin, strongest setting of each
    labels = [k for k in STRONGEST if not _cell(a, k, "protection_only", "crossed").empty]
    fig, ax = new_figure(ylabel="Share of days crossing a seam")
    _bars(ax, labels, CONDS, GROUPS, lambda k, c: _cell(a, k, c, "crossed").mean() * np.ones(1))
    w = 0.8 / len(CONDS)
    for j, k in enumerate(labels):  # day counts on every bar, so zero bars stay visible
        for i, c in enumerate(CONDS):
            x = _cell(a, k, c, "crossed")
            ax.text(j + (i - (len(CONDS) - 1) / 2) * w, x.mean() + 0.02, f"{int(x.sum())}/{len(x)}",
                    rotation=90, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(labels)), [STRONGEST[k] for k in labels], fontsize=13)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=12, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.0)
    made += save(fig, out, "depot_crossing", pdf)

    # dispatcher: crossings and cascade depth vs share of the fleet compromised
    d = a[a.scenario.isin(["dispatch", "nominal"])].copy()
    for col, ylabel, name, ylim in (("crossed", "Share of days crossing a seam", "depot_dispatch_crossing", (-0.05, 1.05)),
                                    ("depth", "Cascade depth", "depot_dispatch_depth", (0.9, 2.1))):
        fig, ax = new_figure(xlabel="Share of AV fleet compromised", ylabel=ylabel)
        line_by(ax, d, "strength", col, "condition", CONDS, LABEL, STYLE)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.legend(fontsize=12, loc="upper left")
        made += save(fig, out, name, pdf)

    # what each condition pays: attack-free and the strongest attack of each origin
    groups = {c: GROUPS[c] for c in ("protection_only", "blind", "contracts")}
    fig, ax = new_figure(ylabel="Unserved energy [kWh]")
    _bars(ax, list(COSTED), list(groups), groups, lambda k, c: _cell(a, k, c, "total_unserved_kWh"))
    ax.set_xticks(range(len(COSTED)), list(COSTED.values()), fontsize=12)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax.legend(fontsize=12, loc="upper left", ncol=1, handlelength=1.0)
    made += save(fig, out, "depot_unserved", pdf)

    # paired extra cost of blind tightening over attribution, same day
    dl = _deltas(a)
    keys = [k for k in DELTA if ((dl.scenario == k[0]) & (dl.strength == k[1])).any()]
    style = {"delta": ("tab:olive", "Blind − WARRANT")}
    fig, ax = new_figure(ylabel="Blind − WARRANT [kWh]")
    _bars(ax, keys, ["delta"], style, lambda k, _: dl[(dl.scenario == k[0]) & (dl.strength == k[1])].delta)
    for i, k in enumerate(keys):
        x = dl[(dl.scenario == k[0]) & (dl.strength == k[1])].delta
        ax.scatter(np.full(len(x), i) + np.linspace(-0.15, 0.15, len(x)), x, s=14, color="k", zorder=3)
    ax.set_xticks(range(len(keys)), [DELTA[k] for k in keys], fontsize=12)
    made += save(fig, out, "depot_blind_delta", pdf)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results = Path(args.results)
    if not (results / "depot_summary.parquet").exists():
        print("depot: no depot_summary.parquet; skipped")
        return []
    for name in ("depot", "depot_aux2kW"):
        if not (results / f"{name}_summary.parquet").exists():
            continue
        t, p = table(load(results, name)), paired(load(results, name))
        t.to_csv(results / f"{name}_table.csv", index=False)
        p.to_csv(results / f"{name}_paired.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(f"== {name}")
            print(t.round(2).to_string(index=False))
            print(p.round(4).to_string(index=False))
    made = figures(load(results), results / "figures", args.pdf or None)
    for m in made:
        print("->", m)
    return made


if __name__ == "__main__":
    main()
