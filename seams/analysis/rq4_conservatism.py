import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from seams.analysis.figures import new_figure, save
from seams.analysis.rq3 import _bars

KEY = ["c", "scenario", "strength", "seed"]
SCENARIO = {"nominal": ("tab:gray", "Nominal"), "navigation": ("tab:green", "Navigation"),
            "aggregator": ("tab:blue", "Aggregator"), "fdi": ("tab:red", "Grid telemetry")}
ALL = {"all": ("navy", "All")}


def combined(results):
    s = pd.read_parquet(results / "rq4_conservatism_summary.parquet")
    r = pd.read_parquet(results / "rq4_summary.parquet").query("condition in ['classical', 'agents_floor']")
    r = r[r.set_index(["scenario", "strength"]).index.isin(s.set_index(["scenario", "strength"]).index)]
    return pd.concat([s, r.assign(c=1.0)], ignore_index=True)


def paired(s):
    cl = s[s.condition == "classical"].set_index(KEY)
    fl = s[s.condition == "agents_floor"].set_index(KEY)
    d = pd.DataFrame({"d_unserved_kWh": fl.unserved_kWh - cl.unserved_kWh,
                      "d_home_unserved_kWh": fl.home_unserved_kWh - cl.home_unserved_kWh,
                      "d_fallback": fl.periods_fallback - cl.periods_fallback,
                      "periods_agreed": fl.periods_agreed, "certified_share": fl.certified_share}).dropna(subset=["d_unserved_kWh"])
    return d.reset_index()


def table(s, d):
    rows = []
    for c, g in d.groupby("c"):
        x = g.d_unserved_kWh.to_numpy()
        nz = x[np.abs(x) > 1e-9]
        gate = s[(s.c == c) & (s.condition == "agents_floor")]
        rows.append(dict(c=c, runs=len(x), d_unserved_kWh=x.mean(), better=int((x < -1e-9).sum()),
                         worse=int((x > 1e-9).sum()), p_wilcoxon=wilcoxon(nz).pvalue if len(nz) > 1 else np.nan,
                         d_home_unserved_kWh=g.d_home_unserved_kWh.mean(), d_fallback=g.d_fallback.mean(),
                         periods_agreed=g.periods_agreed.mean(), certified_share=g.certified_share.min(),
                         unsound=int(gate.unsound_periods.sum()), oos=int(gate.out_of_scope_violations.sum()),
                         phi_violations=gate.phi_safe_violation_steps.sum()))
    return pd.DataFrame(rows)


def figures(d, out, pdf):
    cs = sorted(d.c.unique())
    made = []
    for col, ylabel, name, style, key in (
            ("d_unserved_kWh", "Δ hub unserved energy [kWh]", "rq4_conservatism_unserved", ALL, None),
            ("d_unserved_kWh", "Δ hub unserved energy [kWh]", "rq4_conservatism_unserved_scenario", SCENARIO, "scenario"),
            ("periods_agreed", "Agreed periods per day", "rq4_conservatism_agreed_periods", ALL, None)):
        groups = [g for g in style if key is None or g in set(d[key])]
        fig, ax = new_figure(xlabel="Conservatism c", ylabel=ylabel)
        _bars(ax, cs, groups, style, lambda c, g: d[(d.c == c) & ((d[key] == g) if key else True)][col].astype(float))
        ax.set_xticks(range(len(cs)), [f"{c:g}" for c in cs])
        if key:
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
    s = combined(results)
    d = paired(s)
    t = table(s, d)
    t.to_csv(results / "rq4_conservatism_table.csv", index=False)
    made = figures(d, results / "figures", args.pdf or None)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(4).to_string(index=False))
        print(d.groupby(["c", "scenario"]).d_unserved_kWh.mean().round(3).unstack().to_string())
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
