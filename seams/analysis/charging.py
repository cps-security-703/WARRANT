import argparse
from pathlib import Path

import pandas as pd
import yaml

from seams.analysis.figures import new_figure, save

KW_PER_A = 0.208  # aggregate hub current limit [A] at 208 V -> kW (504.8 A = 105 kW)
WINDOW = (960, 1170)  # fine steps (min): 16:00-19:30, as the voltage trajectories
ATTACK = (990, 1110)


def trajectory(runs, out, pdf, seed=2000):
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="H1 power [kW]")
    for cond, color, name in (("protection_only", "tab:gray", "Prot. only"), ("contracts", "tab:blue", "WARRANT")):
        d = runs / "rq1" / f"{cond}_EDF_fdi_x-0.04_s{seed}"
        f = pd.read_parquet(d / "fine.parquet")
        f = f[(f.hub == "H1") & (f.t_fine >= WINDOW[0]) & (f.t_fine < WINDOW[1])]
        t = f.t_fine / 60.0
        ax.plot(t, f.d_h, color=color, ls="--", lw=1.2, label=f"{name}: derating")
        ax.plot(t, f.P_pcc, color=color, lw=1.6, label=f"{name}: power")
        if cond == "contracts":
            delta = yaml.safe_load(open(d / "config.yaml"))["clock"]["contract_period_s"]
            c = pd.read_parquet(d / "contract.parquet")
            c = c[c.hub == "H1"]
            tc = (c.k + 1) * delta / 60.0
            c = c[(tc >= WINDOW[0]) & (tc < WINDOW[1])]
            ax.step((c.k + 1) * delta / 3600.0, c.Pbar_H1, where="post", color="k", ls=":", lw=1.4,
                    label="WARRANT: cap")
    ax.axvspan(ATTACK[0] / 60.0, ATTACK[1] / 60.0, color="tab:red", alpha=0.08)
    ax.set_ylim(0, 180)
    ax.legend(fontsize=12, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.4, handletextpad=0.4,
              framealpha=0.95, borderaxespad=0.2)
    return save(fig, out, "traj_fdi_charging", pdf)


def scheduler_table(results):
    r = pd.read_parquet(results / "rq1_summary.parquet")
    r = r.assign(strength=r.strength.fillna(0.0), crossed=r.tau_cross_s.notna().astype(float))
    fdi = r[(r.scenario == "fdi") & (r.strength == -0.04)]
    rows = []
    for s, g in r.groupby("scheduler"):
        f, n = fdi[fdi.scheduler == s], g[g.scenario == "nominal"]
        row = {"scheduler": s}
        for cond, tag in (("protection_only", "PO"), ("static_contracts", "S"), ("contracts", "W")):
            row[f"crossed_{tag}"] = f[f.condition == cond].crossed.mean()
        full = g[(g.condition == "contracts") & (g.strength.abs().isin([1.0, 0.04]))]  # full-strength attacks
        row["attribution_W"] = full.attribution_ok.mean()
        for cond, tag in (("protection_only", "PO"), ("contracts", "W")):
            row[f"nominal_{tag}"] = n[n.condition == cond].total_unserved_kWh.mean()
            row[f"fdi_{tag}"] = f[f.condition == cond].total_unserved_kWh.mean()
        row["unsound_W"] = g[g.condition == "contracts"].unsound_periods.sum()
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(results / "rq1_scheduler.csv", index=False)
    return t


def morning_table(results):
    r = pd.read_parquet(results / "fdi_morning_summary.parquet")
    r = r.assign(strength=r.strength.fillna(0.0), crossed=r.tau_cross_s.notna().astype(float))
    base = r[r.strength == 0].set_index(["feeder", "condition", "seed"]).total_unserved_kWh
    rows = []
    for (feeder, s, cond), g in r[r.strength != 0].groupby(["feeder", "strength", "condition"]):
        extra = g.set_index(["feeder", "condition", "seed"]).total_unserved_kWh - base
        rows.append({"feeder": feeder, "bias": s, "condition": cond, "crossed": g.crossed.mean(),
                     "attribution": g.attribution_ok.dropna().astype(float).mean() if cond == "contracts" else None,
                     "tau_blame_s": g.tau_blame_s.mean(), "unserved": g.total_unserved_kWh.mean(),
                     "over_nominal": extra.dropna().mean(), "unsound": g.unsound_periods.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "fdi_morning_table.csv", index=False)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--runs", default="~/seams_runs")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results, runs = Path(args.results), Path(args.runs).expanduser()
    made = trajectory(runs, results / "figures", args.pdf or None)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(scheduler_table(results).round(2).to_string(index=False))
    if (results / "fdi_morning_summary.parquet").exists():
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(morning_table(results).round(2).to_string(index=False))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
