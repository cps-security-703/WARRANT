"""RQ5 (Phase 9): the Byzantine negotiator against honest negotiation.

    python -m seams.analysis.rq5 [--runs ~/seams_runs/rq5] [--results seams/results] [--pdf]

Input: the rq5 campaign (seams/experiments/campaigns/rq5.yaml): <results>/rq5_summary.parquet and
per run contract.parquet, trips.parquet, negotiations.jsonl. Every Byzantine run has an honest
twin (condition "honest", same scenario and seed). Output: <results>/rq5_table.csv and
<results>/figures/rq5_*.png.

Per Byzantine run, against its honest twin:
  loss per honest operator   charging: hub unserved energy [kWh]; fleet: extra trip time [veh h]
                             plus refused and stranded vehicles; utility: lost Phi_safe voltage
                             margin [mpu]; residents: HOME unserved energy [kWh]. The compromised
                             operator's own row is reported but is not a loss to an honest operator
  no agreement               share of triggered negotiations that end without agreement (the
                             classical contract or theta_safe is installed)
  theta deviation            mean over periods and parties of |P_bar - P_bar of the honest twin| [kW]
  gate                       Phi_safe violation steps (in scope), unsound periods, out-of-scope
                             violations: all must be 0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seams.analysis.figures import new_figure, save
from seams.analysis.rq3 import _bars, trip_delay_veh_h

STRATEGIES = ("cost_shift", "stall", "setup", "misreport", "stall_limit")
STRATEGY_LABEL = {"cost_shift": "Shift", "stall": "Stall", "setup": "Setup", "misreport": "Misrep.",
                  "stall_limit": "Stall+lim."}
DOMAINS = {"fleet": ("tab:green", "Transport"), "charging": ("tab:blue", "Charging"),
           "utility": ("tab:red", "Grid")}
ATTACK_OF = {"fleet": "navigation", "charging": "aggregator", "utility": "fdi"}
KEY = ["scenario", "strength", "seed"]
PARTIES = ("H1", "H2", "H3", "HOME")


def split(condition):
    """'stall_limit_fleet' -> ('stall_limit', 'fleet')."""
    strategy, _, role = condition.rpartition("_")
    return strategy, role


def no_agreement_share(run_dir):
    f = Path(run_dir) / "negotiations.jsonl"
    if not f.exists():
        return np.nan
    recs = [json.loads(line) for line in f.read_text().splitlines()]
    return float(np.mean([not r.get("agreement") for r in recs])) if recs else np.nan


def theta_deviation(run_dir, twin_dir):
    cols = ["k", "hub"] + [f"Pbar_{p}" for p in PARTIES]
    a, b = (pd.read_parquet(Path(d) / "contract.parquet", columns=cols).query("hub == 'H1'") for d in (run_dir, twin_dir))
    m = a.merge(b, on="k", suffixes=("_a", "_b"))
    return float(np.mean([np.abs(m[f"Pbar_{p}_a"] - m[f"Pbar_{p}_b"]).mean() for p in PARTIES]))


def compare(summary, runs):
    honest = summary[summary.condition == "honest"].set_index(KEY)
    rows = []
    for _, r in summary[summary.condition != "honest"].iterrows():
        key = (r.scenario, r.strength, r.seed)
        if key not in honest.index:
            continue
        h = honest.loc[key]
        strategy, role = split(r.condition)
        d, hd = Path(runs) / r.run_id, Path(runs) / h.run_id
        rows.append(dict(
            condition=r.condition, strategy=strategy, role=role, scenario=r.scenario, seed=r.seed,
            attack=r.scenario != "nominal",
            d_charging_kWh=r.unserved_kWh - h.unserved_kWh,
            d_fleet_veh_h=trip_delay_veh_h(d, hd),
            d_fleet_vehicles=(r.refused_vehicles + r.stranded_vehicles) - (h.refused_vehicles + h.stranded_vehicles),
            d_utility_margin_mpu=1000.0 * (h.phi_safe_margin_v_pu - r.phi_safe_margin_v_pu),
            d_residents_kWh=r.home_unserved_kWh - h.home_unserved_kWh,
            d_total_kWh=r.total_unserved_kWh - h.total_unserved_kWh,
            no_agreement=no_agreement_share(d), no_agreement_honest=no_agreement_share(hd),
            d_fallback_periods=r.periods_fallback - h.periods_fallback,
            theta_dev_kW=theta_deviation(d, hd),
            phi_violations=r.phi_safe_violation_steps, unsound=r.unsound_periods, oos=r.out_of_scope_violations))
    return pd.DataFrame(rows)


LOSSES = (("d_charging_kWh", "Δ hub unserved energy [kWh]", "charging"),
          ("d_fleet_veh_h", "Δ trip delay [veh·h]", "fleet"),
          ("d_utility_margin_mpu", "Lost voltage margin [mpu]", "utility"),
          ("d_residents_kWh", "Δ home unserved energy [kWh]", "residents"))


def figures(cmp, out, pdf):
    made = []
    strategies = [s for s in STRATEGIES if s in set(cmp.strategy)]
    roles = [r for r in DOMAINS if r in set(cmp.role)]
    style = {r: DOMAINS[r] for r in roles}

    def bars(col, ylabel, name, sub):
        fig, ax = new_figure(ylabel=ylabel)
        _bars(ax, strategies, roles, style, lambda s, r: sub[(sub.strategy == s) & (sub.role == r)][col].astype(float))
        ax.set_xticks(range(len(strategies)), [STRATEGY_LABEL[s] for s in strategies], rotation=30)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))
        ax.legend(fontsize=12, ncol=3, loc="upper center", columnspacing=0.6, handlelength=1.0)
        return save(fig, out, name, pdf)

    for attack, tag in ((True, "attack"), (False, "nominal")):
        sub = cmp[cmp.attack == attack]
        for col, ylabel, op in LOSSES:
            if cmp[col].abs().max() < 1e-9:  # e.g. contracts never change trips: no figure, the table shows 0
                print(f"skip rq5_loss_{op}_{tag}: identical to the honest twin in every run")
                continue
            made += bars(col, ylabel, f"rq5_loss_{op}_{tag}", sub)
        made += bars("no_agreement", "No-agreement share", f"rq5_no_agreement_{tag}", sub)
        made += bars("theta_dev_kW", "Mean |P̄ − honest P̄| [kW]", f"rq5_theta_deviation_{tag}", sub)
    made += bars("d_total_kWh", "Δ total unserved energy [kWh]", "rq5_setup_combined",
                 cmp[cmp.attack & cmp.strategy.isin(["setup", "cost_shift", "stall", "misreport"])])
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/seams_runs/rq5")
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    runs, results = Path(args.runs).expanduser(), Path(args.results)
    summary = pd.read_parquet(results / "rq5_summary.parquet")
    gate = summary[["phi_safe_violation_steps", "unsound_periods", "out_of_scope_violations"]].sum()
    print("gate (all runs, must be 0):", gate.to_dict())
    cmp = compare(summary, runs)
    t = cmp.groupby(["strategy", "role", "attack"]).agg(
        runs=("seed", "size"), **{c: (c, "mean") for c in (
            "d_charging_kWh", "d_fleet_veh_h", "d_fleet_vehicles", "d_utility_margin_mpu", "d_residents_kWh",
            "d_total_kWh", "no_agreement", "no_agreement_honest", "d_fallback_periods", "theta_dev_kW")},
        phi_violations=("phi_violations", "sum"), unsound=("unsound", "sum"), oos=("oos", "sum")).reset_index()
    t.to_csv(results / "rq5_table.csv", index=False)
    made = figures(cmp, results / "figures", args.pdf or None)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(3).to_string(index=False))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
