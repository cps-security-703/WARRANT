"""Regenerate every figure from the campaign logs with one command (the Phase 6 gate).

    python -m seams.analysis.make_figures [--results seams/results] [--pdf]

Inputs: <results>/{rq1,rq2,delta,electrification,feeder123}_summary.parquet (one row per run)
and, for trajectories, the
kept fine logs of seed 2000 under the rq1 campaign's out_dir. Every figure is its own file
(analysis/figures.py style); means with bootstrap 95 % CIs over seeds. Missing inputs are skipped
with a note, so partial campaigns still render what they can.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seams.analysis.figures import new_figure, save
from seams.cosim.config import resolve_path

SCENARIO_LABEL = {"navigation": "Navigation spoofing", "fdi": "Grid telemetry falsification", "aggregator": "Aggregator release"}
STRENGTH_LABEL = {"navigation": "Fraction of fleet spoofed", "fdi": "Falsification bias [pu]", "aggregator": "Share of chargers released"}
CONDITION_LABEL = {"protection_only": "Protection only", "static_contracts": "Static contracts",
                   "contracts": "WARRANT", "oracle": "Centralized oracle"}
CONDITION_STYLE = {"protection_only": ("tab:red", "o"), "static_contracts": ("tab:orange", "s"),
                   "contracts": ("tab:blue", "^"), "oracle": ("tab:green", "D")}
COSTS = {"total_unserved_kWh": "Unserved energy [kWh]", "trip_delay_veh_s": "Trip delay per vehicle [s]",
         "refused_vehicles": "Refused vehicles"}
SCENARIOS = ("navigation", "fdi", "aggregator")


def ci(values, n_boot=2000, seed=0):
    """Mean and bootstrap 95 % CI half-widths (lo, hi) of a 1-D sample."""
    x = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    m = x.mean()
    return m, m - np.quantile(boots, 0.025), np.quantile(boots, 0.975) - m


def contained(df):
    """A run is contained when nothing fails outside its origin domain(s)."""
    def ok(r):
        fail = set(filter(None, str(r.get("failing") or "").split(",")))
        orig = set(filter(None, str(r.get("origin") or "").split(",")))
        return fail <= orig
    return df.apply(ok, axis=1)


def prepare(df):
    df = df.copy()
    df["contained"] = contained(df)
    df["trip_delay_veh_s"] = df.trip_delay_s / df.trip_delay_n.replace(0, np.nan)
    return df


def line_by(ax, sub, x, y, by, order, label_map, style_map=None):
    # nested marker sizes and line styles keep coincident series visible
    for i, g in enumerate(order):
        d = sub[sub[by] == g]
        if d.empty:
            continue
        xs, ms, los, his = [], [], [], []
        for xv, grp in d.groupby(x):
            m, lo, hi = ci(grp[y])
            xs.append(xv), ms.append(m), los.append(lo), his.append(hi)
        color, marker = (style_map or {}).get(g, (None, "o"))
        ax.errorbar(xs, ms, yerr=[los, his], marker=marker, color=color, capsize=3, label=label_map.get(g, g),
                    markersize=max(4, 10 - 2.5 * i), linestyle=("-", "--", ":")[i % 3])


def legend(ax):
    ax.legend(fontsize=11)


# --- RQ1 ---------------------------------------------------------------------------
def rq1(df, out):
    made = []
    edf = df[df.scheduler == "EDF"]
    conds = ["protection_only", "static_contracts", "contracts"]
    for s in SCENARIOS:
        sub = edf[edf.scenario == s]
        if sub.empty:
            continue
        fig, ax = new_figure(SCENARIO_LABEL[s], STRENGTH_LABEL[s], "Cascade depth")
        line_by(ax, sub, "strength", "depth", "condition", conds, CONDITION_LABEL, CONDITION_STYLE)
        ax.set_ylim(0.8, 3.2)
        legend(ax)
        made += save(fig, out, f"rq1_cascade_depth_{s}")

        smax = sub.strength.loc[sub.strength.abs().idxmax()]
        strongest = df[(df.scenario == s) & (df.strength == smax)]
        fig, ax = new_figure(SCENARIO_LABEL[s], "Scheduler", "Contained runs [%]")
        scheds = ["EDF", "LLF", "RR"]
        width = 0.25
        for i, c in enumerate(conds):
            vals = [100 * strongest[(strongest.condition == c) & (strongest.scheduler == sc)].contained.mean() for sc in scheds]
            ax.bar(np.arange(3) + (i - 1) * width, vals, width, color=CONDITION_STYLE[c][0], label=CONDITION_LABEL[c])
        ax.set_xticks(np.arange(3), scheds)
        ax.set_ylim(0, 105)
        legend(ax)
        made += save(fig, out, f"rq1_containment_by_scheduler_{s}")

    # detection time at the strongest strength: IDS (protection only) vs contract monitors
    fig, ax = new_figure("Time to detect (strongest attack)", "Scenario", "Time to detect [min]")
    width = 0.25
    series = [("IDS", "protection_only", "tau_detect_ids_s", "tab:gray"),
              ("Static contracts", "static_contracts", "tau_detect_s", "tab:orange"),
              ("WARRANT", "contracts", "tau_detect_s", "tab:blue")]
    for i, (lab, cond, col, color) in enumerate(series):
        ms, los, his, missed = [], [], [], []
        for s in SCENARIOS:
            sub = edf[(edf.scenario == s) & (edf.condition == cond)]
            sub = sub[sub.strength.abs() == sub.strength.abs().max()] if not sub.empty else sub
            vals = sub[col] / 60.0 if col in sub else pd.Series(dtype=float)
            m, lo, hi = ci(vals)
            ms.append(m), los.append(lo), his.append(hi)
            missed.append(vals.isna().mean() if len(vals) else np.nan)
        x = np.arange(len(SCENARIOS)) + (i - 1) * width
        ax.bar(x, np.nan_to_num(ms), width, yerr=[np.nan_to_num(los), np.nan_to_num(his)], capsize=3,
               color=color, label=lab)
        for xi, mi, miss in zip(x, ms, missed):
            if not np.isnan(miss) and miss > 0:
                ax.text(xi, (0 if np.isnan(mi) else mi) + 0.5, f"{100 * miss:.0f}%\nmissed",
                        ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(SCENARIOS)), ["Navigation", "Grid\ntelemetry", "Aggregator"])
    legend(ax)
    made += save(fig, out, "rq1_detection_time")

    sub = edf[edf.condition == "contracts"]
    if not sub.empty:
        fig, ax = new_figure("Attribution (contracts + reneg.)", "Attack strength (level)", "Correct first blame [%]")
        for s in SCENARIOS:
            d = sub[sub.scenario == s]
            levels = sorted(d.strength.unique(), key=abs)
            ys = [100 * d[d.strength == lv].attribution_ok.fillna(False).astype(float).mean() for lv in levels]
            ax.plot(range(1, len(levels) + 1), ys, marker="o", label=SCENARIO_LABEL[s])
        ax.set_ylim(-5, 105)
        legend(ax)
        made += save(fig, out, "rq1_attribution_accuracy")

    nom = df[df.scenario == "nominal"]
    if not nom.empty and "false_incidents" in nom:
        fig, ax = new_figure("False attribution (nominal days)", "Scheduler", "False incidents per day")
        for i, c in enumerate(["static_contracts", "contracts"]):
            ms, los, his = zip(*[ci(nom[(nom.condition == c) & (nom.scheduler == sc)].false_incidents)
                                 for sc in ["EDF", "LLF", "RR"]])
            ax.bar(np.arange(3) + (i - 0.5) * 0.35, ms, 0.35, yerr=[los, his], capsize=3,
                   color=CONDITION_STYLE[c][0], label=CONDITION_LABEL[c])
        ax.set_xticks(np.arange(3), ["EDF", "LLF", "RR"])
        legend(ax)
        made += save(fig, out, "rq1_false_attribution")

    made += containment_window(df, out)
    return made


def containment_window(df, out):
    """FDI: tau_cross (protection-only twin) - tau_detect (contracts), per seed and strength."""
    base = df[(df.condition == "protection_only") & (df.scenario == "fdi")][["scheduler", "strength", "seed", "tau_cross_s"]]
    con = df[(df.condition == "contracts") & (df.scenario == "fdi")]
    j = con.merge(base, on=["scheduler", "strength", "seed"], suffixes=("", "_base"))
    j = j[j.tau_cross_s_base.notna() & j.tau_detect_s.notna()]
    if j.empty:
        return []
    j["margin_min"] = (j.tau_cross_s_base - j.tau_detect_s - j.tau_reneg_s.fillna(0)) / 60.0
    fig, ax = new_figure("Containment window (grid telemetry)", STRENGTH_LABEL["fdi"], "Window margin [min]")
    line_by(ax, j[j.scheduler == "EDF"], "strength", "margin_min", "condition", ["contracts"], CONDITION_LABEL, CONDITION_STYLE)
    ax.axhline(0, color="k", linewidth=1)
    legend(ax)
    return save(fig, out, "rq1_containment_window_fdi")


# --- RQ2 ---------------------------------------------------------------------------
def rq2(df, out):
    made = []
    for s in SCENARIOS:
        sub = df[(df.scenario == s) & (df.strength.abs() >= df[df.scenario == s].strength.abs().median())]
        if sub.empty:
            continue
        for cost, label in COSTS.items():
            fig, ax = new_figure(SCENARIO_LABEL[s], label, "Contained runs [%]")
            c = sub[sub.condition == "contracts"]
            pts = []
            for cv, grp in c.groupby("c"):
                pts.append((ci(grp[cost])[0], 100 * grp.contained.mean(), cv))
            pts.sort(key=lambda p: p[2])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-^", color=CONDITION_STYLE["contracts"][0],
                        label=CONDITION_LABEL["contracts"])
                for j, (x, y, cv) in enumerate(pts):  # alternate above/below so close points stay legible
                    ax.annotate(f"c={cv}", (x, y), textcoords="offset points",
                                xytext=(4, -14 if j % 2 == 0 else 6), fontsize=10)
            for ref in ("protection_only", "oracle"):
                r = sub[sub.condition == ref]
                if not r.empty:
                    color, marker = CONDITION_STYLE[ref]
                    ax.plot([ci(r[cost])[0]], [100 * r.contained.mean()], marker, color=color, markersize=9,
                            linestyle="none", label=CONDITION_LABEL[ref])
            ax.set_ylim(-5, 115)  # room for the c labels above the 100 % line
            legend(ax)
            made += save(fig, out, f"rq2_frontier_{s}_{cost}")
    nom = df[df.scenario == "nominal"]
    for cost, label in COSTS.items():
        if cost == "trip_delay_veh_s" or nom.empty:
            continue  # nominal runs are their own trip-delay reference
        fig, ax = new_figure("Cost of contracts on nominal days", "Conservatism c", label)
        c = nom[nom.condition == "contracts"]
        xs = sorted(c.c.unique())
        ms, los, his = zip(*[ci(c[c.c == x][cost]) for x in xs]) if xs else ([], [], [])
        ax.errorbar(xs, ms, yerr=[los, his], marker="^", capsize=3, color=CONDITION_STYLE["contracts"][0],
                    label=CONDITION_LABEL["contracts"])
        for ref in ("protection_only", "oracle"):
            r = nom[nom.condition == ref]
            if not r.empty:
                ax.axhline(ci(r[cost])[0], color=CONDITION_STYLE[ref][0], linestyle="--" if ref == "protection_only" else ":", linewidth=2.5 if ref == "protection_only" else 1.5, label=CONDITION_LABEL[ref])
        legend(ax)
        made += save(fig, out, f"rq2_nominal_cost_{cost}")
    return made


# --- Delta ---------------------------------------------------------------------------
def delta(df, rq1_df, out):
    made = []
    d = df.copy()
    d["delta_min"] = d.delta.astype(float) / 60.0
    fig, ax = new_figure("Contract period", "Contract period [min]", "Contained runs [%]")
    for s in SCENARIOS:
        sub = d[d.scenario == s]
        xs = sorted(sub.delta_min.unique())
        ax.plot(xs, [100 * sub[sub.delta_min == x].contained.mean() for x in xs], marker="o", label=SCENARIO_LABEL[s])
    ax.set_ylim(-5, 105)
    legend(ax)
    made += save(fig, out, "delta_containment")
    if rq1_df is not None:
        base = rq1_df[(rq1_df.condition == "protection_only") & (rq1_df.scheduler == "EDF") & (rq1_df.scenario == "fdi")]
        base = base[base.strength == -0.04][["seed", "tau_cross_s"]]
        j = d[d.scenario == "fdi"].merge(base, on="seed", suffixes=("", "_base"))
        j["margin_min"] = (j.tau_cross_s_base - j.tau_detect_s - j.tau_reneg_s.fillna(0)) / 60.0
        fig, ax = new_figure("Containment window (grid telemetry)", "Contract period [min]", "Window margin [min]")
        xs = sorted(j.delta_min.unique())
        ms, los, his = zip(*[ci(j[j.delta_min == x].margin_min) for x in xs]) if xs else ([], [], [])
        ax.errorbar(xs, ms, yerr=[los, his], marker="^", capsize=3, color=CONDITION_STYLE["contracts"][0])
        ax.axhline(0, color="k", linewidth=1)
        made += save(fig, out, "delta_window_margin_fdi")
    nom = d[d.scenario == "nominal"]
    fig, ax = new_figure("Nominal-day cost", "Contract period [min]", COSTS["total_unserved_kWh"])
    xs = sorted(nom.delta_min.unique())
    ms, los, his = zip(*[ci(nom[nom.delta_min == x].total_unserved_kWh) for x in xs]) if xs else ([], [], [])
    ax.errorbar(xs, ms, yerr=[los, his], marker="^", capsize=3, color=CONDITION_STYLE["contracts"][0])
    made += save(fig, out, "delta_nominal_unserved")
    return made


# --- electrification axis (one campaign per feeder) ---------------------------------------
ELEC_ORDER = ["today", "2035", "2045"]
ELEC_TICK = {"today": "Today", "2035": "2035", "2045": "2045"}


def electrification(df, out, prefix):
    """Per scenario: cascade depth and contained share vs electrification level, protection
    only vs contracts; nominal-day cost; feeder-only (out-of-scope) violations under contracts."""
    made = []
    d = df.copy()
    d["elec"] = d.elec.astype(str)
    levels = [e for e in ELEC_ORDER if e in set(d.elec)]
    xs = np.arange(len(levels))
    conds = [c for c in ("protection_only", "contracts") if c in set(d.condition)]

    def by_level(ax, sub, y, scale=1.0):
        for i, c in enumerate(conds):
            cs = sub[sub.condition == c]
            stats = [ci(scale * cs[cs.elec == e][y].astype(float)) for e in levels]
            ms, los, his = zip(*stats)
            color, marker = CONDITION_STYLE[c]
            ax.errorbar(xs + (i - 0.5) * 0.08, ms, yerr=[los, his], marker=marker, color=color, capsize=3,
                        label=CONDITION_LABEL[c], markersize=max(4, 9 - 3 * i), linestyle=("-", "--")[i % 2])
        ax.set_xticks(xs, [ELEC_TICK[e] for e in levels])

    for s in SCENARIOS:
        sub = d[d.scenario == s]
        if sub.empty:
            continue
        fig, ax = new_figure(SCENARIO_LABEL[s], "Electrification level", "Cascade depth")
        by_level(ax, sub, "depth")
        ax.set_ylim(0.8, 3.2)
        legend(ax)
        made += save(fig, out, f"{prefix}_cascade_depth_{s}")
        fig, ax = new_figure(SCENARIO_LABEL[s], "Electrification level", "Contained runs [%]")
        by_level(ax, sub.assign(c100=sub.contained.astype(float)), "c100", 100.0)
        ax.set_ylim(-5, 105)
        legend(ax)
        made += save(fig, out, f"{prefix}_containment_{s}")
    nom = d[d.scenario == "nominal"]
    if not nom.empty:
        fig, ax = new_figure("Nominal-day cost", "Electrification level", COSTS["total_unserved_kWh"])
        by_level(ax, nom, "total_unserved_kWh")
        legend(ax)
        made += save(fig, out, f"{prefix}_nominal_unserved")
        nom = nom.assign(grid_fail=nom.fail_grid.astype(float))
        fig, ax = new_figure("Nominal days with a grid failure", "Electrification level", "Days [%]")
        by_level(ax, nom, "grid_fail", 100.0)
        ax.set_ylim(-5, 105)
        legend(ax)
        made += save(fig, out, f"{prefix}_nominal_grid_failure")
    return made


# --- trajectories (seed 2000, kept fine logs) -------------------------------------------
def trajectories(run_root, out, seed=2000):
    made = []
    plots = {
        "fdi": ("x-0.04", "H1 voltage [pu]", [("V_true", "True", "-"), ("V_reported", "Reported", "--"),
                                              ("V_derating", "Derating input", ":")]),
        "aggregator": ("x1.0", "H1 power [kW]", [("P_req", "Requested", "--"), ("P_pcc", "At PCC", "-")]),
        "navigation": ("x1.0", "H1 queue [vehicles]", [("queue", "Queue", "-")]),
    }
    for s, (tag, ylabel, cols) in plots.items():
        for cond in ("protection_only", "static_contracts", "contracts"):
            d = Path(run_root) / f"{cond}_EDF_{s}_{tag}_s{seed}"
            if not (d / "fine.parquet").exists():
                continue
            f = pd.read_parquet(d / "fine.parquet")
            f = f[(f.hub == "H1") & (f.t_fine >= 960) & (f.t_fine < 1170)]
            fig, ax = new_figure(f"{SCENARIO_LABEL[s]}: {CONDITION_LABEL[cond]}", "Time of day [h]", ylabel)
            for col, lab, ls in cols:
                if col in f:
                    ax.plot(f.t_fine / 60.0, f[col], ls, label=lab)
            ax.axvspan(990 / 60.0, 1110 / 60.0, color="tab:red", alpha=0.08)
            if s == "fdi":
                ax.axhline(0.95, color="k", linewidth=1)
            legend(ax)
            made += save(fig, out, f"traj_{s}_{cond}")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--out", default=None, help="default: <results>/figures")
    ap.add_argument("--pdf", action="store_true", help="also write PDFs (end of pipeline)")
    args = ap.parse_args(argv)
    results = resolve_path(args.results)
    out = Path(args.out) if args.out else results / "figures"
    if args.pdf:
        import os
        os.environ["FIG_PDF"] = "1"
    made = []
    load = lambda n: prepare(pd.read_parquet(results / f"{n}_summary.parquet")) if (results / f"{n}_summary.parquet").exists() else None
    r1, r2, dl = load("rq1"), load("rq2"), load("delta")
    el, f123 = load("electrification"), load("feeder123")
    for name, df, fn in (("rq1", r1, lambda d: rq1(d, out)), ("rq2", r2, lambda d: rq2(d, out)),
                         ("delta", dl, lambda d: delta(d, r1, out)),
                         ("electrification", el, lambda d: electrification(d, out, "elec34")),
                         ("feeder123", f123, lambda d: electrification(d, out, "elec123"))):
        if df is None:
            print(f"skip {name}: no summary yet")
            continue
        made += fn(df)
    spec = yaml.safe_load(resolve_path("seams/experiments/campaigns/rq1.yaml").read_text())
    made += trajectories(Path(spec["out_dir"]).expanduser() / spec["name"], out)
    # RQ3 (trained attackers), RQ4 and RQ5 (agent campaigns) have their own modules; run them here so one
    # command regenerates every figure
    from seams.analysis import rq3 as rq3_mod, rq4 as rq4_mod, rq5 as rq5_mod
    pdf = ["--pdf"] if args.pdf else []
    for name, mod, runs, need in (("rq3", rq3_mod, "~/seams_runs/rq3", None),
                                  ("rq4", rq4_mod, "~/seams_runs/rq4", results / "rq4_summary.parquet"),
                                  ("rq5", rq5_mod, "~/seams_runs/rq5", results / "rq5_summary.parquet")):
        if not Path(runs).expanduser().exists() or (need is not None and not need.exists()):
            print(f"skip {name}: no runs yet")
            continue
        mod.main(["--runs", runs, "--results", str(results)] + pdf)
    from seams.analysis import blind as blind_mod, dwell as dwell_mod, rq4_conservatism as cons_mod
    for mod, need in ((dwell_mod, "dwell"), (blind_mod, "blind"), (cons_mod, "rq4_conservatism")):
        if (results / f"{need}_summary.parquet").exists():
            mod.main(["--results", str(results)] + pdf)
    from seams.analysis import charging as charging_mod  # RQ1 charging side + scheduler table
    if (results / "rq1_summary.parquet").exists():
        charging_mod.main(["--results", str(results)] + pdf)
    from seams.analysis import testbed as testbed_mod  # appendix testbed/operating-point figures
    if Path("~/seams_runs/electrification").expanduser().exists():
        testbed_mod.main(["--results", str(results)] + pdf)
    from seams.analysis import review as review_mod  # E1-E3, E5: detect-and-respond, ramps, disturbances, scaling
    review_mod.main(["--results", str(results)] + pdf)
    from seams.analysis import depot as depot_mod  # robotaxi depot: cascades from every origin, blind vs WARRANT
    depot_mod.main(["--results", str(results)] + pdf)
    print(f"{len(made)} files -> {out} (plus rq3-rq5, dwell, blind, conservatism, charging, testbed, review, depot counts above)")
    return made


if __name__ == "__main__":
    main()
