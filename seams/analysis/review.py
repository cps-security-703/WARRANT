"""Review experiments (Nov 2026): tables and figures for E1-E3 and E5.

    python -m seams.analysis.review [--results seams/results] [--pdf]

E1 (detect_respond_summary): detect-and-respond baselines against protection only and WARRANT
   (rq1_summary, EDF) on the same FDI days: crossed share, unserved energy, detection time.
E2 (detect_respond_summary, scenario fdi_ramp; ramp 0.004 from rq1 / E1): time to detection and
   to crossing against the FDI ramp rate -> figure fdi_ramp_race.
E3 (disturbances_summary): accidental disturbances, protection only vs WARRANT.
E7 (two_meter_summary): coordinated two-meter FDI against protection only, the E1 baseline and
   WARRANT with the MV ("contracts") and PCC-only ("contracts_anchor") distrust estimators;
   attack-free cost of the anchor estimator against WARRANT's default (rq1, EDF, same days).
E5 (scaling.csv): certification time against the number of hubs -> figure scaling_time.
R1/R2 (peak_shift_summary, fdi_mask_summary, depot_mask_summary): attack classes requested in review, load shifting
   by release time and safety-masking (positive-bias) telemetry, protection only vs WARRANT.
Each part runs only if its input exists. Tables are written as CSV next to the summaries.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from seams.analysis.figures import new_figure, save

COND = {"protection_only": "Prot. only", "ids_response": "IDS response", "ids_always": "Always estimate",
        "contracts": "WARRANT", "contracts_anchor": "WARRANT, anchor"}


def _prep(r):
    r = r.copy()
    r["strength"] = pd.to_numeric(r["strength"], errors="coerce")
    r["crossed"] = r.tau_cross_s.notna().astype(float)
    return r


def _paired(a, b, col="total_unserved_kWh"):
    """Mean b - a over seeds, days b better / worse, sign-test p."""
    m = a.set_index("seed")[col].to_frame("a").join(b.set_index("seed")[col].to_frame("b"), how="inner")
    d = m.b - m.a
    better, worse = int((d < -1e-6).sum()), int((d > 1e-6).sum())
    p = binomtest(better, better + worse).pvalue if better + worse else 1.0
    return float(d.mean()), better, worse, p


def e1(results):
    dr = _prep(pd.read_parquet(results / "detect_respond_summary.parquet"))
    rq1 = _prep(pd.read_parquet(results / "rq1_summary.parquet"))
    rq1 = rq1[(rq1.scheduler == "EDF") & rq1.condition.isin(["protection_only", "contracts"])]
    rq1 = rq1[rq1.scenario.isin(["nominal", "fdi"])]
    both = pd.concat([rq1, dr[dr.scenario.isin(["nominal", "fdi"])]], ignore_index=True)
    rows = []
    for (cond, scen, s), g in both.groupby(["condition", "scenario", "strength"], dropna=False):
        rows.append({"condition": cond, "scenario": scen, "bias": s, "days": len(g), "crossed": g.crossed.mean(),
                     "unserved": g.total_unserved_kWh.mean(),
                     "tau_detect_min": (g.tau_blame_s if cond == "contracts" else g.tau_detect_ids_s).median() / 60,
                     "tau_cross_min": g.tau_cross_s.median() / 60})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_e1.csv", index=False)
    # paired energy: each detect-and-respond baseline and WARRANT against protection only
    pairs = []
    for scen, s in [("nominal", np.nan), *[("fdi", b) for b in (-0.005, -0.01, -0.02, -0.04)]]:
        sel = lambda c: both[(both.condition == c) & (both.scenario == scen) &
                             ((both.strength == s) if s == s else both.strength.isna())]
        for c in ("ids_response", "ids_always", "contracts"):
            mean, better, worse, p = _paired(sel("protection_only"), sel(c))
            pairs.append({"scenario": scen, "bias": s, "condition": c, "d_unserved": mean,
                          "better": better, "worse": worse, "p": p})
    pd.DataFrame(pairs).to_csv(results / "review_e1_paired.csv", index=False)
    return t, pd.DataFrame(pairs)


def e2(results, out, pdf):
    dr = _prep(pd.read_parquet(results / "detect_respond_summary.parquet"))
    rq1 = _prep(pd.read_parquet(results / "rq1_summary.parquet"))
    ramp = dr[dr.scenario == "fdi_ramp"].copy()
    # ramp 0.004 at bias -0.04 = the rq1 default (protection only, WARRANT) and E1 (IDS response)
    base = pd.concat([rq1[(rq1.scheduler == "EDF") & (rq1.scenario == "fdi") & (rq1.strength == -0.04) &
                          rq1.condition.isin(["protection_only", "contracts"])],
                      dr[(dr.condition == "ids_response") & (dr.scenario == "fdi") & (dr.strength == -0.04)]])
    base = base.assign(strength=0.004)
    ramp = pd.concat([ramp, base], ignore_index=True)
    rows = []
    for (cond, rate), g in ramp.groupby(["condition", "strength"]):
        rows.append({"condition": cond, "ramp": rate, "crossed": g.crossed.mean(),
                     "unserved": g.total_unserved_kWh.mean(),
                     "tau_detect_min": (g.tau_blame_s if cond == "contracts" else g.tau_detect_ids_s).median() / 60,
                     "tau_cross_min": g.tau_cross_s.median() / 60,
                     "reach_min": 0.04 / rate if rate > 0 else 0.0})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_e2.csv", index=False)
    # x: minutes the ramp takes to reach the full -0.04 p.u. bias (0 = step)
    by = lambda c: t[t.condition == c].sort_values("reach_min")
    po, w, ir = by("protection_only"), by("contracts"), by("ids_response")
    fig, ax = new_figure(xlabel="Time to full bias [min]", ylabel="Time from onset [min]")
    ax.plot(po.reach_min, po.tau_cross_min, "o-", color="tab:gray", label="Crossing, prot. only")
    ax.plot(w.reach_min, w.tau_detect_min, "s-", color="tab:blue", label="WARRANT blame")
    ax.plot(ir.reach_min, ir.tau_detect_min, "^--", color="tab:orange", label="BDD alarm")
    ticks = sorted(po.reach_min)
    ax.set_xticks(ticks, ["step" if v == 0 else f"{v:g}" for v in ticks])
    ax.legend(fontsize=11, loc="upper left")
    made = save(fig, out, "fdi_ramp_race", pdf)
    return t, made


def e3(results):
    r = _prep(pd.read_parquet(results / "disturbances_summary.parquet"))
    rows = []
    for (scen, s), g in r.groupby(["scenario", "strength"], dropna=False):
        po, w = g[g.condition == "protection_only"], g[g.condition == "contracts"]
        mean, better, worse, p = _paired(po, w)
        rows.append({"scenario": scen, "strength": s, "days": len(w),
                     "blamed_W": (w.first_blame.fillna("") != "").mean(),
                     "blame_grid_W": (w.first_blame == "grid").mean(),
                     "tau_blame_min": w.tau_blame_s.median() / 60,
                     "fallback_W": w.periods_fallback.mean(),
                     "crossed_PO": po.crossed.mean(), "crossed_W": w.crossed.mean(),
                     "phi_viol_PO": (po.phi_safe_violation_steps.fillna(0) > 0).mean() if "phi_safe_violation_steps" in po
                     else np.nan,
                     "fail_grid_PO": po.fail_grid.mean() if "fail_grid" in po else np.nan,
                     "unserved_PO": po.total_unserved_kWh.mean(), "unserved_W": w.total_unserved_kWh.mean(),
                     "d_unserved": mean, "better": better, "worse": worse, "p": p,
                     "unsound_W": w.unsound_periods.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_e3.csv", index=False)
    return t


def e7(results):
    r = _prep(pd.read_parquet(results / "two_meter_summary.parquet"))
    rq1 = _prep(pd.read_parquet(results / "rq1_summary.parquet"))
    rq1 = rq1[(rq1.scheduler == "EDF") & (rq1.condition == "contracts")]
    rows = []
    for (cond, scen, s), g in r.groupby(["condition", "scenario", "strength"], dropna=False):
        ci = binomtest(int(g.crossed.sum()), len(g)).proportion_ci(method="exact")
        rows.append({"condition": cond, "scenario": scen, "bias": s, "days": len(g), "crossed": g.crossed.mean(),
                     "crossed_lo": ci.low, "crossed_hi": ci.high,
                     "blame_grid": (g.first_blame == "grid").mean() if "first_blame" in g else np.nan,
                     "tau_blame_min": g.tau_blame_s.median() / 60, "tau_cross_min": g.tau_cross_s.median() / 60,
                     "unserved": g.total_unserved_kWh.mean(), "fallback": g.periods_fallback.mean(),
                     "unsound": g.unsound_periods.sum(), "phi_viol": g.phi_safe_violation_steps.fillna(0).sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_e7.csv", index=False)
    # the anchor estimator's cost against WARRANT's default on the same days (nominal, single-meter FDI)
    pairs = []
    for scen, s in (("nominal", np.nan), ("fdi", -0.04)):
        sel = lambda d: d[(d.scenario == scen) & ((d.strength == s) if s == s else d.strength.isna())]
        mean, better, worse, p = _paired(sel(rq1), sel(r[r.condition == "contracts_anchor"]))
        pairs.append({"scenario": scen, "bias": s, "d_unserved": mean, "better": better, "worse": worse, "p": p})
    return t, pd.DataFrame(pairs)


def e7_screen(results):
    """anchor_screen: the PCC-only estimator on both feeders at every load, against the MV estimator
    (WARRANT's default) on the same days from electrification / feeder123."""
    a = _prep(pd.read_parquet(results / "anchor_screen_summary.parquet"))
    a["feeder"] = a.condition.str.replace("anchor_", "")
    m = pd.concat([_prep(pd.read_parquet(results / f"{c}_summary.parquet")).assign(feeder=f)
                   for c, f in (("electrification", "ieee34"), ("feeder123", "ieee123"))])
    m = m[m.condition == "contracts"]
    rows = []
    for (f, el, scen), g in a.groupby(["feeder", "elec", "scenario"]):
        ref = m[(m.feeder == f) & (m.elec == el) & (m.scenario == ("fdi" if scen == "fdi2" else scen))]
        mean, better, worse, p = _paired(ref, g)
        rows.append({"feeder": f, "elec": el, "scenario": scen, "days": len(g), "crossed_anchor": int(g.crossed.sum()),
                     "crossed_mv_fdi": int(ref.crossed.sum()), "blame_grid": (g.first_blame == "grid").mean(),
                     "unserved_anchor": g.total_unserved_kWh.mean(), "d_unserved_vs_mv": mean,
                     "better": better, "worse": worse, "unsound": g.unsound_periods.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_e7_screen.csv", index=False)
    return t


def both(results):
    """Experiment A (both_eval): the combined distrust estimator against the anchor screen (same
    cells) and the MV estimator (electrification / feeder123; two_meter for IEEE 34 2035 fdi2)."""
    def load(name, prefix=None):
        r = _prep(pd.read_parquet(results / f"{name}_summary.parquet"))
        r["strength"] = r.strength.fillna(0.0)
        return r.assign(feeder=r.condition.str.replace(prefix, "")) if prefix else r
    b, a = load("both_eval", "both_"), load("anchor_screen", "anchor_")
    m = pd.concat([load(c).assign(feeder=f) for c, f in (("electrification", "ieee34"), ("feeder123", "ieee123"))])
    m = m[m.condition == "contracts"]
    tm = load("two_meter")
    tm = tm[tm.condition == "contracts"].assign(feeder="ieee34", elec="2035")
    m = pd.concat([m, tm[tm.scenario == "fdi2"]])
    rows = []
    for (f, el, scen, st), g in b.groupby(["feeder", "elec", "scenario", "strength"]):
        sel = lambda r: r[(r.feeder == f) & (r.elec == el) & (r.scenario == scen) & (r.strength == st)]
        ga, gm = sel(a), sel(m)
        mean, better, worse, p = _paired(gm, g) if len(gm) else (np.nan, 0, 0, np.nan)
        rows.append({"feeder": f, "elec": el, "scenario": scen, "strength": st, "days": len(g),
                     "crossed_both": int(g.crossed.sum()), "crossed_anchor": int(ga.crossed.sum()) if len(ga) else np.nan,
                     "crossed_mv": int(gm.crossed.sum()) if len(gm) else np.nan,
                     "blame_grid": int((g.first_blame == "grid").sum()), "unserved_both": g.total_unserved_kWh.mean(),
                     "d_unserved_vs_mv": mean, "better": better, "worse": worse, "unsound": g.unsound_periods.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_both.csv", index=False)
    return t


def cost_decomp(results):
    """Experiment D: attack-free cost with V2's margins dropped in turn (tab:costdecomp)."""
    c = pd.read_parquet(results / "cost_decomp_summary.parquet")
    rows = []
    for (el, cond), g in c.groupby(["elec", "condition"]):
        full = c[(c.elec == el) & (c.condition == "full")]
        mean, better, worse, p = _paired(full, g)
        rows.append({"elec": el, "condition": cond, "unserved": g.total_unserved_kWh.mean(),
                     "fallback_periods": g.periods_fallback.mean(), "d_vs_full": mean, "better": better,
                     "worse": worse, "p": p, "unsound": g.unsound_periods.sum(),
                     "violation_steps": g.phi_safe_violation_steps.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_cost_decomp.csv", index=False)
    return t


def horizon30(results):
    """30 min vs 1 h V2 horizon (IEEE 34, EDF): horizon30 against the contracts runs of
    electrification (1 h) and, for FDI -0.02 at 2035, of rq1; protection only for reference."""
    def load(name):
        r = _prep(pd.read_parquet(results / f"{name}_summary.parquet"))
        r["strength"] = r.strength.fillna(0.0)
        return r
    h = load("horizon30")
    e = load("electrification")
    rq1 = load("rq1")
    rq1 = rq1[(rq1.scheduler == "EDF") & (rq1.scenario == "fdi") & (rq1.strength == -0.02)].assign(elec="2035")
    ref = pd.concat([e, rq1])
    rows = []
    for (el, scen, st), g in h.groupby(["elec", "scenario", "strength"]):
        sel = lambda c: ref[(ref.condition == c) & (ref.elec == el) & (ref.scenario == scen) & (ref.strength == st)]
        w, po = sel("contracts"), sel("protection_only")
        mean, better, worse, p = _paired(w, g)
        rows.append({"elec": el, "scenario": scen, "strength": st, "days": len(g),
                     "crossed_1h": int(w.crossed.sum()), "crossed_30": int(g.crossed.sum()),
                     "crossed_po": int(po.crossed.sum()) if len(po) else np.nan,
                     "E_po": po.total_unserved_kWh.mean() if len(po) else np.nan, "E_1h": w.total_unserved_kWh.mean(),
                     "E_30": g.total_unserved_kWh.mean(), "d_30_vs_1h": mean, "better": better, "worse": worse, "p": p,
                     "fb_1h": w.periods_fallback.mean(), "fb_30": g.periods_fallback.mean(),
                     "blame_30": int((g.first_blame == g.origin).sum()) if scen != "nominal" else 0,
                     "unsound_30": g.unsound_periods.sum(), "viol_30": g.out_of_scope_violations.sum(),
                     "viol_steps_30": g.phi_safe_violation_steps.sum()})
    t = pd.DataFrame(rows)
    t.to_csv(results / "review_horizon30.csv", index=False)
    return t


def attack_classes(results):
    """R1 (peak_shift) and R2 (fdi_mask): per load, attack and condition: crossings, grid
    failures (Phi_safe), first blame, unserved energy over the attack-free twin; WARRANT - PO paired."""
    rows = []
    for name in ("peak_shift", "fdi_mask", "depot_mask"):
        f = results / f"{name}_summary.parquet"
        if not f.exists():
            continue
        r = _prep(pd.read_parquet(f))
        r["strength"] = r.strength.fillna(0.0)
        if "elec" not in r:  # depot_mask: IEEE 123 at 2035 only
            r["elec"] = "2035"
        for (el, scen, st), g in r.groupby(["elec", "scenario", "strength"]):
            po, w = (g[g.condition == c] for c in ("protection_only", "contracts"))
            nom = {c: r[(r.elec == el) & (r.scenario == "nominal") & (r.condition == c)] for c in ("protection_only", "contracts")}
            d_w = _paired(po, w)
            row = {"campaign": name, "elec": el, "scenario": scen, "strength": st, "days": len(w),
                   "crossed_po": int(po.crossed.sum()), "crossed_w": int(w.crossed.sum()),
                   "grid_fail_po": int((po.fail_grid == True).sum()), "grid_fail_w": int((w.fail_grid == True).sum()),
                   "viol_steps_po": po.phi_safe_violation_steps.sum(), "viol_steps_w": w.phi_safe_violation_steps.sum(),
                   "tau_cross_po_min": po.tau_cross_s.median() / 60,
                   "blamed_w": int((w.first_blame == w.origin).sum()) if scen != "nominal" else 0,
                   "blame_any_w": int((w.first_blame.fillna("") != "").sum()),
                   "tau_blame_w_min": w.tau_blame_s.median() / 60,
                   "E_po": po.total_unserved_kWh.mean(), "E_w": w.total_unserved_kWh.mean(),
                   "dE_po_vs_nom": _paired(nom["protection_only"], po)[0],
                   "dE_w_vs_nom": _paired(nom["contracts"], w)[0],
                   "w_minus_po": d_w[0], "w_better": d_w[1], "w_worse": d_w[2], "p": d_w[3],
                   "fallback_w": w.periods_fallback.mean(), "unsound_w": w.unsound_periods.sum(),
                   "oos_w": w.out_of_scope_violations.sum()}
            rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(results / "review_attack_classes.csv", index=False)
    return t


def e5(results, out, pdf):
    s = pd.read_csv(results / "scaling.csv")
    g = s.groupby("hubs")
    med = g[["opt_ms", "v1_ms", "v2_ms", "exit_ms", "total_ms"]].median()
    fig, ax = new_figure(xlabel="Hubs", ylabel="Time per renegotiation [s]")
    for col, name, st in (("total_ms", "Total", "o-"), ("opt_ms", "Optimizer", "s--"), ("v2_ms", "V2", "^--")):
        ax.plot(med.index, med[col] / 1000, st, label=name)
    ax.axhline(2.0, color="k", ls=":", lw=1.2, label="Solver budget")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=11, loc="upper left")
    made = save(fig, out, "scaling_time", pdf)
    med.to_csv(results / "review_e5.csv")
    return med, made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results = Path(args.results)
    out, pdf = results / "figures", args.pdf or None
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        if (results / "detect_respond_summary.parquet").exists():
            t, pairs = e1(results)
            print("E1\n", t.round(2).to_string(index=False), "\n", pairs.round(4).to_string(index=False))
            t, _ = e2(results, out, pdf)
            print("E2\n", t.round(2).to_string(index=False))
        if (results / "disturbances_summary.parquet").exists():
            print("E3\n", e3(results).round(3).to_string(index=False))
        if (results / "two_meter_summary.parquet").exists():
            t, pairs = e7(results)
            print("E7\n", t.round(3).to_string(index=False), "\n", pairs.round(4).to_string(index=False))
        if (results / "anchor_screen_summary.parquet").exists():
            print("E7 screen\n", e7_screen(results).round(3).to_string(index=False))
        if (results / "both_eval_summary.parquet").exists():
            print("A (both)\n", both(results).round(2).to_string(index=False))
        if (results / "cost_decomp_summary.parquet").exists():
            print("D (cost)\n", cost_decomp(results).round(3).to_string(index=False))
        if (results / "horizon30_summary.parquet").exists():
            print("Horizon 30 min\n", horizon30(results).round(2).to_string(index=False))
        if (results / "peak_shift_summary.parquet").exists() or (results / "fdi_mask_summary.parquet").exists():
            print("R1/R2 attack classes\n", attack_classes(results).round(2).to_string(index=False))
        if (results / "scaling.csv").exists():
            med, _ = e5(results, out, pdf)
            print("E5\n", med.round(0).to_string())


if __name__ == "__main__":
    main()
