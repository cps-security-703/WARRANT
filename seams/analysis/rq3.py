"""RQ3 (Phase 7): summary table and figures for the trained adaptive attackers.

    python -m seams.analysis.rq3 [--runs ~/seams_runs/rq3] [--results seams/results] [--pdf]

Input: one directory per attacker from seams.experiments.train_attackers (eval.parquet, the
eval/attack_s*/nominal_s* run logs, the sac/dqn monitor CSVs, q_r*.json). Output:
<results>/rq3_summary.csv (one row per attacker and held-out seed) and <results>/figures/rq3_*.png.
Attackers not trained yet are skipped.

Per seed, beyond the campaign metrics already in eval.parquet:
  trip_delay_veh_h   sum over vehicles that arrived in both runs of the extra travel time
                     (attack minus nominal twin, matched by vehicle id)
  gap_periods        abstraction gap: contract periods whose coarse view was compliant (every
                     finite robustness >= 0, no blame) while a fine step inside the period violated
                     Phi_safe on the true state or a hub drew more than its contracted P_bar
  headroom_used      inter-sample: share of the voltage headroom the attack used at its deepest,
                     1 - min margin_v (attack) / min margin_v (nominal twin) over the attack window
                     [onset, onset + duration], all hubs, clipped at 0; 1 = a violation
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seams.analysis.figures import new_figure, save
from seams.analysis.make_figures import ci

SURFACE = {"fdi": "Grid telemetry", "aggregator": "Aggregator", "navigation": "Navigation"}
BOX_STYLE = {"grey": ("tab:gray", "Grey box"), "white": ("tab:blue", "White box")}
EPS_STYLE = {True: ("tab:blue", "With ε"), False: ("tab:red", "Without ε")}
P_TOL_KW = 1e-3


def attackers(runs):
    """(name, mode, surface, box, year, use_epsilon, stress, rounds) for every trained attacker
    under runs. Tags: 2045, noeps, stress (no detection penalty or termination, larger bounds),
    r<n> (training continued to n rounds)."""
    out = []
    for d in sorted(Path(runs).glob("*/eval.parquet")):
        name = d.parent.name
        mode, surface, box, *tag = name.split("_")
        out.append(dict(name=name, mode=mode, surface=surface, box=box,
                        year="2045" if "2045" in tag else "2035", use_epsilon="noeps" not in tag,
                        stress="stress" in tag, rounds=len(list(d.parent.glob("q_r*.json")))))
    return out


def latest(df):
    """One attacker per configuration: the longest-trained one."""
    key = ["mode", "surface", "box", "year", "use_epsilon", "stress"]
    best = df.groupby(key).rounds.transform("max")
    return df[df.rounds == best]


def trip_delay_veh_h(att_dir, nom_dir):
    a, n = (pd.read_parquet(Path(d) / "trips.parquet").dropna(subset=["arrive_s"]) for d in (att_dir, nom_dir))
    m = a.merge(n, on="vehicle_id", suffixes=("_a", "_n"))
    extra = (m.arrive_s_a - m.depart_s_a) - (m.arrive_s_n - m.depart_s_n)
    return float(extra.sum() / 3600.0)


def gap_periods(run_dir):
    """Abstraction gap of one run (see the module docstring). Contract row k monitors fine steps
    per*k .. per*k + per - 1 (t_fine is the fine-step index); the P_bar it decides is in force
    over the next period."""
    clock = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())["clock"]
    per = round(float(clock["contract_period_s"]) / float(clock["fine_period_s"]))
    c = pd.read_parquet(Path(run_dir) / "contract.parquet")
    f = pd.read_parquet(Path(run_dir) / "fine.parquet")
    rho = [col for col in c.columns if col.startswith("rho_")]
    r = c[rho].astype(float).where(np.isfinite(c[rho].astype(float)))
    c["ok"] = (r.fillna(0.0) >= 0).all(axis=1) & ~c.blamed.fillna(False).astype(bool)
    coarse_ok = c.groupby("k").ok.all()
    f["k"] = (f.t_fine // per).astype(int)
    cap = c[c.hub != "HOME"][["k", "hub"]].copy()
    cap["Pbar"] = [row[f"Pbar_{row.hub}"] for _, row in c[c.hub != "HOME"].iterrows()]
    cap["k"] += 1
    f = f.merge(cap, on=["k", "hub"], how="left")
    bad = (f.margin_v < 0) | (f.margin_loading < 0) | (f.P_sched > f.Pbar + P_TOL_KW)
    fine_bad = bad.groupby(f.k).any()
    both = coarse_ok.to_frame("ok").join(fine_bad.rename("bad"), how="inner")
    return int((both.ok & both.bad).sum())


def headroom_used(att_dir, nom_dir, onset_s, duration_s):
    """Share of the nominal twin's smallest voltage margin over the attack window that the attack
    used (see the module docstring); NaN if the twin itself has no headroom there."""
    lo, hi = [], []
    for d in (att_dir, nom_dir):
        f = pd.read_parquet(Path(d) / "fine.parquet", columns=["t_fine", "margin_v"])
        clock = yaml.safe_load((Path(d) / "config.yaml").read_text())["clock"]
        k0 = round(onset_s / float(clock["fine_period_s"]))
        k1 = k0 + round(duration_s / float(clock["fine_period_s"]))
        lo.append(float(f[f.t_fine.between(k0, k1)].margin_v.min()))
    m_att, m_nom = lo
    return max(0.0, 1.0 - m_att / m_nom) if m_nom > 0 else np.nan


def summary(runs):
    rows = []
    for a in attackers(runs):
        d = Path(runs) / a["name"]
        ev = pd.read_parquet(d / "eval.parquet")
        w = yaml.safe_load((d / "rl.yaml").read_text())["reward"]
        for _, r in ev.iterrows():
            # inter-sample: the return is margin consumed + bonus x new violations - penalty x detected
            # (logged directly as ep_consumed since the episode tracks it)
            consumed = (r.ep_consumed if "ep_consumed" in ev else
                        r["return"] - w["violation_bonus"] * r.ep_violations + w["detection_penalty"] * r.ep_detected)
            att, nom = d / "eval" / f"attack_s{r.seed}", d / "eval" / f"nominal_s{r.seed}"
            rows.append({**a, "seed": int(r.seed), "arm_target": r.arm_target, "arm_onset_s": r.arm_onset_s,
                         "arm_duration_s": r.arm_duration_s, "detected": bool(r.ep_detected),
                         "episode_damage": float(r.ep_damage), "episode_violations": int(r.ep_violations),
                         "margin_consumed": float(consumed) if a["mode"] == "inter" else np.nan,
                         "extra_unserved_kWh": r.att_total_unserved_kWh - r.nom_total_unserved_kWh,
                         "extra_refused": (r.att_refused_vehicles or 0) - (r.nom_refused_vehicles or 0),
                         "unsound_periods": r.att_unsound_periods,
                         "trip_delay_veh_h": trip_delay_veh_h(att, nom),
                         "headroom_used": (100 * headroom_used(att, nom, r.arm_onset_s, r.arm_duration_s)
                                           if a["mode"] == "inter" else np.nan),
                         "gap_att": gap_periods(att), "gap_nom": gap_periods(nom)})
    return pd.DataFrame(rows)


def _bars(ax, labels, groups, style, values):
    """Grouped bars: labels on x, one bar per group key, mean and bootstrap 95 % CI."""
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        stats = [ci(values(lab, g)) for lab in labels]
        m = [s[0] for s in stats]
        err = np.array([[s[1] for s in stats], [s[2] for s in stats]])
        color, name = style[g]
        ax.bar(np.arange(len(labels)) + (i - (len(groups) - 1) / 2) * w, m, w, yerr=err, capsize=4,
               color=color, label=name)
    ax.axhline(0, color="k", lw=0.8)


def figures(df, runs, out, pdf):
    made = []
    df = latest(df)
    within = df[(df["mode"] == "within") & ~df.stress]
    surf = [s for s in SURFACE if s in set(within.surface)]
    boxes = [b for b in BOX_STYLE if b in set(within.box)]
    for col, ylabel, name in (("extra_unserved_kWh", "Extra unserved energy [kWh]", "unserved"),
                              ("trip_delay_veh_h", "Extra trip delay [veh·h]", "trip_delay"),
                              ("detected", "Detection rate", "detection"),
                              ("gap_att", "Abstraction-gap periods", "gap")):
        fig, ax = new_figure(ylabel=ylabel)
        _bars(ax, surf, boxes, BOX_STYLE,
              lambda s, b: within[(within.surface == s) & (within.box == b)][col].astype(float))
        ax.set_xticks(range(len(surf)), [SURFACE[s] for s in surf])
        if col in ("detected", "gap_att") and within[col].max() <= 1:
            ax.set_ylim(0, 1)
        ax.legend(fontsize=12)
        made += save(fig, out, f"rq3_within_{name}", pdf)

    inter = df[df["mode"] == "inter"]
    for year, stress in sorted(set(zip(inter.year, inter.stress))):
        sub = inter[(inter.year == year) & (inter.stress == stress)]
        group = year + ("_stress" if stress else "")
        surf = [s for s in SURFACE if s in set(sub.surface)]
        eps = [e for e in EPS_STYLE if e in set(sub.use_epsilon)]
        for col, ylabel, name in (("margin_consumed", "Φ_safe margin consumed", "margin"),
                                  ("headroom_used", "Voltage headroom used [%]", "headroom"),
                                  ("episode_violations", "Φ_safe violation steps", "violations"),
                                  ("extra_unserved_kWh", "Extra unserved energy [kWh]", "unserved"),
                                  ("trip_delay_veh_h", "Extra trip delay [veh·h]", "trip_delay"),
                                  ("detected", "Detection rate", "detection"),
                                  ("gap_att", "Abstraction-gap periods", "gap")):
            fig, ax = new_figure(ylabel=ylabel)
            _bars(ax, surf, eps, EPS_STYLE,
                  lambda s, e: sub[(sub.surface == s) & (sub.use_epsilon == e)][col].astype(float))
            ax.set_xticks(range(len(surf)), [SURFACE[s] for s in surf])
            if col in ("detected", "episode_violations", "gap_att") and sub[col].max() <= 1:
                ax.set_ylim(0, 1)
            if col == "headroom_used":
                ax.axhline(100, color="k", ls="--", lw=1.2, label="Violation")
                ax.set_ylim(0, 110)
            ax.legend(fontsize=12)
            made += save(fig, out, f"rq3_inter_{name}_{group}", pdf)

    for a in attackers(runs):
        made += learning_curve(Path(runs) / a["name"], out, pdf)
    return made


def learning_curve(d, out, pdf, window=50):
    """SAC episode return over all rounds (rolling mean), round boundaries dashed."""
    parts = sorted(d.glob("sac_monitor_r*.monitor.csv"), key=lambda p: int(p.name.split("_r")[1].split(".")[0]))
    if not parts:
        return []
    fig, ax = new_figure(xlabel="Episode", ylabel="Episode return")
    r, start = [], 0
    for p in parts:
        x = pd.read_csv(p, skiprows=1).r.tolist()
        if start:
            ax.axvline(start, color="k", ls="--", lw=0.8)
        r += x
        start += len(x)
    s = pd.Series(r)
    ax.plot(s.index, s, color="tab:gray", alpha=0.3, lw=0.6)
    ax.plot(s.index, s.rolling(window, min_periods=1).mean(), color="tab:blue", lw=2)
    return save(fig, out, f"rq3_learning_{d.name}", pdf)


def q_rounds(runs):
    """Best-arm Q value per training round, per attacker (convergence table)."""
    rows = {}
    for a in attackers(runs):
        qs = sorted(Path(runs, a["name"]).glob("q_r*.json"))
        rows[a["name"]] = [round(max(json.loads(p.read_text())), 2) for p in qs]
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/seams_runs/rq3")
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    runs, results = Path(args.runs).expanduser(), Path(args.results)
    df = summary(runs)
    df.to_csv(results / "rq3_summary.csv", index=False)
    made = figures(df, runs, results / "figures", args.pdf or None)
    cols = ["detected", "episode_damage", "episode_violations", "margin_consumed", "headroom_used", "extra_unserved_kWh", "trip_delay_veh_h",
            "extra_refused", "unsound_periods", "gap_nom", "gap_att"]
    print(df.groupby("name")[cols].mean().round(2).to_string())
    print("best-arm Q per round:", json.dumps(q_rounds(runs), indent=1))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
