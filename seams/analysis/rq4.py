import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from seams.analysis.figures import new_figure, save
from seams.analysis.rq3 import _bars

KEY = ["scenario", "strength", "seed"]
# naive (plan_mode lock) and floor protocols per backbone; conditions not run yet are skipped
AGENTS = {"agents": ("tab:blue", "Gemini"), "agents_floor": ("navy", "Gemini+floor"),
          "agents_haiku": ("tab:orange", "Haiku"), "agents_haiku_floor": ("saddlebrown", "Haiku+floor")}
ALL = {"classical": ("tab:gray", "Classical"), **AGENTS}
GROUPS = ("foreseen", "unforeseen")
GROUP_LABEL = {"foreseen": "Foreseen", "unforeseen": "Unforeseen"}
HUBS = ("H1", "H2", "H3")


def group_of(scenario, foreseen):
    if scenario in foreseen:
        return "foreseen"
    return "unforeseen" if scenario.startswith("u_") else "nominal"


def negotiations(runs, foreseen):
    rows = []
    for f in sorted(Path(runs).glob("agents*/negotiations.jsonl")):
        run = f.parent.name
        cond = run.split("_EDF_")[0]
        scen = re.sub(r"_x[-0-9.]+$", "", re.sub(r"_s\d+$", "", run.split("_EDF_")[1]))
        for line in f.read_text().splitlines():
            r = json.loads(line)
            row = dict(condition=cond, scenario=scen, group=group_of(scen, foreseen), run=run, t_s=r.get("t_s"),
                       trigger=r.get("trigger"), default_status=r.get("default_status"), skipped=r.get("skipped"),
                       agreement=bool(r.get("agreement")), timeout=bool(r.get("timeout")), latency_s=r.get("latency_s"),
                       n_candidates=r.get("n_candidates") or 0, n_certified=r.get("n_certified") or 0,
                       n_malformed=r.get("n_malformed") or 0, rounds=r.get("rounds"), chosen_source=r.get("chosen_source"))
            for side in ("default_cost", "chosen_cost"):
                for role, v in (r.get(side) or {}).items():
                    row[f"{side}_{role}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def paired(summary):
    c = summary[summary.condition == "classical"].set_index(KEY)
    out = []
    for cond in AGENTS:
        a = summary[summary.condition == cond].set_index(KEY)
        if a.empty:
            continue
        d = pd.DataFrame({"condition": cond, "group": a.group,
                          "d_unserved_kWh": a.unserved_kWh - c.unserved_kWh,
                          "d_home_unserved_kWh": a.home_unserved_kWh - c.home_unserved_kWh,
                          "d_refused": a.refused_vehicles - c.refused_vehicles,
                          "d_trip_delay_h": (a.trip_delay_s - c.trip_delay_s) / 3600.0,
                          "d_fallback": a.periods_fallback - c.periods_fallback}).dropna(subset=["d_unserved_kWh"])
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True)


def lock_in(runs, cond):
    rows = []
    prefix = f"{cond}_EDF_"
    cols = ["k", "hub", "status"] + [f"Pbar_{h}" for h in HUBS]
    for d in sorted(Path(runs).glob(prefix + "*/contract.parquet")):
        twin = Path(runs) / d.parent.name.replace(prefix, "classical_EDF_", 1) / "contract.parquet"
        if not twin.exists():
            continue
        a, c = (pd.read_parquet(p, columns=cols).query("hub == 'H1'") for p in (d, twin))
        m = a.merge(c, on="k", suffixes=("_a", "_c"))
        g = m[m.status_a == "agreed"].copy()
        if g.empty:
            continue
        g["lead"] = g.groupby((g.k.diff() != 1).cumsum()).cumcount() + 1
        g["dP"] = sum(g[f"Pbar_{h}_a"] - g[f"Pbar_{h}_c"] for h in HUBS) / len(HUBS)
        g["twin_fallback"] = g.status_c == "fallback"
        rows.append(g[["lead", "dP", "twin_fallback"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["lead", "dP", "twin_fallback"])


def containment(summary, runs, rq2_summary, period_s):
    po = rq2_summary[rq2_summary.condition == "protection_only"].set_index(["scenario", "strength", "seed"]).tau_cross_s
    rows = []
    for _, r in summary.iterrows():
        tc = po.get((r.scenario, r.strength, r.seed))
        if tc is None or pd.isna(tc) or pd.isna(r.tau_blame_s):
            continue
        row = dict(condition=r.condition, scenario=r.scenario, strength=r.strength, seed=r.seed, tau_cross_s=tc,
                   classical_margin_s=tc - (r.tau_blame_s + r.tau_reneg_s), agent_margin_s=np.nan)
        f = Path(runs) / r.run_id / "negotiations.jsonl"
        if f.exists():
            negs = [json.loads(line) for line in f.read_text().splitlines()]
            agreed = [n for n in negs if n.get("agreement")]
            if negs and agreed:
                wait = agreed[0]["t_s"] - negs[0]["t_s"] + np.ceil(agreed[0]["latency_s"] / period_s) * period_s
                row["agent_margin_s"] = tc - (r.tau_blame_s + wait)
        rows.append(row)
    return pd.DataFrame(rows)


def table(summary, pairs, negs):
    t = summary.groupby(["condition", "group", "scenario"]).agg(
        runs=("seed", "size"), unserved_kWh=("unserved_kWh", "mean"), periods_fallback=("periods_fallback", "mean"),
        periods_agreed=("periods_agreed", "mean"), unsound_periods=("unsound_periods", "sum"),
        out_of_scope_violations=("out_of_scope_violations", "sum"), certified_share=("certified_share", "min"))
    d = pairs.groupby(["condition", "scenario"])[["d_unserved_kWh", "d_refused", "d_trip_delay_h", "d_fallback"]].mean()
    n = negs.groupby(["condition", "scenario"]).agg(
        negotiations=("agreement", "size"), agreement_rate=("agreement", "mean"),
        skipped=("skipped", lambda s: s.notna().mean()), latency_med_s=("latency_s", "median"),
        latency_max_s=("latency_s", "max"), candidates=("n_candidates", "sum"), certified=("n_certified", "sum"),
        malformed=("n_malformed", "sum"))
    return t.reset_index().merge(d.reset_index(), on=["condition", "scenario"], how="left") \
            .merge(n.reset_index(), on=["condition", "scenario"], how="left")


def _legend(ax, n):
    lo, hi = ax.get_ylim()
    rows = -(-n // (2 if n > 3 else 1))
    ax.set_ylim(lo, hi + (hi - lo) * 0.13 * rows)
    ax.legend(fontsize=12, ncol=2 if n > 3 else 1, loc="upper center", columnspacing=0.8, handlelength=1.2)


def figures(summary, pairs, negs, runs, out, pdf, deadline_s):
    made = []
    agents = {c: v for c, v in AGENTS.items() if c in set(summary.condition)}
    conds = {"classical": ALL["classical"], **agents}
    run_level = negs.groupby(["condition", "group", "run"])

    # time to agreement: CDF against the deadline
    fig, ax = new_figure(xlabel="Time to agreement [s]", ylabel="Share of agreements")
    for cond, (color, name) in agents.items():
        x = np.sort(negs[(negs.condition == cond) & negs.agreement].latency_s.dropna().to_numpy())
        if len(x):
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, label=name)
    ax.axvline(deadline_s, color="k", ls="--", label="Deadline")
    ax.set_xlim(0, deadline_s * 1.05)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=12)
    made += save(fig, out, "rq4_time_to_agreement", pdf)

    # per-run rates, foreseen vs unforeseen
    per_run = run_level.agg(cand=("n_candidates", "sum"), cert=("n_certified", "sum"),
                            agree=("agreement", "mean")).reset_index()
    per_run["cert_share"] = per_run.cert / per_run.cand.where(per_run.cand > 0)
    for col, ylabel, name in (("cert_share", "Share of proposals certified", "certified_share"),
                              ("agree", "Agreement rate", "agreement_rate")):
        fig, ax = new_figure(ylabel=ylabel)
        _bars(ax, GROUPS, list(agents), agents,
              lambda g, c: per_run[(per_run.group == g) & (per_run.condition == c)][col])
        ax.set_xticks(range(len(GROUPS)), [GROUP_LABEL[g] for g in GROUPS])
        ax.set_ylim(0, 1)
        _legend(ax, len(agents))
        made += save(fig, out, f"rq4_{name}", pdf)

    # fallback periods per day: all three conditions
    fig, ax = new_figure(ylabel="Fallback periods per day")
    _bars(ax, GROUPS, list(conds), conds,
          lambda g, c: summary[(summary.group == g) & (summary.condition == c)].periods_fallback.astype(float))
    ax.set_xticks(range(len(GROUPS)), [GROUP_LABEL[g] for g in GROUPS])
    _legend(ax, len(conds))
    made += save(fig, out, "rq4_fallback_periods", pdf)

    # availability against the classical twin
    for col, ylabel, name in (("d_unserved_kWh", "Δ unserved energy [kWh]", "unserved_delta"),
                              ("d_fallback", "Δ fallback periods", "fallback_delta")):
        fig, ax = new_figure(ylabel=ylabel)
        _bars(ax, GROUPS, list(agents), agents,
              lambda g, c: pairs[(pairs.group == g) & (pairs.condition == c)][col].astype(float))
        ax.set_xticks(range(len(GROUPS)), [GROUP_LABEL[g] for g in GROUPS])
        _legend(ax, len(agents))
        made += save(fig, out, f"rq4_{name}", pdf)

    # every paired day, sorted: lock spreads to both sides of zero, floor stays at or below it
    fig, ax = new_figure(xlabel="Days, sorted", ylabel="Δ unserved vs. classical [kWh]")
    for cond, (color, name) in agents.items():
        d = np.sort(pairs[pairs.condition == cond].d_unserved_kWh.astype(float).to_numpy())
        ax.plot(np.arange(len(d)), d, color=color, lw=2, ls="-" if "floor" in cond else "--", label=name)
    ax.axhline(0, color="k", lw=0.8)
    _legend(ax, len(agents))
    made += save(fig, out, "rq4_paired_days", pdf)

    # per-operator forecast cost (the agents' own tools), default vs agreed contract
    agreed = negs[negs.agreement]
    for cond, (color, name) in agents.items():
        a = agreed[agreed.condition == cond]
        roles = [r for r in ("fleet", "charging", "utility") if f"default_cost_{r}" in a]
        style = {"default": ("tab:gray", "Default"), "chosen": (color, "Agreed")}
        fig, ax = new_figure(ylabel="Forecast operator cost")
        _bars(ax, roles, ["default", "chosen"], style, lambda r, s: a[f"{s}_cost_{r}"].astype(float))
        ax.set_xticks(range(len(roles)), [r.capitalize() for r in roles])
        ax.legend(fontsize=12)
        made += save(fig, out, f"rq4_operator_cost_{name.lower().replace('+', '_')}", pdf)

    # plan lock-in: agreed P_bar minus the classical twin's, by lead within the plan
    fig, ax = new_figure(xlabel="Periods since agreement", ylabel="Plan − classical P̄ [kW]")
    for cond, (color, name) in agents.items():
        li = lock_in(runs, cond)
        if li.empty:
            continue
        g = li[li.lead <= 15].groupby("lead").dP
        m, s, n = g.mean(), g.std(), g.size()
        ax.errorbar(m.index, m, yerr=1.96 * s / np.sqrt(n), color=color, marker="o", ms=4, capsize=3, label=name)
    ax.axhline(0, color="k", lw=0.8)
    _legend(ax, len(agents))
    made += save(fig, out, "rq4_plan_lock_in", pdf)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="~/seams_runs/rq4")
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--config", default="seams/config/phase8.yaml")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    runs, results = Path(args.runs).expanduser(), Path(args.results)
    cfg = yaml.safe_load(Path(args.config).read_text())
    foreseen = set(cfg["rq4"]["foreseen"])
    deadline = float(cfg["agents"]["deadline_s"])

    summary = pd.read_parquet(results / "rq4_summary.parquet")
    summary["group"] = [group_of(s, foreseen) for s in summary.scenario]
    negs = negotiations(runs, foreseen)
    pairs = paired(summary)
    t = table(summary, pairs, negs)
    t.to_csv(results / "rq4_table.csv", index=False)
    negs.to_csv(results / "rq4_negotiations.csv", index=False)
    if (results / "rq2_summary.parquet").exists():
        period = float(yaml.safe_load(Path("seams/config/phase5.yaml").read_text()).get("clock", {}).get("contract_period_s", 120))
        cw = containment(summary, runs, pd.read_parquet(results / "rq2_summary.parquet"), period)
        cw.to_csv(results / "rq4_containment.csv", index=False)
        print(cw.groupby("condition").agg(runs=("seed", "size"), classical_margin_min_s=("classical_margin_s", "min"),
                                          agreed=("agent_margin_s", "count"),
                                          agent_margin_med_s=("agent_margin_s", "median"),
                                          agreed_in_window=("agent_margin_s", lambda s: int((s > 0).sum()))).to_string())
    made = figures(summary, pairs, negs, runs, results / "figures", args.pdf or None, deadline)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(t.round(3).to_string(index=False))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
