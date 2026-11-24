import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from seams.cosim.config import load_config, resolve_path


def _nominal(job):
    from seams.cosim.master import Master
    cfg_path, overrides, seed, run_dir = job
    # permissive theta: shields enforce nothing; no verifier (epsilon does not exist yet)
    cfg = load_config(cfg_path, {**overrides, "run.seed": seed, "attacks": [],
                                 "contracts.force_calibration": True, "verifier": None,
                                 "renegotiation": None, "ids.kind": None})  # the BDD is fitted here
    Master(cfg, f"calib_s{seed}", run_dir=run_dir).run()
    return pd.read_parquet(run_dir / "contract.parquet"), pd.read_parquet(run_dir / "fine.parquet")


def calibration_days(cfg, n):
    b = cfg["grid"]["background"]
    if b.get("source", "hourly") != "nrel_la" or b.get("day"):
        return [{} for _ in range(n)]
    from seams.domains.grid.loadshape import candidate_days
    d = candidate_days(b)
    d = d.assign(peak=d.filter(regex="^q[0-9]").max(axis=1)).sort_values("peak").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(d) - 1, n)).astype(int))
    return [{"grid.background.day": d.date[i]} for i in idx] + [{} for _ in range(n - len(idx))]


def bg_levels(cfg):
    from seams.domains.grid.loadshape import envelope_top
    top = envelope_top(cfg["grid"]["background"])
    return lambda fracs: [round(float(f) * top, 4) for f in fracs]


def epsilon_section(cfg, contract, fine, seeds, p_bar):
    from seams.contracts import epsilon as E

    e = cfg["contracts"]["epsilon"]
    hubs = [h["id"] for h in cfg["hubs"]]
    bg = bg_levels(cfg)
    per = round(float(cfg["clock"]["contract_period_s"]) / float(cfg["clock"]["fine_period_s"]))
    # beyond the V2 horizon too: distrust mode re-anchors up to drift_horizon periods back
    K = max(int(cfg["verifier"]["horizon_periods"]), int(e.get("drift_horizon", 0)))
    D, cpl = E.drift_envelope(fine, contract, hubs, per, K, e["drift_quantile"], parties=list(p_bar))
    rng = np.random.default_rng(np.random.SeedSequence(list(seeds)))
    lin = E.linearization(cfg, rng, bg(e["lin_bg_levels"]), e["lin_anchors"], e["lin_samples"], p_bar)
    load = E.load_limit(cfg, bg([e["load_bg_worst"]])[0], p_bar)
    ceil = E.voltage_ceiling(cfg, rng, bg(e["ceiling_bg_levels"]), e["lin_anchors"], e["lin_samples"], p_bar)
    gamma = E.step_gamma(cfg, rng, bg(e["lin_bg_levels"]), e["lin_anchors"], e["lin_samples"], p_bar,
                         float(cfg["contracts"]["sensitivity_dp_kw"]))
    return {
        "D": {h: {k: v * (1 + e["drift_margin"]) for k, v in D[h].items()} for h in hubs},
        "lin": {h: max(lin[h], 0.0) * (1 + e["lin_margin"]) for h in hubs},
        "P_load_max": load,
        "cpl_V": cpl,
        "V_ceiling": {h: ceil[h] + e["ceiling_margin"] for h in hubs},
        "gamma": {h: gamma[h] * (1 + e["lin_margin"]) for h in hubs},
    }


def write_theta_safe(cfg, theta_static):
    from seams.contracts import epsilon as E

    r = cfg["renegotiation"]
    p_safe = {h: float(v) * r["safe_frac"] for h, v in theta_static["P_bar"].items()}
    ok, evidence = E.certify_safe(cfg, p_safe, bg_levels(cfg)(r["safe_bg_levels"]), r["safe_v_margin"])
    # how much of the real operating envelope lies above the certificate's scope
    from seams.domains.grid.loadshape import candidate_days
    b = cfg["grid"]["background"]
    q = candidate_days(b).filter(regex="^q[0-9]").to_numpy() * float(b.get("peak_mult", 1.0))
    evidence["out_of_scope_share"] = float((q > evidence["scope_bg_max"]).mean())
    certified = bool(ok) or evidence["out_of_scope_share"] <= float(r.get("safe_max_out_of_scope", 0.0))
    safe = dict(theta_static, version="safe-v1", P_bar=p_safe, P_assumed=p_safe)
    safe["meta"] = dict(theta_static["meta"], offline_certified=certified, safe_certificate=evidence)
    path = resolve_path(r["theta_safe_file"])
    path.write_text(yaml.safe_dump(safe, sort_keys=False))
    print(f"theta_safe: P_bar={p_safe} offline_certified={certified} evidence={evidence} -> {path}")
    if not certified:
        raise SystemExit("theta_safe failed its offline certificate; lower renegotiation.safe_frac")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, e.g. clock.contract_period_s=60")
    args = ap.parse_args(argv)
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, overrides)
    c, cal = cfg["contracts"], cfg["contracts"]["calibration"]
    out_root = resolve_path(cfg["run"]["out_dir"]) / f"calibration_{int(cfg['clock']['contract_period_s'])}s"
    jobs = [(args.config, {**overrides, **day}, s, out_root / f"s{s}")
            for s, day in zip(cal["seeds"], calibration_days(cfg, len(cal["seeds"])))]
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_nominal, jobs))
    df = pd.concat([c for c, _ in results], ignore_index=True)
    fine = pd.concat([f for _, f in results], ignore_index=True)
    hub_ids = [h["id"] for h in cfg["hubs"]]
    df = df[df.hub.isin(hub_ids)].copy()  # HOME rows carry no arrivals or physics

    lam_bar, beta, evidence = {}, {}, {}
    for (h, run), g in df.groupby(["hub", "run_id"]):
        g = g.sort_values("k")
        for w in c["windows"]:
            df.loc[g.index, f"lam_w{w}"] = g.lam_sig.rolling(int(w)).sum()
    for h, g in df.groupby("hub"):
        lam_bar[h], beta[h] = {}, {}
        for w in c["windows"]:
            lam_q = float(np.quantile(g[f"lam_w{w}"].dropna(), cal["lam_quantile"]))
            lam_bar[h][int(w)] = int(math.ceil(lam_q * (1 + cal["lam_margin_frac"]))) + int(cal["lam_margin"])
            r = g[f"resid_{w}"].abs().dropna()
            beta[h][int(w)] = float(np.quantile(r, cal["beta_quantile"]) * (1 + cal["beta_margin"]))
        evidence[h] = {
            **{f"lam_w{w}_max": int(g[f"lam_w{w}"].max()) for w in c["windows"]},
            **{f"resid_{w}_max": float(g[f"resid_{w}"].abs().max()) for w in c["windows"]},
            **{f"resid_{w}_p50": float(g[f"resid_{w}"].abs().median()) for w in c["windows"]},
        }
    phi = cfg["grid"]["phi_safe"]
    p_bar = {h["id"]: float(h["p_allow_kw"]) for h in cfg["hubs"]}
    if cfg["grid"].get("home_ev", {}).get("enabled"):
        # the honest home portfolio's largest calibration-day draw, plus a margin: an honest
        # aggregator never exceeds it on a nominal day
        peak = float(fine[fine.hub == hub_ids[0]].home_req_kw.max())
        p_bar["HOME"] = peak * (1 + float(cal.get("home_p_margin", 0.25)))
        evidence["HOME"] = {"home_kw_max": peak}
    theta = {
        "version": "static-v1",
        "lam_bar": lam_bar,
        "P_bar": p_bar,
        "V_lo": float(phi["v_min"]), "V_hi": float(phi["v_max"]),
        "beta": beta,
        "meta": {"calibration_seeds": list(cal["seeds"]), "periods": int(len(df) / len(lam_bar)),
                 "rules": dict(cal), "evidence": evidence},
    }
    if c.get("epsilon") and cfg.get("verifier"):
        theta["epsilon"] = epsilon_section(cfg, df, fine, cal["seeds"], p_bar)
        theta["meta"]["epsilon_rules"] = dict(c["epsilon"])
    path = resolve_path(c["theta_file"])
    path.write_text(yaml.safe_dump(theta, sort_keys=False))
    ids = cfg.get("ids") or {}
    ref = float(ids.get("bdd_reference_delta_s", cfg["clock"]["contract_period_s"]))
    if "bdd" in str(ids.get("kind", "")).split("+") and float(cfg["clock"]["contract_period_s"]) == ref:
        import json
        from seams.baselines.bdd import calibrate as bdd_calibrate
        t = cfg["grid"].get("telemetry", {})
        sigma = float(t.get("noise_sigma_pu", 0.0))
        model = bdd_calibrate(fine, hub_ids, sigma, sigma, float(ids.get("bdd_alpha", 1e-4)),
                              int(ids.get("bdd_persist", 2)), float(ids.get("bdd_margin", 1.2)),
                              float(ids.get("bdd_est_margin", 0.25)))
        resolve_path(ids["bdd_file"]).write_text(json.dumps(model, indent=2))
        print(f"BDD: threshold {model['threshold']:.1f} (nominal held max {model['nominal_held_J_max']:.1f}) -> {ids['bdd_file']}")
    if cfg.get("renegotiation") and "epsilon" in theta:
        write_theta_safe(cfg, theta)
    for d in out_root.iterdir():
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() or f.is_symlink():
                    f.unlink()
    print(yaml.safe_dump({k: theta[k] for k in ("lam_bar", "P_bar", "beta")}, sort_keys=False))
    print("evidence:", yaml.safe_dump(evidence, sort_keys=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
