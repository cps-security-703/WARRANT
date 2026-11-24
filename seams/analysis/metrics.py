from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DOMAINS = ("transport", "charging", "grid")


def load_run(run_dir):
    run_dir = Path(run_dir)
    frames = {t: pd.read_parquet(run_dir / f"{t}.parquet") for t in ("fine", "contract", "session", "events")}
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return frames, cfg


def _sustained(flag, n):
    run = flag.groupby((~flag).cumsum()).cumsum()
    return run >= n


def domain_failures(fine, cfg):
    m = cfg["metrics"]
    fine_s = float(cfg["clock"]["fine_period_s"])
    p_full = {h["id"]: float(h["p_allow_kw"]) for h in cfg["hubs"]}
    f = fine.sort_values(["hub", "t_fine"]).copy()
    f["transport"] = f.spillback_m > m["spillback_fail_m"]
    curtailed = f.d_h < (1.0 - m["curtail_frac"]) * f.hub.map(p_full)
    n = max(1, round(m["curtail_min_s"] / fine_s))
    f["curtail_sustained"] = curtailed.groupby(f.hub, group_keys=False).apply(lambda s: _sustained(s, n))
    f["charging"] = f.curtail_sustained | f.get("offline", False)
    f["grid"] = (f.margin_v < 0) | (f.margin_loading < 0)
    return f.groupby("t_fine")[list(DOMAINS)].any()


def attack_window(events):
    on = events[events.event.str.startswith("attack_onset")]
    if on.empty:
        return None
    origins = sorted({e.split(":", 1)[1] for e in on.event})
    return origins, int(on.k.min()), sorted(set(on.hub))


def cascade(run_dir):
    frames, cfg = load_run(run_dir)
    fine, events = frames["fine"], frames["events"]
    fine_s = float(cfg["clock"]["fine_period_s"])
    fails = domain_failures(fine, cfg)
    win = attack_window(events)
    out = {"run_id": fine.run_id.iloc[0]}
    if win is None:
        out.update({f"fail_{d}": bool(fails[d].any()) for d in DOMAINS})
        return out
    origins, k_on, targets = win
    after = fails[fails.index >= k_on]
    first = {d: (int(after.index[after[d]][0]) if after[d].any() else None) for d in DOMAINS}
    failing = {d for d, k in first.items() if k is not None}
    others = [first[d] for d in DOMAINS if d not in origins and first[d] is not None]
    ids = events[(events.event == "ids_alarm") & (events.k >= k_on)]
    trips = events[(events.event == "trip") & (events.k >= k_on)]
    out.update(
        origin=",".join(origins), target=",".join(targets), onset_k=k_on,
        depth=len(failing | set(origins)),
        failing=",".join(sorted(failing)),
        tau_cross_s=(min(others) - k_on) * fine_s if others else np.nan,
        tau_detect_ids_s=(int(ids.k.min()) - k_on) * fine_s if not ids.empty else np.nan,
        trips=len(trips),
        **{f"first_{d}_s": ((first[d] - k_on) * fine_s if first[d] is not None else np.nan) for d in DOMAINS},
    )
    return out


def availability_cost(run_dir):
    frames, _ = load_run(run_dir)
    s = frames["session"]
    out = {
        "sessions": len(s),
        "unserved_kWh": float(s.unserved_kWh.sum()),
        "delivered_kWh": float(s.delivered_kWh.sum()),
    }
    home_file = Path(run_dir) / "home.parquet"
    home = pd.read_parquet(home_file) if home_file.exists() else pd.DataFrame()
    home_unserved = float(home[home.ends_in_horizon].unserved_kWh.sum()) if len(home) else 0.0
    out.update(home_unserved_kWh=home_unserved, total_unserved_kWh=out["unserved_kWh"] + home_unserved)
    return out


def attribution(run_dir):
    frames, cfg = load_run(run_dir)
    c, events = frames["contract"], frames["events"]
    if "theta" not in c.columns:
        return {}
    fine_s = float(cfg["clock"]["fine_period_s"])
    per = int(round(float(cfg["clock"]["contract_period_s"]) / fine_s))
    # per-formula robustness columns (rho_rho_* are the domain-level A/G aggregates)
    rho_cols = [col for col in c.columns if col.startswith("rho_") and not col.startswith("rho_rho_")]
    violated = c[(c[rho_cols] < 0).any(axis=1)]
    inc = events[events.event.str.startswith("incident:")].copy()
    inc["domain"] = inc.event.str.split(":").str[1]
    win = attack_window(events)
    out = {"theta": c.theta.iloc[0]}
    if win is None:
        out.update(false_incidents=int(inc.k.nunique()))
        return out
    origins, k_on, _ = win
    pre = inc[inc.k < k_on]
    post = inc[inc.k >= k_on]
    first_k = post.k.min() if not post.empty else None
    first = sorted(set(post[post.k == first_k].domain)) if first_k is not None else []
    v_after = violated[(violated.k + 1) * per - 1 >= k_on]  # periods ending at or after onset
    out.update(
        false_incidents=int(pre.k.nunique()),
        tau_detect_s=(((int(v_after.k.min()) + 1) * per - k_on) * fine_s) if not v_after.empty else np.nan,
        first_blame=",".join(first),
        attribution_ok=bool(first) and set(first) <= set(origins),
        tau_blame_s=((int(first_k) + 1 - k_on) * fine_s) if first_k is not None else np.nan,
    )
    return out


def _safe_scope(cfg):
    from seams.cosim.config import resolve_path
    path = (cfg.get("renegotiation") or {}).get("theta_safe_file")
    if not path or not resolve_path(path).exists():
        return None
    meta = yaml.safe_load(resolve_path(path).read_text()).get("meta", {})
    return meta.get("safe_certificate", {}).get("scope_bg_max")


def certification(run_dir):
    frames, cfg = load_run(run_dir)
    c, f = frames["contract"], frames["fine"]
    if "certified" not in c.columns:
        return {}
    per = int(round(float(cfg["clock"]["contract_period_s"]) / float(cfg["clock"]["fine_period_s"])))
    K = int(cfg["verifier"]["horizon_periods"])
    cert = c.groupby("k").certified.first()
    viol = f[(f.margin_v < 0) | (f.margin_loading < 0)]
    scope = _safe_scope(cfg)
    out_of_scope = viol[viol.bg_mult > scope] if scope is not None else viol.iloc[0:0]
    bad_fine = viol[~viol.index.isin(out_of_scope.index)].t_fine.unique()
    bad_period = set((bad_fine // per).tolist())
    unsound = [int(n) for n, ok in cert.items() if ok and any((n + j) in bad_period for j in range(1, K + 1))]
    return {
        "certified_share": float(cert.mean()),
        "unsound_periods": len(unsound),
        "out_of_scope_violations": int(out_of_scope.t_fine.nunique()),
        # guideline metrics: fine steps violating Phi_safe within theta_safe's scope (RQ5 gate: 0),
        # and the Phi_safe margin (min over the day of the voltage and loading margins)
        "phi_safe_violation_steps": int(len(bad_fine)),
        "phi_safe_margin_v_pu": float(f.margin_v.min()),
        "phi_safe_margin_loading": float(f.margin_loading.min()),
        "v2_ms_mean": float(c.v2_ms.mean()),
        "v2_ms_max": float(c.v2_ms.max()),
        "refused": int(f.refused.sum()) if "refused" in f.columns else 0,
        "limiter_steps": int(f.limiter_active.sum()) if "limiter_active" in f.columns else 0,
    }


def renegotiation(run_dir):
    frames, cfg = load_run(run_dir)
    c, ev = frames["contract"], frames["events"]
    if "status" not in c.columns:
        return {}
    fine_s = float(cfg["clock"]["fine_period_s"])
    per_hub = c.groupby("k").first()
    out = {f"periods_{s}": int((per_hub.status == s).sum()) for s in ("static", "renegotiated", "fallback", "agreed")}
    out["solve_ms_max"] = float(c.v2_ms.max())
    win = attack_window(ev)
    if win is not None:
        k_on = win[1]
        blame = ev[ev.event.str.startswith("incident:") & (ev.k >= k_on)]
        act = ev[ev.event.str.startswith(("theta:", "mode_enter:")) & (ev.k >= k_on)]
        if not blame.empty and not act.empty:
            k_b = int(blame.k.min())
            later = act[act.k >= k_b]
            out["tau_reneg_s"] = float((int(later.k.min()) - k_b) * fine_s) if not later.empty else np.nan
    return out


def containment(run_dir, baseline_dir):
    a, r, base = attribution(run_dir), renegotiation(run_dir), cascade(baseline_dir)
    t = a.get("tau_detect_s", np.nan) + r.get("tau_reneg_s", 0.0)
    return {
        "tau_cross_baseline_s": base.get("tau_cross_s", np.nan),
        "tau_detect_s": a.get("tau_detect_s", np.nan),
        "tau_reneg_s": r.get("tau_reneg_s", np.nan),
        "window_margin_s": base.get("tau_cross_s", np.nan) - t,
        "depth": cascade(run_dir).get("depth"),
        "baseline_depth": base.get("depth"),
    }
