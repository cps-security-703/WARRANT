"""Epsilon terms that link the coarse contract view to the fine physical run.

eps_lin[h]       largest gap between the full AC solve (regulators acting) and the verifier's
                 anchored quasi-steady-state model V0 + Sq (P - P0), over the operating envelope
                 (background level x anchor point x hub powers in [0, P_bar]), by sampling.
P_load_max[j]    hub power at which the hub transformer reaches 100 % loading, at the worst
                 (heaviest) background, other hubs at P_bar. Loading is then a linear bound.
D[h][k]          drift envelope: everything the anchored model misses k periods ahead on
                 nominal days (background drift, tap moves, within-period ripple), one-sided in
                 the undervoltage direction:
                 max(Vbar(n) + sum_j Sq_hj(n) (P_fine_j(t) - Pbar_j(n)) - V_fine(t)), t in period n+k.
V_ceiling[h]     regulated-feeder voltage ceiling over the whole envelope (overvoltage side).
cpl_V[h]         within-period ripple: |V_fine(t) - Vbar(period of t)|, reported for the
                 soundness lemma (it is contained in D).

All are empirical bounds: the soundness lemma uses them as assumptions and RQ3's inter-sample
attacker tests them.
"""
import numpy as np


def _p_bar(cfg, p_bar):
    """Power range per party: the hubs' honest allowance, plus HOME's if given."""
    out = {h["id"]: float(h["p_allow_kw"]) for h in cfg["hubs"]}
    out.update({j: float(v) for j, v in (p_bar or {}).items()})
    return out


def linearization(cfg, rng, bg_levels, n_anchor, n_samples, p_bar=None):
    """One-sided error of the verifier's anchored model V0 + Sq (P - P0), in the dangerous
    (undervoltage) direction: max(pred - actual). From each anchor (regulators settled), jump to
    a random hub-power vector and let the regulators re-settle from the anchor's taps, as they
    would within one fine step. Regulators only raise voltage after a load increase, so the
    error in this direction stays small; the overvoltage side is covered by voltage_ceiling()."""
    from seams.domains.grid.dss_env import GridAdapter

    grid = GridAdapter(cfg)
    hubs, parties = list(grid.hubs), grid.parties()
    p_bar = _p_bar(cfg, p_bar)
    err = {h: 0.0 for h in hubs}
    for bg in bg_levels:
        grid.set_background(bg)
        for _ in range(n_anchor):
            p0 = {j: float(rng.uniform(0, p_bar[j])) for j in parties}
            _set_all(grid, p0)
            v0 = grid.solve().v_lv
            taps0 = {r: grid._tap(r) for r in grid.regulators}
            Sq = grid.sensitivity_qss(p_bar)
            for _ in range(n_samples):
                p = {j: float(rng.uniform(0, p_bar[j])) for j in parties}
                for r, t in taps0.items():
                    grid._set_tap(r, t)
                _set_all(grid, p)
                v = grid.solve().v_lv
                for h in hubs:
                    pred = v0[h] + sum(Sq[h][j] * (p[j] - p0[j]) for j in parties)
                    err[h] = max(err[h], pred - v[h])
            for r, t in taps0.items():
                grid._set_tap(r, t)
    return err


def step_gamma(cfg, rng, bg_levels, n_anchor, n_samples, p_bar=None, dp_kw=2.0):
    """gamma[h]: the monitor's physics model (small-signal S, T published at the anchor) on a
    large power step, |dV - S dP - T dtap| per kW of total |dP|, worst case over random steps
    from settled anchors with the regulators acting. The grid's physics guarantee is
    beta_h,w + gamma_h * sum |dP| over the window: a linear model's large-signal error is part
    of what the grid publishes, so a big (legitimate or not) power step on the charging side
    does not look like a grid-side inconsistency."""
    from seams.domains.grid.dss_env import GridAdapter

    grid = GridAdapter(cfg)
    hubs, parties = list(grid.hubs), grid.parties()
    p_bar = _p_bar(cfg, p_bar)
    gamma = {h: 0.0 for h in hubs}
    for bg in bg_levels:
        grid.set_background(bg)
        for _ in range(n_anchor):
            p0 = {j: float(rng.uniform(0, p_bar[j])) for j in parties}
            _set_all(grid, p0)
            st0 = grid.solve()
            taps0 = {r: grid._tap(r) for r in grid.regulators}
            S, T = grid.sensitivity(dp_kw)
            for _ in range(n_samples):
                p = {j: float(rng.uniform(0, p_bar[j])) for j in parties}
                for r, t in taps0.items():
                    grid._set_tap(r, t)
                _set_all(grid, p)
                st = grid.solve()
                dtap = [a - b for a, b in zip(st.taps, st0.taps)]
                size = sum(abs(p[j] - p0[j]) for j in parties)
                for h in hubs:
                    pred = sum(S[h][j] * (p[j] - p0[j]) for j in parties)
                    pred += sum(T[h][r] * d for r, d in zip(grid.regulators, dtap))
                    gamma[h] = max(gamma[h], abs(st.v_lv[h] - st0.v_lv[h] - pred) / max(size, 1.0))
            for r, t in taps0.items():
                grid._set_tap(r, t)
            _set_all(grid, p0)
    return gamma


def _set_all(grid, p):
    """Set every party's power (HOME spread by its portfolio shares) and record it as the
    operating point the sensitivity probes restore."""
    for j, kw in p.items():
        grid.set_party_power(j, kw)
    if grid.home:
        grid.home_kw = {n: p.get("HOME", 0.0) * w for n, w in grid.home_weights.items()}


def voltage_ceiling(cfg, rng, bg_levels, n_anchor, n_samples, p_bar=None):
    """Highest hub voltage the regulated feeder reaches anywhere in the operating envelope:
    background in bg_levels, hub powers in [0, P_bar] including the all-hubs-at-zero corner,
    regulators re-settling from settled anchor taps. Overvoltage is driven by load loss, which no
    contract parameter controls (P's lower bound is always 0), so this bound is a property of
    the feeder, checked once."""
    from seams.domains.grid.dss_env import GridAdapter

    grid = GridAdapter(cfg)
    hubs, parties = list(grid.hubs), grid.parties()
    p_bar = _p_bar(cfg, p_bar)
    top = {h: 0.0 for h in hubs}
    for bg in bg_levels:
        grid.set_background(bg)
        for _ in range(n_anchor):
            _set_all(grid, {j: float(rng.uniform(0, p_bar[j])) for j in parties})
            grid.solve()
            taps0 = {r: grid._tap(r) for r in grid.regulators}
            targets = [{j: 0.0 for j in parties}] + [
                {j: float(rng.uniform(0, p_bar[j])) for j in parties} for _ in range(n_samples)
            ]
            for p in targets:
                for r, t in taps0.items():
                    grid._set_tap(r, t)
                _set_all(grid, p)
                v = grid.solve().v_lv
                for h in hubs:
                    top[h] = max(top[h], v[h])
    return top


def load_limit(cfg, bg_worst, p_bar=None):
    """Bisection per hub on the power that brings its transformer to 100 % loading (other
    parties at their P_bar)."""
    from seams.domains.grid.dss_env import GridAdapter

    grid = GridAdapter(cfg)
    grid.set_background(bg_worst)
    p_bar = _p_bar(cfg, p_bar)
    out = {}
    for j in grid.hubs:
        _set_all(grid, {o: p_bar[o] for o in grid.parties()})
        lo, hi = 0.0, 2.0 * float(grid.hubs[j]["xfmr_kva"])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            grid.set_hub_power(j, mid)
            if grid.solve().xfmr_loading[j] > 100.0:
                hi = mid
            else:
                lo = mid
        out[j] = lo
    return out


def certify_safe(cfg, p_safe, bg_levels, v_margin):
    """Offline certificate for theta_safe: every corner of the power box [0, p_safe] keeps hub
    voltages >= v_min + margin and transformers <= 100 %, with regulators acting, at every
    background level up to scope_bg_max. scope_bg_max is found by bisection; above it the
    feeder breaks Phi_safe even at theta_safe (at high electrification, even with no charging at
    all): a grid-side condition no charging contract can address, reported, not hidden.
    Returns (ok, evidence); ok = the scope covers the whole configured envelope."""
    import itertools
    from seams.domains.grid.dss_env import GridAdapter

    grid = GridAdapter(cfg)
    phi = cfg["grid"]["phi_safe"]
    parties = grid.parties()

    def worst(bg):
        grid.set_background(bg)
        wv, wl = 9.0, 0.0
        for corner in itertools.product((False, True), repeat=len(parties)):
            _set_all(grid, {j: (p_safe[j] if hi else 0.0) for j, hi in zip(parties, corner)})
            st = grid.solve()
            wv, wl = min(wv, min(st.v_lv.values())), max(wl, max(st.xfmr_loading.values()))
        return wv, wl

    passes = lambda w: w[0] >= phi["v_min"] + v_margin and w[1] <= phi["loading_max"]
    top = max(bg_levels)
    results = {float(bg): worst(bg) for bg in bg_levels}
    if passes(results[top]):
        scope = top
    else:
        lo, hi = min(bg_levels), top
        if not passes(results[lo]):
            lo = 0.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if passes(worst(mid)) else (lo, mid)
        scope = lo
    inside = [w for bg, w in results.items() if bg <= scope]
    _set_all(grid, {j: 0.0 for j in parties})
    v_no_ev = worst(top)[0]
    return scope >= top, {
        "min_V": min(w[0] for w in inside) if inside else None,
        "max_loading": max(w[1] for w in inside) if inside else None,
        "bg_levels": list(bg_levels), "v_margin": v_margin, "scope_bg_max": scope,
        "envelope_top": top, "min_V_at_top_no_charging": v_no_ev,
    }


def drift_envelope(fine, contract, hubs, per, K, quantile, parties=None):
    """D[h][k] for k = 1..K from one or more nominal runs (fine and contract logs with Sq_<j>);
    parties: power parties (hubs, plus HOME, whose power is the logged home_kw)."""
    parties = list(parties or hubs)
    devs = {h: {k: [] for k in range(1, K + 1)} for h in hubs}
    ripple = {h: [] for h in hubs}
    for run_id, f in fine.groupby("run_id"):
        c = contract[contract.run_id == run_id]
        V = {h: f[f.hub == h].sort_values("t_fine").V_true.to_numpy() for h in hubs}
        P = {h: f[f.hub == h].sort_values("t_fine").P_pcc.to_numpy() for h in hubs}
        if "HOME" in parties:
            P["HOME"] = f[f.hub == hubs[0]].sort_values("t_fine").home_kw.to_numpy()
        N = len(V[hubs[0]]) // per
        Vbar = {h: V[h][: N * per].reshape(N, per).mean(axis=1) for h in hubs}
        Pbar = {j: P[j][: N * per].reshape(N, per).mean(axis=1) for j in parties}
        S = {h: {j: c[c.hub == h].sort_values("k")[f"Sq_{j}"].to_numpy() for j in parties} for h in hubs}
        for h in hubs:
            ripple[h].append(np.abs(V[h][: N * per].reshape(N, per) - Vbar[h][:, None]).ravel())
            for n in range(N):
                for k in range(1, K + 1):
                    if n + k >= N:
                        break
                    t = slice((n + k) * per, (n + k + 1) * per)
                    pred = Vbar[h][n] + sum(S[h][j][n] * (P[j][t] - Pbar[j][n]) for j in parties)
                    devs[h][k].append(pred - V[h][t])  # one-sided: actual below prediction
    D = {h: {k: max(0.0, float(np.quantile(np.concatenate(devs[h][k]), quantile))) for k in devs[h]} for h in hubs}
    # an envelope must not shrink with lead time
    for h in hubs:
        for k in range(2, K + 1):
            D[h][k] = max(D[h][k], D[h][k - 1])
    cpl = {h: float(np.concatenate(ripple[h]).max()) for h in hubs}
    return D, cpl
