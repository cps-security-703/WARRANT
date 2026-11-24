import numpy as np


def _p_bar(cfg, p_bar):
    out = {h["id"]: float(h["p_allow_kw"]) for h in cfg["hubs"]}
    out.update({j: float(v) for j, v in (p_bar or {}).items()})
    return out


def linearization(cfg, rng, bg_levels, n_anchor, n_samples, p_bar=None):
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
    for j, kw in p.items():
        grid.set_party_power(j, kw)
    if grid.home:
        grid.home_kw = {n: p.get("HOME", 0.0) * w for n, w in grid.home_weights.items()}


def voltage_ceiling(cfg, rng, bg_levels, n_anchor, n_samples, p_bar=None):
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
