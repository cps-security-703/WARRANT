"""Review experiment E5: how certification scales with the number of hubs (solver only, no
co-simulation). Times the classical path of one renegotiation: the optimizer (Eq. opt), V1 on its
candidate, V2 on the candidate, and the exit-awareness check (V2 on theta_safe).

Instances are synthetic, built from the calibrated IEEE 34 / 2035 / 120 s contract: every hub
copies H1's quotas, caps, drift and linearization bounds; sensitivities are drawn around the
logged IEEE 34 values (own -1.9e-4 to -2.5e-4 pu/kW; cross -0.4e-4 to -0.7e-4, scaled by 3/n so a
feeder with n hubs keeps the 3-hub coupling per hub); anchors at 0.99-1.03 pu and 0-30 kW.

    python -m seams.experiments.scaling [--hubs 3 5 10 20 50 100] [--reps 5] [--results seams/results]

Output: <results>/scaling.csv (one row per instance).
"""
import argparse
import random
import time
from fractions import Fraction
from pathlib import Path

import pandas as pd

from seams.contracts.params import Theta
from seams.contracts.renegotiate import optimize_P_bar
from seams.contracts.verifier import v1_compatibility, v2_refinement

THETA = "seams/config/contracts_static_ieee34_2035_120s.yaml"
SAFE = "seams/config/contracts_safe_ieee34_2035_120s.yaml"
PHI = {"v_min": 0.95, "v_max": 1.05}
HOME = "HOME"


def instance(base, safe, n, rng):
    hubs = [f"H{i + 1}" for i in range(n)]
    parties = hubs + [HOME]
    ref = "H1"
    copy = lambda d: {h: d[ref] for h in hubs}
    theta = base.replace(f"scale-{n}", lam_bar=copy(base.lam_bar), lam_assumed=copy(base.lam_assumed),
                         beta=copy(base.beta), P_bar={**copy(base.P_bar), HOME: base.P_bar[HOME]},
                         P_assumed={**copy(base.P_assumed), HOME: base.P_assumed[HOME]})
    th_safe = safe.replace(f"safe-{n}", lam_bar=copy(safe.lam_bar), lam_assumed=copy(safe.lam_assumed),
                           beta=copy(safe.beta), P_bar={**copy(safe.P_bar), HOME: safe.P_bar[HOME]},
                           P_assumed={**copy(safe.P_assumed), HOME: safe.P_assumed[HOME]})
    e = base.eps
    eps = {"D": copy(e["D"]), "lin": copy(e["lin"]), "P_load_max": copy(e["P_load_max"]),
           "V_ceiling": copy(e["V_ceiling"])}
    S = {h: {j: (-rng.uniform(1.9e-4, 2.5e-4) if j == h else -rng.uniform(0.4e-4, 0.7e-4) * 3 / n)
             for j in hubs} for h in hubs}
    for h in hubs:
        S[h][HOME] = -rng.uniform(0.2e-4, 0.6e-4) * 3 / n
    anchor = {"V": {h: rng.uniform(0.99, 1.03) for h in hubs},
              "P": {j: rng.uniform(0.0, 30.0) for j in parties}, "S": S}
    return theta, th_safe, eps, anchor


def run(n, rep, base, safe, windows, K, weights_one=True):
    rng = random.Random(1000 * n + rep)
    theta, th_safe, eps, anchor = instance(base, safe, n, rng)
    caps = {j: Fraction(theta.P_bar[j]) for j in theta.P_bar}
    floors = {j: Fraction(th_safe.P_bar[j]) for j in theta.P_bar}
    weights = {j: 1 for j in theta.P_bar}
    t0 = time.perf_counter()
    pb, opt_ms = optimize_P_bar(theta, anchor, eps, PHI, K, caps, floors, {}, weights)
    row = {"hubs": n, "rep": rep, "K": K, "opt_ms": opt_ms, "feasible": pb is not None}
    if pb is not None:
        cand = theta.replace("cand", P_bar=pb, P_assumed=pb)
        v1 = v1_compatibility(cand, windows, K)
        v2 = v2_refinement(cand, anchor, eps, PHI, K)
        ex = v2_refinement(th_safe, anchor, eps, PHI, K)
        row.update(v1_ms=v1.solve_ms, v2_ms=v2.solve_ms, exit_ms=ex.solve_ms,
                   certified=v1.ok and v2.ok and ex.ok)
    row["total_ms"] = 1000.0 * (time.perf_counter() - t0)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs", type=int, nargs="+", default=[3, 5, 10, 20, 50, 100])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--K", type=int, default=30)  # one hour at Delta = 120 s
    ap.add_argument("--results", default="seams/results")
    args = ap.parse_args(argv)
    base, safe = Theta.load(THETA), Theta.load(SAFE)
    windows = sorted(base.lam_bar["H1"])
    rows = []
    for n in args.hubs:
        for rep in range(args.reps):
            rows.append(run(n, rep, base, safe, windows, args.K))
            r = rows[-1]
            print(f"n={n:4d} rep={rep} opt={r['opt_ms']:8.0f} ms total={r['total_ms']:8.0f} ms "
                  f"certified={r.get('certified')}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(args.results) / "scaling.csv"
    df.to_csv(out, index=False)
    print(df.groupby("hubs")[["opt_ms", "v1_ms", "v2_ms", "exit_ms", "total_ms"]].median().round(0).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
