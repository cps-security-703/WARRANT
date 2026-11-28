"""Online seam monitors (discrete time, one tick = one contract period Delta), robustness and
the attribution rule.

Every formula is `always(linear inequality)`, so the per-period robustness rho_k is the signed
margin of the inequality and the running robustness of `always` is the min over periods so far.
Both are logged; attribution uses rho_k. (rtamt is not needed for this fragment.)

Formulas per hub h, period k (w ranges over the configured windows, in periods):
  G_T_arr_w      lam_bar_h,w - sum of lam_h over the last w periods  transport guarantee
  A_C_arrivals   min over w of the same                              charging assumption (from transport)
  G_C_power      min(P_bar_h - P_req_h, d_h - P_req_h)               charging guarantee
  A_C_voltage    band margin of V_rep_h at k-1                       charging assumption (from grid)
  A_G_power      P_bar_j - P_j for every power party j (hubs, HOME)   grid assumption (from charging)
  (HOME, the aggregator's home portfolio, has one row: G_C_power = P_bar_HOME - P_req_HOME)
  G_G_band       min(V_rep_h - V_lo, V_hi - V_rep_h)                 grid guarantee
  G_G_phys_w     beta_h,w + gamma_h sum_j |dP_j| - |dV_rep_h - sum_j S_hj dP_j - sum_r T_hr dtap_r|
                 over w periods (gamma: the published model's large-signal error per kW moved)
                                                                     grid guarantee (physics consistency)

Causality of the circular charging <-> grid pair: grid physics responds to hub power within the
same period, so grid guarantees at k are conditioned on A_G at k; charging responds to grid
outputs one step later (derating is applied at the next fine step), so its voltage assumption is
taken at k-1. Transport has no assumption (A_T = true).

Attribution rule: blame domain i at (k, h) if rho(A_i) >= 0 and rho(G_i) < 0. A violated
assumption is the supplier's violated guarantee, so it is attributed through the supplier's own
(A, G) check; it only *excuses* i. First blame wins per incident (a contiguous run of periods
with any blame); every blame is logged.
"""
from collections import deque

import numpy as np

INF = float("inf")
HOME = "HOME"
KW_FORMULAS = {"G_C_power", "A_G_power"}
PU_FORMULAS = {"A_C_voltage", "G_G_band"}


class SeamMonitor:
    def __init__(self, theta, hubs, regulators, windows, tol, parties=None):
        self.th = theta.floats()
        self.theta_version = theta.version
        self.hubs = list(hubs)
        # power parties of the charging -> grid seam: the hubs, plus the aggregator's home
        # portfolio (HOME) when it is modeled; HOME has a power guarantee but no voltage
        self.parties = list(parties) if parties else list(hubs)
        self.regs = list(regulators)
        self.windows = sorted(int(w) for w in windows)
        self.tol = tol  # numerical tolerances: rho >= -tol counts as satisfied
        self.hist = deque(maxlen=max(self.windows) + 1)  # (signals, S, T) per period
        self.published = None  # (S, T) published at the end of the previous period
        self.prev_band = None
        self.running = {}      # (formula, hub) -> min rho so far
        self.in_incident = False

    def publish(self, S, T):
        self.published = (S, T)

    def set_theta(self, theta):
        """Monitor against the contract in force (switched atomically at a period boundary)."""
        self.th = theta.floats()
        self.theta_version = theta.version

    # --- robustness ------------------------------------------------------------
    def _physics(self, h, w):
        """Residual of dV over the last w periods against the published linear model, and the
        window's total power movement sum_j |dP_j| (for the large-signal term)."""
        if len(self.hist) <= w:
            return np.nan, 0.0
        seq = list(self.hist)[-(w + 1):]
        dv = seq[-1][0].V_rep[h] - seq[0][0].V_rep[h]
        pred, moved = 0.0, 0.0
        for prev, cur in zip(seq[:-1], seq[1:]):
            S, T = prev[1], prev[2]  # model published at the end of the earlier period
            if S is None:
                return np.nan, 0.0
            moved += sum(abs(cur[0].P[j] - prev[0].P[j]) for j in S[h])
            pred += sum(S[h][j] * (cur[0].P[j] - prev[0].P[j]) for j in S[h])
            pred += sum(T[h][r] * (cur[0].taps[i] - prev[0].taps[i]) for i, r in enumerate(self.regs))
        return dv - pred, moved

    def update(self, sig):
        """Feed one period's signals. Returns (rho rows per hub, blames [(domain, hub)], residuals)."""
        S, T = self.published if self.published else (None, None)
        self.hist.append((sig, S, T))
        th, tol = self.th, self.tol  # tol = {"kw": ..., "pu": ...}
        band = {h: min(sig.V_rep[h] - th["V_lo"], th["V_hi"] - sig.V_rep[h]) for h in self.hubs}
        prev_band = self.prev_band or {h: INF for h in self.hubs}
        A_G = min(th["P_bar"][j] - sig.P[j] for j in self.parties)

        rows, blames, resid = {}, [], {}
        for h in self.hubs:
            r = {}
            for w in self.windows:
                if len(self.hist) >= w:
                    lam_w = sum(s.lam[h] for s, _, _ in list(self.hist)[-w:])
                    r[f"G_T_arr_{w}"] = th["lam_bar"][h][w] - lam_w
                else:
                    r[f"G_T_arr_{w}"] = INF
            r.update({
                "G_C_power": min(th["P_bar"][h] - sig.P_req[h], sig.d_applied[h] - sig.P_req[h]),
                "A_C_voltage": prev_band[h],
                "A_G_power": A_G,
                "G_G_band": band[h],
            })
            for w in self.windows:
                res, moved = self._physics(h, w)
                resid[(h, w)] = res
                tol_w = th["beta"][h][w] + th["gamma"].get(h, 0.0) * moved
                r[f"G_G_phys_{w}"] = tol_w - abs(res) if not np.isnan(res) else INF
            g_t = min(r[f"G_T_arr_{w}"] for w in self.windows)
            r["A_C_arrivals"] = g_t
            # numerical tolerance per formula, in its own units: a margin in (-tol, 0) is 0
            for name in r:
                t = tol["kw"] if name in KW_FORMULAS else tol["pu"] if name in PU_FORMULAS or name.startswith("G_G_phys") else 0.0
                if -t < r[name] < 0:
                    r[name] = 0.0
            dom = {
                "transport": (INF, g_t),
                "charging": (min(r["A_C_arrivals"], r["A_C_voltage"]), r["G_C_power"]),
                "grid": (r["A_G_power"], min(r["G_G_band"], *[r[f"G_G_phys_{w}"] for w in self.windows])),
            }
            for d, (a, g) in dom.items():
                if a >= 0 and g < 0:
                    blames.append((d, h))
                r[f"rho_A_{d}"], r[f"rho_G_{d}"] = a, g
            for name, val in r.items():
                key = (name, h)
                self.running[key] = min(self.running.get(key, INF), val)
            rows[h] = r
        self.prev_band = band
        if HOME in self.parties:
            # the home portfolio: only the charging guarantee (pre-cap request within P_bar)
            # and the grid's assumption on it; no arrivals, no voltage, no physics of its own
            g = th["P_bar"][HOME] - sig.P_req[HOME]
            g = 0.0 if -tol["kw"] < g < 0 else g
            r = {"G_C_power": g, "A_G_power": A_G,
                 "rho_A_charging": INF, "rho_G_charging": g, "rho_A_grid": A_G, "rho_G_grid": INF,
                 "rho_A_transport": INF, "rho_G_transport": INF}
            if g < 0:
                blames.append(("charging", HOME))
            for name, val in r.items():
                self.running[(name, HOME)] = min(self.running.get((name, HOME), INF), val)
            rows[HOME] = r

        # all blames of an incident's first period; attribution is scored on this whole set
        first = list(blames) if blames and not self.in_incident else None
        self.in_incident = bool(blames)
        return rows, blames, first, resid
