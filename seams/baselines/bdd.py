"""Chi-square bad-data detection (BDD), the grid-side detector a utility actually runs with
state estimation (Monticelli; Liu, Ning & Reiter, CCS 2009 for the stealthy-FDI limit).
Detection only, like the IDS: an alarm triggers no containment.

Measurements per hub h: the LV voltage reported at the hub (V_rep_h, the FDI target), the MV
voltage reported by the feeder RTU at the hub's bus (V_mv_rep_h, an independent meter), and the
trusted PCC power P_h. The service transformer ties them together; the estimator's model is the
linear fit  V_lv ~ a_h + b_h V_mv + c_h P_h  on nominal calibration days (true values), with
model error sigma_m,h. Per step:
    r_h = V_rep_h - (a_h + b_h V_mv_rep_h + c_h P_h)
    J   = sum_h r_h^2 / (sigma_m,h^2 + sigma_lv^2 + b_h^2 sigma_mv^2)   ~ chi^2 with n_hubs dof
alarm while J > chi^2_{n, 1-alpha} for `persist` consecutive steps. A falsification of V_rep
alone breaks the LV/MV consistency and is caught; one applied consistently to both meters (the
stealthy FDI of Liu et al.) is not, which RQ3 uses.
"""
import json

import numpy as np

from seams.cosim.config import resolve_path


def calibrate(fine, hubs, sigma_lv, sigma_mv, alpha, persist, margin, est_margin=0.25):
    """Fit on true values, then set the alarm threshold: the chi^2 quantile, raised if needed to
    margin x the largest J that persisted `persist` steps on the nominal calibration days
    (reported, noisy values), so the detector raises no alarm there (as the contracts).
    Also records est_err_max_h = (1 + est_margin) x the largest |V_rep_h - estimate| per fine step
    on those days: the error bound of the MV-based estimate that distrust mode uses
    (seams.contracts.renegotiate), calibrated like beta."""
    from scipy.stats import chi2
    params = fit(fine, hubs, sigma_lv, sigma_mv)
    J = None
    for h in hubs:
        f = fine[fine.hub == h].sort_values(["run_id", "t_fine"])
        p = params[h]
        r = f.V_reported - (p["a"] + p["b"] * f.V_mv_reported + p["c"] * f.P_pcc)
        p["est_err_max"] = (1.0 + est_margin) * float(r.abs().max())
        z = (r / p["sigma"]) ** 2
        J = z.to_numpy() if J is None else J + z.to_numpy()
    runs = fine[fine.hub == hubs[0]].sort_values(["run_id", "t_fine"]).run_id.to_numpy()
    # the J level held for `persist` consecutive steps (min over each window), within a run
    held = [min(J[i - persist + 1:i + 1]) for i in range(persist - 1, len(J))
            if runs[i] == runs[i - persist + 1]]
    nominal_max = float(max(held))
    threshold = max(float(chi2.ppf(1.0 - alpha, df=len(hubs))), margin * nominal_max)
    return {"hubs": params, "threshold": threshold, "nominal_held_J_max": nominal_max,
            "alpha": alpha, "persist": persist, "margin": margin, "est_margin": est_margin}


def fit(fine, hubs, sigma_lv, sigma_mv):
    """Model parameters per hub from nominal fine logs (true V_true, V_mv, P_pcc)."""
    params = {}
    for h in hubs:
        f = fine[fine.hub == h]
        X = np.column_stack([np.ones(len(f)), f.V_mv.to_numpy(), f.P_pcc.to_numpy()])
        y = f.V_true.to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sigma_m = float(np.std(y - X @ coef))
        a, b, c = (float(x) for x in coef)
        params[h] = {"a": a, "b": b, "c": c, "sigma_model": sigma_m,
                     "sigma": float(np.sqrt(sigma_m ** 2 + sigma_lv ** 2 + (b * sigma_mv) ** 2))}
    return params


class BadDataDetector:
    name = "chi2_bdd"

    def __init__(self, cfg):
        c = cfg["ids"]
        with open(resolve_path(c["bdd_file"])) as f:
            model = json.load(f)
        self.params = model["hubs"]
        self.threshold = float(model["threshold"])
        self.persist = int(model["persist"])
        self.run = 0

    def observe(self, t_s, ctx):
        """Returns hub -> (alarm, normalized residual^2); the chi^2 test is over all hubs, so an
        alarm is raised at the hub with the largest contribution."""
        z = {}
        for h, x in ctx.items():
            p = self.params[h]
            r = x["v_rep"] - (p["a"] + p["b"] * x["v_mv_rep"] + p["c"] * x["p"])
            z[h] = (r / p["sigma"]) ** 2
        J = float(sum(z.values()))
        self.run = self.run + 1 if J > self.threshold else 0
        alarm = self.run >= self.persist
        top = max(z, key=z.get)
        return {h: (alarm and h == top, z[h]) for h in ctx}


class AlarmResponse:
    """Detect-and-respond baseline (evaluation only): protection only, plus a local response to
    the BDD. While the BDD alarms at hub h, derating there reads the MV-based estimate
    a_h + b_h V_mv_rep_h + c_h P_h (the same estimate WARRANT's distrust mode uses), and goes back
    to V_rep_h after exit_quiet_s in which the two agree within est_err_max_h (WARRANT's exit rule).
    mode "always": derating never reads V_rep and always uses the estimate.

    This assumes the detector itself is trusted. In the threat model it is not: it belongs to the
    grid operator, which a telemetry attacker controls, so the baseline is optimistic."""

    def __init__(self, cfg, hubs, fine_period_s):
        r = cfg["ids_response"]
        with open(resolve_path(cfg["ids"]["bdd_file"])) as f:
            self.params = json.load(f)["hubs"]
        self.mode = r.get("mode", "alarm")
        if self.mode not in ("alarm", "always"):
            raise ValueError(f"ids_response.mode: {self.mode!r}")
        self.detector = r.get("detector", BadDataDetector.name)
        self.quiet_steps = int(round(float(r.get("exit_quiet_s", 1800)) / fine_period_s))
        self.consistent = {}  # hub in response -> consecutive consistent steps
        self.hubs = list(hubs)

    def active(self):
        return self.hubs if self.mode == "always" else list(self.consistent)

    def estimate(self, h, p, v_mv):
        q = self.params[h]
        return q["a"] + q["b"] * v_mv[h] + q["c"] * p[h]

    def update(self, alarms, v_rep, v_mv, p):
        """After each fine step's detection; returns [(event, hub)] for the log."""
        events = []
        if self.mode == "always":
            return events
        for h in self.hubs:
            if alarms.get(h) and h not in self.consistent:
                self.consistent[h] = 0
                events.append(("ids_response_enter", h))
        for h in list(self.consistent):
            ok = abs(v_rep[h] - self.estimate(h, p, v_mv)) <= self.params[h]["est_err_max"]
            self.consistent[h] = self.consistent[h] + 1 if ok and not alarms.get(h) else 0
            if self.consistent[h] >= self.quiet_steps:
                del self.consistent[h]
                events.append(("ids_response_exit", h))
        return events


class CompositeIDS:
    """Several detectors side by side: the hub alarms if any does. per_detector holds the
    last alarms per detector name, for detector-level metrics."""

    def __init__(self, detectors):
        self.detectors = detectors
        self.per_detector = {}

    def observe(self, t_s, ctx):
        out = {h: (False, 0.0) for h in ctx}
        for d in self.detectors:
            res = d.observe(t_s, ctx)
            self.per_detector[d.name] = {h: a for h, (a, _) in res.items()}
            for h, (a, s) in res.items():
                out[h] = (out[h][0] or a, max(out[h][1], s))
        return out
