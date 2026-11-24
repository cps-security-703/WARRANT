import json

import numpy as np

from seams.cosim.config import resolve_path


def calibrate(fine, hubs, sigma_lv, sigma_mv, alpha, persist, margin, est_margin=0.25):
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
