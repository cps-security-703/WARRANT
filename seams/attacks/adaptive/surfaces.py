"""Attack surfaces for the adaptive attackers (RQ3): the scripted injection points with their
strength set at every decision step instead of once.

Each surface maps an action in [-1, 1]^2 to its strength parameters (the SAC layer:
magnitude and ramp/pattern) and exposes the grey-box observation the attacker has at its
injection point; the discrete choices (target, onset, duration) are the DQN layer's.

  fdi         a0 -> target bias in [-bias_max, +bias_max] pu, a1 -> ramp rate; the falsification
              moves toward the target bias at that rate. Sees its meter: true and reported V.
  aggregator  a0 -> share of connected EVs at the maximum rate, a1 > 0 -> hold every EV at zero
              (load-altering build-up). Sees what its scheduler sees: connected EVs, requested
              and scheduled power, and the aggregate limit it is given.
  navigation  a0 -> share of the fleet whose routing inputs are spoofed, a1 -> queue shown at the
              other hubs. Sees the published (public) availability and queues of every hub.
"""
import math

import numpy as np

from seams.attacks.scripted.aggregator import HOME, AggregatorAttack
from seams.attacks.scripted.fdi import FdiAttack
from seams.attacks.scripted.navigation import NavigationAttack
from seams.domains.charging.malicious_scheduler import MaliciousScheduler

FOREVER_S = 1e9


def _unit(a):
    return (float(np.clip(a, -1.0, 1.0)) + 1.0) / 2.0  # [-1, 1] -> [0, 1]


class AdaptiveFdi(FdiAttack):
    def __init__(self, target, master, bias_max, ramp_max):
        super().__init__({"target": target, "onset_s": 0.0, "duration_s": FOREVER_S,
                          "bias_pu": 0.0, "ramp_pu_per_min": 0.0}, master)
        self.bias_max, self.ramp_max = float(bias_max), float(ramp_max)
        self.dt_min = master.clock.fine_period_s / 60.0

    def set_action(self, a):
        self.bias = float(np.clip(a[0], -1, 1)) * self.bias_max
        self.ramp = max(_unit(a[1]) * self.ramp_max, 1e-6)

    def act(self, t_s):
        f = self.master.telemetry.falsification
        cur = f.get(self.target, 0.0)
        step = self.ramp * self.dt_min
        f[self.target] = cur + float(np.clip(self.bias - cur, -step, step))


class AdaptiveAggregator:
    """One compromised aggregator acting on one or several targets (hubs and/or HOME)."""
    origin = "charging"

    def __init__(self, targets, master):
        self.parts = [AggregatorAttack({"target": t, "onset_s": 0.0, "duration_s": FOREVER_S,
                                        "fraction": 0.0}, master) for t in targets]
        self.master = master
        self.target = targets[0] if len(targets) == 1 else "ALL"
        self.onset_s, self.offset_s = 0.0, FOREVER_S
        self.active = False
        self.hold, self.fraction = False, 0.0

    def set_action(self, a):
        self.fraction = _unit(a[0])
        self.hold = float(a[1]) > 0.0

    def step(self, t_s):
        """Same contract as ScriptedAttack.step: called with the next fine period's start."""
        now = self.onset_s <= t_s < self.offset_s
        if now:
            phase = "hold" if self.hold else "release"
            for p in self.parts:
                if (phase, self.fraction) != (p.phase, p.fraction):
                    p.fraction = self.fraction
                    if p.target == HOME:
                        self.master.home.set_mode(phase, self.fraction)
                    else:
                        self.master.hubs[p.target].set_scheduler(
                            MaliciousScheduler(0.0 if self.hold else self.fraction))
                    p.phase = phase
        elif self.active:
            for p in self.parts:
                p.stop(t_s)
        self.active = now


class AdaptiveNavigation(NavigationAttack):
    def __init__(self, target, master, spoof_max):
        super().__init__({"target": target, "onset_s": 0.0, "duration_s": FOREVER_S,
                          "fraction": 0.0, "spoof_queue": 0}, master)
        self.spoof_max = int(spoof_max)
        self.u = {}  # vid -> uniform draw: affected while u < fraction (fraction can change)

    def set_action(self, a):
        self.fraction = _unit(a[0])
        self.spoof_queue = int(round(_unit(a[1]) * self.spoof_max))

    def _is_affected(self, vid):
        if vid not in self.u:
            self.u[vid] = float(self.rng.random())
        return self.u[vid] < self.fraction


def make_surface(surface, target, master, cfg):
    s = cfg["surfaces"][surface]
    if surface == "fdi":
        return AdaptiveFdi(target, master, s["bias_max_pu"], s["ramp_max_pu_per_min"])
    if surface == "aggregator":
        targets = list(master.hubs) + ([HOME] if master.home is not None else []) if target == "ALL" else [target]
        return AdaptiveAggregator(targets, master)
    if surface == "navigation":
        return AdaptiveNavigation(target, master, s["spoof_max"])
    raise ValueError(surface)


# --- observations ------------------------------------------------------------------------
def time_features(t_s):
    x = 2 * math.pi * (t_s % 86400) / 86400
    return [math.sin(x), math.cos(x)]


def grey_obs(surface, attack, master, fine_rows):
    """What the attacker sees at its injection point over the last decision step (fine records
    of all hubs), normalized to O(1)."""
    hubs = list(master.hubs)
    if surface == "fdi":
        r = [x for x in fine_rows if x["hub"] == attack.target] or [{}]
        v_true = np.mean([x.get("V_true", 1.0) for x in r])
        v_rep = np.mean([x.get("V_reported", 1.0) for x in r])
        f = master.telemetry.falsification.get(attack.target, 0.0)
        return [(v_true - 1.0) / 0.05, (v_rep - 1.0) / 0.05, f / 0.05]
    if surface == "aggregator":
        out = []
        for h in hubs:
            r = [x for x in fine_rows if x["hub"] == h] or [{}]
            n = len(master.hubs[h].station_ids)
            mine = attack.target in (h, "ALL")
            out += [np.mean([x.get("occupied_evse", 0) for x in r]) / n if mine else 0.0,
                    np.mean([x.get("P_req", 0.0) for x in r]) / 100.0 if mine else 0.0,
                    np.mean([x.get("P_sched", 0.0) for x in r]) / 100.0 if mine else 0.0,
                    np.mean([x.get("d_h", 0.0) for x in r]) / 100.0 if mine else 0.0]
        if master.home is not None:
            mine = attack.target in (HOME, "ALL")
            r = [x for x in fine_rows if x["hub"] == hubs[0]] or [{}]
            out += [np.mean([x.get("home_req_kw") or 0.0 for x in r]) / 300.0 if mine else 0.0,
                    np.mean([x.get("home_kw") or 0.0 for x in r]) / 300.0 if mine else 0.0]
        return out + [float(attack.hold), attack.fraction]
    if surface == "navigation":
        out = []
        for h in hubs:
            p = master.published[h]
            n = len(master.hubs[h].station_ids)
            out += [p["a"] / n, p["queue"] / n]
        return out + [attack.fraction, attack.spoof_queue / max(attack.spoof_max, 1)]
    raise ValueError(surface)


def white_obs(master, target):
    """White-box extras (upper bound): the contract in force and the target's robustness."""
    th = master.theta
    out = []
    for h in master.hubs:
        out.append(float(th.P_bar[h]) / 100.0 if th is not None else 1.0)
    if HOME in (th.P_bar if th is not None else {}):
        out.append(float(th.P_bar[HOME]) / 300.0)
    last = master.logger.rows["contract"][-(len(master.hubs) + 1):]
    rows = [r for r in last if r["k"] == last[-1]["k"]] if last else []
    status = rows[-1].get("status") if rows else None
    out.append(1.0 if status == "fallback" else 0.0)
    rho = [v for r in rows if target in (r.get("hub"), "ALL") for n, v in r.items()
           if n.startswith("rho_") and v is not None and np.isfinite(v)]
    out.append(float(np.tanh(min(rho) / 5.0)) if rho else 1.0)
    distrust = master.reneg is not None and target in master.reneg.distrusted()
    out.append(1.0 if distrust else 0.0)
    return out
