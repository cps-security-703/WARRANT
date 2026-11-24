import os
from multiprocessing import Pipe

import numpy as np

from seams.attacks.adaptive.surfaces import grey_obs, make_surface, time_features, white_obs
from seams.cosim.config import load_config


def _cursor(m):
    return {t: len(rows) for t, rows in m.logger.rows.items()}


def _since(m, cur, table):
    return m.logger.rows[table][cur[table]:]


def period_stats(m, cur, phi):
    fine = _since(m, cur, "fine")
    dt_h = m.clock.fine_period_s / 3600.0
    hubs = list(m.hubs)
    e = sum(r.get("P_pcc", r["P_sched"]) for r in fine) * dt_h
    e += sum((r.get("home_kw") or 0.0) for r in fine if r["hub"] == hubs[0]) * dt_h
    q = sum(r["queue"] for r in fine) * dt_h
    refused = sum(r["refused"] for r in fine)
    margins = [min(r["margin_v"] / phi["v_scale"], r["margin_loading"] / phi["loading_scale"])
               for r in fine if "margin_v" in r]
    by_step = {}
    for r in fine:
        if "margin_v" in r:
            m_r = min(r["margin_v"] / phi["v_scale"], r["margin_loading"] / phi["loading_scale"])
            by_step[r["t_fine"]] = min(by_step.get(r["t_fine"], np.inf), m_r)
    events = [r["event"] for r in _since(m, cur, "events")]
    rho_neg = set()
    for r in _since(m, cur, "contract"):
        for n, v in r.items():
            if n.startswith("rho_") and v is not None and np.isfinite(v) and v < 0:
                rho_neg.add((r["k"], r["hub"], n))
    return dict(E=e, Q=q, R=refused, margin=min(margins) if margins else np.inf,
                margins=[by_step[k] for k in sorted(by_step)],
                violations=sum(1 for x in margins if x < 0),
                blame=any(ev.startswith("blame:") for ev in events),
                ids=any(ev.startswith("ids_alarm") for ev in events), rho_neg=rho_neg)


class Episode:
    def __init__(self, m, rl, arm, cf):
        self.m, self.rl, self.arm, self.cf = m, rl, arm, cf
        self.mode = rl["mode"]
        self.per = round(m.clock.contract_period_s / m.clock.fine_period_s)
        self.n_step = self.per if self.mode == "within" else 1
        self.n_dec = int(round(arm["duration_s"] / m.clock.fine_period_s)) // self.n_step
        self.t = 0
        self.phi = dict(m.phi, v_scale=rl["reward"]["v_scale_pu"], loading_scale=rl["reward"]["loading_scale"])
        t_next = m.k * m.clock.fine_period_s
        self.attack = make_surface(rl["surface"], arm["target"], m, rl)
        self.attack.onset_s = t_next
        self.attack.offset_s = t_next + arm["duration_s"]
        m.attacks.append(self.attack)
        self.last_fine = []
        self.totals = dict(damage=0.0, E=0.0, Q=0.0, R=0, violations=0, consumed=0.0, blame=False, ids=False, rho_new=0)
        self.detected = False

    def obs(self):
        m = self.m
        o = time_features(m.k * m.clock.fine_period_s) + [self.t / max(self.n_dec, 1)]
        o += grey_obs(self.rl["surface"], self.attack, m, self.last_fine)
        if self.rl["box"] == "white":
            o += white_obs(m, self.attack.target)
        return np.asarray(o, dtype=np.float32)

    def _advance(self, n_fine):
        cur = _cursor(self.m)
        self.m.advance(n_fine)
        st = period_stats(self.m, cur, self.phi)
        self.last_fine = _since(self.m, cur, "fine")
        return st

    def _score(self, st, i0, n_fine):
        r = self.rl["reward"]
        cf = self.cf
        sl = slice(i0, i0 + n_fine)
        dE = float(np.sum(cf["E"][sl])) - st["E"]
        dQ = st["Q"] - float(np.sum(cf["Q"][sl]))
        dR = st["R"] - int(np.sum(cf["R"][sl]))
        damage = dE + r["w_delay_kwh_per_veh_h"] * dQ + r["w_refused_kwh"] * dR
        cf_m = cf["margin"][sl]
        new_rho = {x for x in st["rho_neg"] if (x[1], x[2]) not in cf["rho_neg"].get(x[0], set())}
        cf_viol = int(np.sum(np.asarray(cf_m) < 0))
        new_viol = max(0, st["violations"] - cf_viol)
        tot = self.totals
        tot["damage"] += damage
        tot["E"] += dE
        tot["Q"] += dQ
        tot["R"] += dR
        tot["violations"] += new_viol
        tot["blame"] |= st["blame"]
        tot["ids"] |= st["ids"]
        tot["rho_new"] += len(new_rho)
        if self.mode == "within":
            reward = damage / r["damage_scale_kwh"]
            detected = st["blame"]
        else:
            consumed = float(np.sum(np.clip(np.asarray(cf_m)[:len(st["margins"])] - np.asarray(st["margins"]), 0, None)))
            tot["consumed"] += consumed
            reward = consumed + r["violation_bonus"] * new_viol
            detected = st["blame"] or bool(new_rho)
        if detected:
            reward -= r["detection_penalty"]
        return reward, detected

    def step(self, action):
        self.attack.set_action(action)
        i0 = self.t * self.n_step
        st = self._advance(self.n_step)
        reward, detected = self._score(st, i0, self.n_step)
        self.t += 1
        self.detected |= detected
        # stress test (reward.terminate_on_detection: false): detection neither ends the episode
        # nor stops the attack, so the attacker can probe Phi_safe however visible it gets
        stop = detected and self.rl["reward"].get("terminate_on_detection", True)
        done = stop or self.t >= self.n_dec
        if done and not stop:  # the window is over: stop the attack, run the no-action tail
            tail = int(round(self.rl["tail_s"] / self.m.clock.fine_period_s))
            st = self._advance(tail)
            r2, detected = self._score(st, self.t * self.n_step, tail)
            reward += r2
            self.detected |= detected
        info = dict(self.totals, detected=self.detected) if done else {}
        return self.obs(), float(reward), bool(done), info


def counterfactual(m, rl, max_fine):
    phi = dict(m.phi, v_scale=rl["reward"]["v_scale_pu"], loading_scale=rl["reward"]["loading_scale"])
    out = dict(E=[], Q=[], R=[], margin=[], rho_neg={})
    for _ in range(max_fine):
        cur = _cursor(m)
        m.advance(1)
        st = period_stats(m, cur, phi)
        out["E"].append(st["E"])
        out["Q"].append(st["Q"])
        out["R"].append(st["R"])
        out["margin"].append(st["margin"])
        for k, h, n in st["rho_neg"]:
            out["rho_neg"].setdefault(k, set()).add((h, n))
    return out


def _template_loop(m, conn, rl, max_fine):
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            os._exit(0)
        pid = os.fork()
        if pid == 0:
            try:
                if msg[0] == "counterfactual":
                    conn.send(counterfactual(m, rl, max_fine))
                elif msg[0] == "episode":
                    _episode_child(m, conn, rl, msg[1], msg[2])
                elif msg[0] == "nominal":  # evaluation twin: the rest of the day without attack
                    _finish_day(m, msg[1])
                    conn.send(("written", msg[1]))
            finally:
                os._exit(0)
        os.waitpid(pid, 0)


def _episode_child(m, conn, rl, arm, cf):
    ep = Episode(m, rl, arm, cf)
    conn.send(ep.obs())
    while True:
        cmd = conn.recv()
        if cmd[0] == "step":
            out = ep.step(cmd[1])
            conn.send(out)
            if out[2] and not rl.get("continue_to_end"):
                return
            if out[2]:
                break
        else:
            return
    # evaluation: continue the day to its horizon and write the run's logs
    run_dir = conn.recv()[1]
    _finish_day(m, run_dir)
    conn.send(("written", run_dir))


def _finish_day(m, run_dir):
    from pathlib import Path
    m.run_dir = m.logger.run_dir = Path(run_dir)
    m.logger.run_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    (m.run_dir / "config.yaml").write_text(yaml.safe_dump(m.cfg, sort_keys=False))
    m.advance(m.clock.fine_steps(m.cfg["run"]["horizon_s"]) - m.k)
    m.sumo.close()
    m.finish()


class SnapshotWorker:
    def __init__(self, rl, seed, run_root):
        from seams.cosim.master import Master
        self.rl = rl
        overrides = dict(rl.get("overrides") or {}, **{"run.seed": seed, "attacks": [], "sumo.libsumo": True})
        cfg = load_config(rl["config"], overrides)
        self.m = Master(cfg, f"rl_s{seed}", run_dir=f"{run_root}/s{seed}")
        self.max_fine = int(round((max(rl["durations_s"]) + rl["tail_s"]) / self.m.clock.fine_period_s))
        self.templates = {}
        self.cf = {}
        self.m.start()
        for slot in sorted(rl["onsets_s"]):
            k = int(round(slot / self.m.clock.fine_period_s))
            self.m.advance(k - self.m.k)
            parent, child = Pipe()
            pid = os.fork()
            if pid == 0:
                parent.close()
                _template_loop(self.m, child, rl, self.max_fine)
            child.close()
            self.templates[slot] = (pid, parent)
        self.conn = None

    def counterfactual_for(self, slot):
        if slot not in self.cf:
            _, conn = self.templates[slot]
            conn.send(("counterfactual",))
            self.cf[slot] = conn.recv()
        return self.cf[slot]

    def _abort(self):
        if self.conn is not None:
            self.conn.send(("abort",))
            self.conn = None

    def reset(self, arm):
        self._abort()
        cf = self.counterfactual_for(arm["onset_s"])
        _, self.conn = self.templates[arm["onset_s"]]
        self.conn.send(("episode", arm, cf))
        return self.conn.recv()

    def step(self, action):
        self.conn.send(("step", np.asarray(action, dtype=np.float32)))
        out = self.conn.recv()
        if out[2] and not self.rl.get("continue_to_end"):
            self.conn = None  # the child has exited; the template reads its pipe again
        return out

    def write(self, run_dir):
        self.conn.send(("write", str(run_dir)))
        out = self.conn.recv()
        self.conn = None
        return out

    def nominal(self, slot, run_dir):
        _, conn = self.templates[slot]
        conn.send(("nominal", str(run_dir)))
        return conn.recv()

    def close(self):
        self._abort()
        for pid, conn in self.templates.values():
            try:
                conn.send(("quit",))
                os.waitpid(pid, 0)
            except (OSError, BrokenPipeError, ChildProcessError):
                pass
        self.templates = {}
        self.m.sumo.close()
