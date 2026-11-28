"""Compromised AV-fleet dispatcher (transport origin). From the onset a share `fraction` of the
AV fleet (domains/transport/av.py) is compromised. Drain phase (onset to converge_s, optional):
the members get back-to-back long rides and no dispatcher-sent charging, so they run down. From
converge_s they are sent to charge, each AV once, with routing inputs spoofed like the
navigation attack's (target hub empty, the others congested), so they converge on the target:
arrival surge -> queue -> the hub held at its limit. The ground-truth attack window is the
whole compromise, drain included."""
import dataclasses

from seams.attacks.scripted.base import ScriptedAttack


class DispatchAttack(ScriptedAttack):
    origin = "transport"

    def __init__(self, spec, master):
        super().__init__(spec, master)
        self.fraction = float(spec["fraction"])
        self.spoof_queue = int(spec.get("spoof_queue", 30))
        self.converge_s = float(spec.get("converge_s", self.onset_s))
        self.converging = False
        self.members = set()
        master.view_hooks.append(self.spoof)

    def start(self, t_s):
        av = self.master.av_fleet
        if av is None:
            raise RuntimeError("dispatch attack needs an AV fleet (config av_fleet)")
        ids = sorted(av.by_id)
        n = int(round(self.fraction * len(ids)))
        self.members = set(self.master.attack_rng.choice(ids, size=n, replace=False)) if n else set()
        av.drain(sorted(self.members))

    def act(self, t_s):
        if not self.converging and t_s >= self.converge_s:
            self.converging = True
            av = self.master.av_fleet
            av.drain(())
            av.force(sorted(self.members))

    def stop(self, t_s):
        self.master.av_fleet.drain(())

    def spoof(self, views, vid):
        av = self.master.av_fleet
        if not self.converging or not self.active or av is None or av.av_of(vid) not in self.members:
            return views
        out = {}
        for h, v in views.items():
            if h == self.target:
                out[h] = dataclasses.replace(v, availability=v.n_evse, queue=0)
            else:
                out[h] = dataclasses.replace(v, availability=-self.spoof_queue, queue=self.spoof_queue)
        return out
