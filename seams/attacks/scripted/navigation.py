import dataclasses

from seams.attacks.scripted.base import ScriptedAttack


class NavigationAttack(ScriptedAttack):
    origin = "transport"

    def __init__(self, spec, master):
        super().__init__(spec, master)
        self.fraction = float(spec["fraction"])
        self.spoof_queue = int(spec["spoof_queue"])
        self.rng = master.attack_rng
        self.affected = {}  # vid -> bool, drawn once per vehicle when first seen
        master.view_hooks.append(self.spoof)

    def _is_affected(self, vid):
        if vid not in self.affected:
            self.affected[vid] = bool(self.rng.random() < self.fraction)
        return self.affected[vid]

    def spoof(self, views, vid):
        if not self.active or vid is None or not self._is_affected(vid):
            return views
        out = {}
        for h, v in views.items():
            if h == self.target:
                out[h] = dataclasses.replace(v, availability=v.n_evse, queue=0)
            else:
                out[h] = dataclasses.replace(v, availability=-self.spoof_queue, queue=self.spoof_queue)
        return out
