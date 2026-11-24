from seams.attacks.scripted.base import ScriptedAttack
from seams.domains.charging.acn_env import make_scheduler
from seams.domains.charging.malicious_scheduler import MaliciousScheduler

HOME = "HOME"  # target name of the aggregator's home-charger portfolio (domains/charging/home.py)


class AggregatorAttack(ScriptedAttack):
    origin = "charging"

    def __init__(self, spec, master):
        super().__init__(spec, master)
        self.fraction = float(spec["fraction"])
        self.release_s = self.onset_s + float(spec.get("hold_s", 0.0))
        self.phase = None

    def _set(self, phase):
        if phase != self.phase:
            if self.target == HOME:  # the aggregator's residential portfolio on the feeder
                self.master.home.set_mode(phase, self.fraction)
            else:
                fraction = 0.0 if phase == "hold" else self.fraction
                self.master.hubs[self.target].set_scheduler(MaliciousScheduler(fraction))
            self.phase = phase

    def start(self, t_s):
        self._set("hold" if t_s < self.release_s else "release")

    def act(self, t_s):
        self._set("hold" if t_s < self.release_s else "release")

    def stop(self, t_s):
        if self.target == HOME:
            self.master.home.set_mode("honest")
        else:
            self.master.hubs[self.target].set_scheduler(make_scheduler(self.master.cfg["evse"]["scheduler"]))
        self.phase = None
