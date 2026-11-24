import json

from seams.attacks.scripted.base import ScriptedAttack
from seams.cosim.config import resolve_path


class FdiAttack(ScriptedAttack):
    origin = "grid"

    def __init__(self, spec, master):
        super().__init__(spec, master)
        self.bias = float(spec["bias_pu"])
        self.ramp = abs(float(spec["ramp_pu_per_min"]))
        self.mv_gain = None
        if spec.get("two_meter"):
            with open(resolve_path(master.cfg["ids"]["bdd_file"])) as f:
                self.mv_gain = 1.0 / json.load(f)["hubs"][self.target]["b"]

    def act(self, t_s):
        minutes = (t_s - self.onset_s) / 60.0
        magnitude = min(abs(self.bias), self.ramp * minutes) if self.ramp > 0 else abs(self.bias)
        offset = magnitude if self.bias > 0 else -magnitude
        self.master.telemetry.falsification[self.target] = offset
        if self.mv_gain is not None:
            self.master.telemetry.falsification_mv[self.target] = offset * self.mv_gain

    def stop(self, t_s):
        self.master.telemetry.falsification[self.target] = 0.0
        self.master.telemetry.falsification_mv[self.target] = 0.0
