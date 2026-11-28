"""Feeder FDI (grid origin): falsify the reported voltage V_rep at the target hub, ramping to
bias_pu at ramp_pu_per_min. A negative bias makes derating curtail needlessly (availability
attack); a positive bias hides a real undervoltage.

two_meter: true (review experiment E7) also falsifies the feeder RTU's MV reading at the hub's bus
by bias / b_h, with b_h the LV/MV slope of the grid's bad-data model, so the chi-square residual
V_rep - (a + b V_mv_rep + c P) does not move: the coordinated, BDD-stealthy FDI of Liu et al.
The PCC power meter stays trusted."""
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
