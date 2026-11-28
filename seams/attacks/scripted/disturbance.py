"""Accidental grid disturbances (review experiment E3): honest events, no attacker. They test
whether attribution blames the grid for a real change it did not hide, and what that costs. The
ground-truth origin is the grid (its equipment or meter), so the metrics score them like a grid
fault; whether a blame is "false" is a matter of interpretation, discussed with the results.

  capacitor_trip  a feeder capacitor bank (spec target: its OpenDSS name, e.g. C848) is switched
                  out for duration_s: a real voltage drop the published model does not explain.
  load_step       a constant-power load of kw kW (pf 0.95) switches on at an MV bus for
                  duration_s: a large customer the background profile does not contain.
  meter_stuck     the hub's LV voltage meter freezes at its last reading for duration_s
                  (sensor failure): the reported value stops following the true voltage.
"""
import opendssdirect as dss

from seams.attacks.scripted.base import ScriptedAttack


class CapacitorTrip(ScriptedAttack):
    origin = "grid"

    def start(self, t_s):
        dss.Text.Command(f"Capacitor.{self.target}.enabled=false")

    def stop(self, t_s):
        dss.Text.Command(f"Capacitor.{self.target}.enabled=true")


class LoadStep(ScriptedAttack):
    origin = "grid"

    def __init__(self, spec, master):
        super().__init__(spec, master)
        self.kw = float(spec["kw"])
        self.bus = str(spec["bus"])
        self.name = f"disturbance_{self.target}"
        kv = float(master.cfg["grid"]["mv_kv"])
        dss.Text.Command(f"New Load.{self.name} bus1={self.bus} phases=3 conn=wye model=1 kV={kv} kW=0 pf=0.95")

    def start(self, t_s):
        dss.Loads.Name(self.name)
        dss.Loads.kW(self.kw)
        dss.Loads.kvar(self.kw * 0.3287)  # pf 0.95

    def stop(self, t_s):
        dss.Loads.Name(self.name)
        dss.Loads.kW(0.0)
        dss.Loads.kvar(0.0)


class MeterStuck(ScriptedAttack):
    origin = "grid"

    def start(self, t_s):
        self.master.telemetry.stuck[self.target] = None  # frozen at the next report

    def stop(self, t_s):
        self.master.telemetry.stuck.pop(self.target, None)
