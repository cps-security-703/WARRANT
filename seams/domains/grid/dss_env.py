from dataclasses import dataclass

import numpy as np
import opendssdirect as dss

from seams.cosim.config import resolve_path

REGULATORS = ("reg1a", "reg1b", "reg1c", "reg2a", "reg2b", "reg2c")  # IEEE 34; see GridAdapter.regulators
HOME = "HOME"  # the aggregator's home-charger portfolio as a power party (domains/charging/home.py)


@dataclass
class GridState:
    v_lv: dict           # hub -> min phase voltage at the hub PCC bus [pu]
    v_mv: dict           # hub -> min phase voltage at the feeder bus [pu]
    xfmr_loading: dict   # hub -> service transformer loading [% of nameplate kVA]
    taps: tuple          # regulator tap positions (integer steps)
    converged: bool


class GridAdapter:
    def __init__(self, cfg):
        g = cfg["grid"]
        self.hubs = {h["id"]: h for h in cfg["hubs"]}
        self.pf = float(g["hub_pf"])
        self.tap_step = float(g["tap_step"])
        dss.Text.Command("Clear")
        dss.Text.Command(f'Compile "{resolve_path(g["feeder"])}"')
        for hid, h in self.hubs.items():
            mv_kv = float(g["mv_kv"])
            dss.Text.Command(
                f"New Transformer.XF_{hid} phases=3 windings=2 buses=({h['bus']}, {hid}_lv) "
                f"conns=(delta, wye) kvs=({mv_kv}, {g['lv_kv']}) kvas=({h['xfmr_kva']}, {h['xfmr_kva']}) "
                f"xhl={g['xfmr_xhl']} %r={g['xfmr_r']}"
            )
            dss.Text.Command(
                f"New Load.EV_{hid} bus1={hid}_lv phases=3 conn=wye model=1 kV={g['lv_kv']} "
                f"kW=0 pf={self.pf} vminpu=0.7"
            )
        # the feeder's voltage regulators, from its own RegControls (IEEE 34: reg1a..reg2c)
        self.regulators = tuple(sorted({self._regcontrol_xfmr(n) for n in dss.RegControls.AllNames()}))
        dss.Text.Command(f'Set VoltageBases="{g["voltage_bases"]}"')
        dss.Text.Command("CalcVoltageBases")
        self.controlmode = "OFF" if g.get("lock_taps") else "STATIC"
        dss.Text.Command(f"Set Controlmode={self.controlmode}")
        # Tight convergence: at OpenDSS's default (1e-4) the result depends on the starting
        # point by ~1e-5 pu, so sensitivity probes would perturb later solves, and 2 kW
        # finite differences (~4e-4 pu) would carry solver noise.
        dss.Text.Command(f"Set Tolerance={g.get('solve_tolerance', 1e-9)}")
        self.background = self._background_loads()
        self.hub_kw = {h: 0.0 for h in self.hubs}
        self.home = {}  # background load element -> its home-charging companion load
        self.home_kw, self.home_weights = {}, {}
        if g.get("home_ev", {}).get("enabled"):
            self._add_home_loads(float(g["home_ev"].get("pf", 0.99)))

    @staticmethod
    def _background_loads():
        base = {}
        for name in dss.Loads.AllNames():
            if not name.startswith(("ev_", "home_")):
                dss.Loads.Name(name)
                base[name] = (dss.Loads.kW(), dss.Loads.kvar())
        return base

    def set_background(self, mult):
        for name, (kw, kvar) in self.background.items():
            dss.Loads.Name(name)
            dss.Loads.kW(kw * mult)
            dss.Loads.kvar(kvar * mult)

    def _add_home_loads(self, pf):
        self.home_pf = pf
        for name in sorted(self.background):
            dss.Loads.Name(name)
            bus, phases = dss.CktElement.BusNames()[0], dss.CktElement.NumPhases()
            conn = "delta" if dss.Loads.IsDelta() else "wye"
            dss.Text.Command(
                f"New Load.home_{name} bus1={bus} phases={phases} conn={conn} model=1 "
                f"kV={dss.Loads.kV()} kW=0 pf={pf} vminpu=0.7"
            )
            self.home[name] = f"home_{name}"
        # default portfolio shares: where chargers are expected (by nominal load); a run
        # replaces them with where its chargers actually are
        self.set_home_weights(self.home_elements())

    def home_elements(self):
        return {name: kw for name, (kw, _) in self.background.items()}

    def set_home_weights(self, weights):
        tot = float(sum(weights.values())) or 1.0
        self.home_weights = {n: w / tot for n, w in weights.items()}
        self.home_kw = {n: 0.0 for n in self.home}

    def set_home_power(self, kw_by_element):
        q = np.tan(np.arccos(self.home_pf)) if self.home else 0.0
        for name, kw in kw_by_element.items():
            dss.Loads.Name(self.home[name])
            dss.Loads.kW(max(0.0, kw))
            dss.Loads.kvar(max(0.0, kw) * q)
            self.home_kw[name] = max(0.0, kw)

    # --- power parties: the hubs, plus the aggregator's home portfolio (HOME) ---------------
    def parties(self):
        return list(self.hubs) + ([HOME] if self.home else [])

    def party_power(self, j):
        return sum(self.home_kw.values()) if j == HOME else self.hub_kw[j]

    def set_party_power(self, j, kw):
        if j != HOME:
            return self.set_hub_power(j, kw)
        self._home_loads({n: max(0.0, kw) * w for n, w in self.home_weights.items()})

    def _home_loads(self, kw_by_element):
        q = np.tan(np.arccos(self.home_pf))
        for name, kw in kw_by_element.items():
            dss.Loads.Name(self.home[name])
            dss.Loads.kW(max(0.0, kw))
            dss.Loads.kvar(max(0.0, kw) * q)

    def _restore_home(self):
        self.set_home_power(dict(self.home_kw))

    def set_hub_power(self, hub, kw):
        kw = max(0.0, kw)
        self.hub_kw[hub] = kw
        dss.Loads.Name(f"EV_{hub}")
        dss.Loads.kW(kw)
        dss.Loads.kvar(kw * np.tan(np.arccos(self.pf)))

    def sensitivity(self, dp_kw):
        dss.Text.Command("Set Controlmode=OFF")
        dss.Solution.Solve()
        base = self._v_lv()
        S, T = {h: {} for h in self.hubs}, {h: {} for h in self.hubs}
        for j in self.parties():
            if j == HOME:  # add dp on top of the actual home loads, spread by the shares
                self._home_loads({n: kw + dp_kw * self.home_weights[n] for n, kw in self.home_kw.items()})
            else:
                p0 = self.hub_kw[j]
                self.set_hub_power(j, p0 + dp_kw)
            dss.Solution.Solve()
            v = self._v_lv()
            if j == HOME:
                self._restore_home()
            else:
                self.set_hub_power(j, p0)
            for h in self.hubs:
                S[h][j] = (v[h] - base[h]) / dp_kw
        for r in self.regulators:
            tap0 = self._tap(r)
            self._set_tap(r, tap0 + self.tap_step)
            dss.Solution.Solve()
            v = self._v_lv()
            self._set_tap(r, tap0)
            for h in self.hubs:
                T[h][r] = v[h] - base[h]
        dss.Solution.Solve()
        dss.Text.Command(f"Set Controlmode={self.controlmode}")
        return S, T

    def sensitivity_qss(self, p_max):
        taps0 = {r: self._tap(r) for r in self.regulators}
        Sq = {h: {} for h in self.hubs}
        dss.Text.Command("Set Controlmode=STATIC")
        for j in self.parties():
            p0 = self.hub_kw.get(j)
            v = {}
            for label, p in (("lo", 0.0), ("hi", float(p_max[j]))):
                for r, t in taps0.items():
                    self._set_tap(r, t)
                self.set_party_power(j, p)
                dss.Solution.Solve()
                v[label] = self._v_lv()
            if j == HOME:
                self._restore_home()
            else:
                self.set_hub_power(j, p0)
            for h in self.hubs:
                Sq[h][j] = (v["hi"][h] - v["lo"][h]) / float(p_max[j])
        dss.Text.Command("Set Controlmode=OFF")
        for r, t in taps0.items():
            self._set_tap(r, t)
        dss.Solution.Solve()
        dss.Text.Command(f"Set Controlmode={self.controlmode}")
        return Sq

    @staticmethod
    def _regcontrol_xfmr(name):
        dss.RegControls.Name(name)
        return dss.RegControls.Transformer().lower()

    @staticmethod
    def _tap(r):
        dss.Transformers.Name(r)
        dss.Transformers.Wdg(2)  # the regulated winding
        return dss.Transformers.Tap()

    @staticmethod
    def _set_tap(r, value):
        dss.Transformers.Name(r)
        dss.Transformers.Wdg(2)
        dss.Transformers.Tap(value)

    def _v_lv(self):
        out = {}
        for hid in self.hubs:
            dss.Circuit.SetActiveBus(f"{hid}_lv")
            out[hid] = float(min(dss.Bus.puVmagAngle()[::2][:3]))
        return out

    def solve(self):
        dss.Solution.Solve()
        v_lv, v_mv, loading = self._v_lv(), {}, {}
        for hid, h in self.hubs.items():
            dss.Circuit.SetActiveBus(str(h["bus"]))
            v_mv[hid] = float(min(dss.Bus.puVmagAngle()[::2]))
            dss.Circuit.SetActiveElement(f"Transformer.XF_{hid}")
            n = dss.CktElement.NumConductors()
            p = dss.CktElement.Powers()[: 2 * n]  # terminal 1 (MV side)
            kva = abs(complex(sum(p[0::2]), sum(p[1::2])))
            loading[hid] = 100.0 * kva / float(h["xfmr_kva"])
        taps = tuple(int(round((self._tap(r) - 1.0) / self.tap_step)) for r in self.regulators)
        return GridState(v_lv, v_mv, loading, taps, bool(dss.Solution.Converged()))
