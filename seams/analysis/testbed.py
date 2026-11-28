"""Testbed and operating-point figures (appendix): the road network, the feeder, and what the grid
and the hubs actually see on an attack-free day.

    python -m seams.analysis.testbed [--results seams/results] [--runs ~/seams_runs] [--pdf]

Inputs: the Pasadena SUMO network and IEEE 34 feeder (with the standard OpenDSS bus coordinates)
under seams/scenarios, the kept fine logs
(seed 2000) of the electrification campaign, and the session/home logs of the rq2 nominal runs
(protection only and contracts at c = 1.0, seeds 2000-2009). Output: <results>/figures/testbed_*.png
and <results>/testbed_energy.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from seams.analysis.figures import new_figure, save
from seams.analysis.make_figures import ci
from seams.cosim.config import PROJECT_ROOT

NET = PROJECT_ROOT / "seams/scenarios/pasadena/net.net.xml"
FEEDER = PROJECT_ROOT / "seams/scenarios/grid/ieee34/ieee34Mod1.dss"
HUB_STYLE = {"H1": ("tab:blue", "H1 workplace"), "H2": ("tab:orange", "H2 public"),
             "H3": ("tab:green", "H3 retail")}
ELEC_STYLE = {"today": ("tab:green", "Today"), "2035": ("tab:blue", "2035"), "2045": ("tab:red", "2045")}
MAJOR = ("highway.motorway", "highway.trunk", "highway.primary", "highway.secondary")
# bus-number placements (offset points, alignment) where the default would collide with a line or label
LABEL_AT = {"808": (0, 5, "center"), "824": (0, 5, "center"), "832": (-5, -7, "right"), "852": (-5, -3, "right"),
            "858": (-5, -9, "right"), "836": (-5, -9, "right")}
V_MIN = 0.95
P_ALLOW = 105.0


def _hubs(run_dir):
    return {h["id"]: h for h in yaml.safe_load(open(run_dir / "config.yaml"))["hubs"]}


def _fine(runs, cond, elec, seed=2000):
    f = pd.read_parquet(runs / "electrification" / f"{cond}_elec{elec}_EDF_nominal_s{seed}" / "fine.parquet")
    return f.assign(t_h=f.t_fine / 60.0)


def road_map(hubs, out, pdf):
    sys.path.insert(0, str(Path("~/sumo/tools").expanduser()))
    import sumolib
    net = sumolib.net.readNet(str(NET))
    fig, ax = new_figure(xlabel="x [km]", ylabel="y [km]")
    for e in net.getEdges():
        xy = np.asarray(e.getShape()) / 1000.0
        major = e.getType() in MAJOR
        ax.plot(xy[:, 0], xy[:, 1], color="0.35" if major else "0.7", lw=1.2 if major else 0.4, zorder=1)
    for h, (color, name) in HUB_STYLE.items():
        x, y = np.asarray(net.convertLonLat2XY(hubs[h]["lon"], hubs[h]["lat"])) / 1000.0
        ax.scatter(x, y, s=180, marker="*", color=color, edgecolor="k", zorder=3, label=name)
    ax.set_aspect("equal")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    return save(fig, out, "testbed_map", pdf)


def _feeder_edges():
    """(bus1, bus2, phases, kind) for every line, regulator and transformer of the feeder."""
    txt = FEEDER.read_text()
    bus = lambda b: b.split(".")[0].lower()
    edges = [(bus(a), bus(b), int(ph), "line") for ph, a, b in
             re.findall(r"New Line\.\S+\s+Phases=(\d)\s+Bus1=(\S+)\s+Bus2=(\S+)", txt, re.I)]
    regs = {(bus(a), bus(b)) for a, b in re.findall(r"buses=\((\S+) (\S+)\)", txt, re.I)}
    edges += [(a, b, 3, "reg") for a, b in regs]
    edges.append(("832", "888", 3, "xfmr"))
    return edges


def feeder(hubs, out, pdf):
    """IEEE 34 one-line diagram on the standard OpenDSS bus coordinates (IEEE34_BusXY.csv)."""
    xy = {r[0].lower(): (float(r[1]), float(r[2])) for r in
          (line.split(",") for line in FEEDER.with_name("IEEE34_BusXY.csv").read_text().split())}
    fig, ax = new_figure()
    for a, b, ph, kind in _feeder_edges():
        (x0, y0), (x1, y1) = xy[a], xy[b]
        if kind == "line":
            ax.plot([x0, x1], [y0, y1], color="0.25" if ph == 3 else "0.6", lw=2.2 if ph == 3 else 1.0,
                    zorder=1, solid_capstyle="round")
        else:
            ax.plot([x0, x1], [y0, y1], color="tab:purple" if kind == "reg" else "tab:brown", lw=4, zorder=2)
    ax.plot([xy["sourcebus"][0], xy["800"][0]], [0, 0], color="0.25", lw=2.2, zorder=1)
    ax.scatter(*zip(*[v for k, v in xy.items() if k != "sourcebus"]), s=12, color="0.25", zorder=3)
    ax.scatter(*xy["sourcebus"], s=150, marker="s", color="k", zorder=4)
    hub_bus = {str(hubs[h]["bus"]): h for h in HUB_STYLE}
    down = {a for a, b, _, _ in _feeder_edges() if xy[b][0] == xy[a][0] and xy[b][1] < xy[a][1]}
    down |= {b for a, b, _, _ in _feeder_edges() if xy[a][0] == xy[b][0] and xy[a][1] < xy[b][1]}
    for name, (x, y) in xy.items():  # bus numbers: below each node, beside it on vertical runs
        if name in ("sourcebus", "814r", "852r") or name in hub_bus:
            continue
        dx, dy, ha = (5, -7, "left") if name in down else (0, -9, "center")
        dx, dy, ha = LABEL_AT.get(name, (dx, dy, ha))
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6, color="0.35")
    for h, (color, label) in HUB_STYLE.items():
        b = str(hubs[h]["bus"])
        ax.scatter(*xy[b], s=260, marker="*", color=color, edgecolor="k", zorder=5)
        dx, dy, ha = {"H1": (0, -22, "center"), "H2": (0, 11, "center"), "H3": (12, -4, "left")}[h]
        ax.annotate(f"{h} ({b})", xy[b], xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=11,
                    fontweight="bold", color=color)
    handles = [plt.Line2D([], [], marker="s", ls="", color="k", ms=8, label="Substation"),
               plt.Line2D([], [], color="0.25", lw=2.2, label="3-phase line"),
               plt.Line2D([], [], color="0.6", lw=1.0, label="1-phase line"),
               plt.Line2D([], [], color="tab:purple", lw=4, label="Regulator"),
               plt.Line2D([], [], color="tab:brown", lw=4, label="24.9/4.16 kV xfmr"),
               plt.Line2D([], [], marker="*", ls="", color="w", markeredgecolor="k", ms=12, label="EV hub")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6,
              framealpha=0.95)
    ax.set_xlim(-450, 6250)
    ax.set_ylim(-2350, 1650)
    ax.axis("off")  # a schematic: the coordinates carry no units
    return save(fig, out, "testbed_feeder", pdf)


def _day_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])


def operating_point(runs, out, pdf):
    made = []
    f = _fine(runs, "contracts", "2035")
    # hub (PCC) voltage on a nominal 2035 day
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="Hub voltage [pu]")
    for h, (color, name) in HUB_STYLE.items():
        g = f[f.hub == h]
        ax.plot(g.t_h, g.V_true, color=color, lw=1.2, label=name)
    ax.axhline(V_MIN, color="k", ls="--", lw=1)
    _day_axis(ax)
    ax.legend(fontsize=11, loc="lower left")
    made += save(fig, out, "testbed_voltage_hubs", pdf)

    # H1 voltage across electrification levels, protection only (the uncontrolled operating point)
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="H1 voltage [pu]")
    for elec, (color, name) in ELEC_STYLE.items():
        g = _fine(runs, "protection_only", elec)
        g = g[g.hub == "H1"]
        ax.plot(g.t_h, g.V_true, color=color, lw=1.2, label=name)
    ax.axhline(V_MIN, color="k", ls="--", lw=1)
    _day_axis(ax)
    ax.legend(fontsize=11, loc="lower left")
    made += save(fig, out, "testbed_voltage_elec", pdf)

    # service-transformer loading per hub
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="Transformer loading [%]")
    for h, (color, name) in HUB_STYLE.items():
        g = f[f.hub == h]
        ax.plot(g.t_h, g.xfmr_loading, color=color, lw=1.2, label=name)
    ax.axhline(100, color="k", ls="--", lw=1)
    _day_axis(ax)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc="upper left")
    made += save(fig, out, "testbed_xfmr_loading", pdf)

    # occupied chargers per hub (20 ports each)
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="Occupied chargers")
    for h, (color, name) in HUB_STYLE.items():
        g = f[f.hub == h]
        ax.plot(g.t_h, g.occupied_evse, color=color, lw=1.2, label=name)
    _day_axis(ax)
    ax.set_ylim(0, 24)
    ax.legend(fontsize=11, loc="upper left", ncol=2, columnspacing=0.6)
    made += save(fig, out, "testbed_occupancy", pdf)

    # hub power delivered vs requested, and the residential EV load on the feeder
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="Charging power [kW]")
    for h, (color, name) in HUB_STYLE.items():
        g = f[f.hub == h]
        ax.plot(g.t_h, g.P_pcc, color=color, lw=1.2, label=name)
    home = f[f.hub == "H1"]
    ax.plot(home.t_h, home.home_kw, color="tab:gray", lw=1.2, label="Homes")
    ax.axhline(P_ALLOW, color="k", ls="--", lw=1, label="Hub limit")
    _day_axis(ax)
    ax.set_ylim(0, 200)
    ax.legend(fontsize=11, loc="upper left", ncol=2, columnspacing=0.6)
    made += save(fig, out, "testbed_power", pdf)

    # H1 on day 2007 at 2035, the evaluation day where WARRANT costs H1 the most: what the certified
    # cap withholds. Period-average power from the contract logs (the baseline runs at its own Delta);
    # the cap logged at contract period k applies from boundary k+1.
    fig, ax = new_figure(xlabel="Time of day [h]", ylabel="H1 power [kW]")
    for cond, tag, color, name in (("protection_only", "protection_only_EDF_nominal_s2007", "tab:gray",
                                    "Prot. only"),
                                   ("contracts", "contracts_c1.0_EDF_nominal_s2007", "tab:blue", "WARRANT")):
        d = runs / "rq2" / tag
        delta = yaml.safe_load(open(d / "config.yaml"))["clock"]["contract_period_s"]
        c = pd.read_parquet(d / "contract.parquet")
        c = c[c.hub == "H1"]
        ax.plot((c.k + 0.5) * delta / 3600.0, c.P_h, color=color, lw=1.2, label=f"{name}: power")
        if cond == "contracts":
            ax.step((c.k + 1) * delta / 3600.0, c.Pbar_H1, where="post", color="k", lw=1, ls="--",
                    label="WARRANT: cap")
    _day_axis(ax)
    ax.set_ylim(0, 160)
    ax.legend(fontsize=13, loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.4, handletextpad=0.4,
              framealpha=0.95, borderaxespad=0.2)
    made += save(fig, out, "testbed_power_h1_2035", pdf)
    return made


def energy(runs, results, out, pdf):
    """Delivered and unserved energy per hub and for homes, 10 attack-free days (IEEE 34, 2035); home
    sessions still plugged at the horizon are left out, as in metrics.availability_cost."""
    rows = []
    for cond, tag in (("protection_only", "protection_only_EDF_nominal"), ("contracts", "contracts_c1.0_EDF_nominal")):
        for seed in range(2000, 2010):
            d = runs / "rq2" / f"{tag}_s{seed}"
            s = pd.read_parquet(d / "session.parquet").groupby("hub")[["requested_kWh", "delivered_kWh"]].sum()
            hm = pd.read_parquet(d / "home.parquet")
            hm = hm[hm.ends_in_horizon][["requested_kWh", "delivered_kWh"]].sum().rename("HOME")
            for site, r in pd.concat([s, hm.to_frame().T]).iterrows():
                rows.append(dict(condition=cond, seed=seed, site=site, requested_kWh=r.requested_kWh,
                                 delivered_kWh=r.delivered_kWh))
    e = pd.DataFrame(rows)
    e.groupby(["condition", "site"])[["requested_kWh", "delivered_kWh"]].mean().round(1) \
        .to_csv(results / "testbed_energy.csv")
    sites = ["H1", "H2", "H3", "HOME"]
    labels = {"H1": "H1", "H2": "H2", "H3": "H3", "HOME": "Homes"}
    bars = (("protection_only", "tab:gray", "Prot. only"), ("contracts", "tab:blue", "WARRANT"))
    fig, ax = new_figure(ylabel="Energy per day [kWh]")
    w = 0.38
    for i, (cond, color, name) in enumerate(bars):
        m = e[e.condition == cond].groupby("site")[["requested_kWh", "delivered_kWh"]].mean().reindex(sites)
        x = np.arange(len(sites)) + (i - 0.5) * w
        ax.bar(x, m.delivered_kWh, w, color=color, label=f"{name}, delivered")
        ax.bar(x, m.requested_kWh - m.delivered_kWh, w, bottom=m.delivered_kWh, color="white", edgecolor=color,
               hatch="///", label=f"{name}, unserved")
    ax.set_xticks(range(len(sites)), [labels[s] for s in sites])
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.3)
    ax.legend(fontsize=11, loc="upper left")
    return save(fig, out, "testbed_energy", pdf), e


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="seams/results")
    ap.add_argument("--runs", default="~/seams_runs")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)
    results, runs = Path(args.results), Path(args.runs).expanduser()
    out, pdf = results / "figures", args.pdf or None
    hubs = _hubs(runs / "electrification" / "contracts_elec2035_EDF_nominal_s2000")
    made = road_map(hubs, out, pdf) + feeder(hubs, out, pdf) + operating_point(runs, out, pdf)
    m, e = energy(runs, results, out, pdf)
    made += m
    print(e.groupby(["condition", "site"])[["requested_kWh", "delivered_kWh"]].mean().round(1))
    print(f"{len(made)} files")


if __name__ == "__main__":
    main()
