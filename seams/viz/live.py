import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba

from seams.cosim.config import PROJECT_ROOT
from seams.viz.stream import CONDITIONS, QUEUE0, SCENARIOS, SEEK0, run_stream

HUB_COLOR = {"H1": "#1f77b4", "H2": "#ff7f0e", "H3": "#2ca02c"}
IDLE_COLOR = "#8a94a6"  # plugged but not drawing power (finished, or waiting for the scheduler)
HUB_NAME = {"H1": "H1 workplace", "H2": "H2 public", "H3": "H3 retail"}
ATTACK_TEXT = {
    "fdi": "False data injection: H1's reported feeder voltage is biased low",
    "fdi_morning": "False data injection at the morning peak: H1's reported voltage is biased low",
    "navigation": "Navigation spoofing: routing apps see H1 empty and the other hubs congested",
    "aggregator": "Compromised aggregator: holds every charger at zero, then releases all at full rate",
    "navigation_redirect": "Navigation app compromise: 'free chargers at H1' pushed to drivers heading to other hubs and to passing EVs",
    "normal": "Attack-free day",
}
MAJOR = ("highway.motorway", "highway.trunk", "highway.primary", "highway.secondary", "highway.tertiary")
QUIET = ("plugin", "release", "V1:", "background_day")


def hhmm(t_s):
    t = int(t_s) // 60
    return f"{t // 60:02d}:{t % 60:02d}"


def parse_time(s):
    h, _, m = s.partition(":")
    return 3600.0 * float(h) + 60.0 * float(m or 0)


def describe(event, hub):
    if event.startswith(QUIET):
        return None
    at = " @" if hub and hub not in ("None",) else ""
    if event.startswith("attack_onset"):
        return f"ATTACK starts ({event.split(':', 1)[1]}){at}"
    if event.startswith("attack_offset"):
        return f"attack ends{at}"
    if event.startswith("blame:"):
        return f"WARRANT blames {event.split(':', 1)[1].upper()}{at}"
    if event.startswith("incident:"):
        return f"{event.split(':', 1)[1]} breaks contract{at}"
    if event.startswith("mode_enter:"):
        d = event.split(":", 1)[1]
        return (f"WARRANT distrusts GRID telemetry{at}" if d == "grid"
                else f"WARRANT degrades {d.upper()}{at}")
    if event.startswith("mode_exit:"):
        return f"WARRANT restores {event.split(':', 1)[1].upper()}{at}"
    if event.startswith("theta:"):
        return f"contract -> {event.split(':', 1)[1]}"
    if event == "trip":
        return f"TRANSFORMER TRIP{at}"
    if event.startswith("ids_alarm"):
        return f"IDS alarm{at}" if event == "ids_alarm" else None
    return f"{event}{at}"


class Band:
    def __init__(self, fig, gs_map, gs_ts, header, net_lines, window, n_bands):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        self.h = header
        self.hubs = [hub["id"] for hub in header["hubs"]]
        self.window = window
        small = n_bands > 1
        self.fs = 9 if small else 11

        # --- transport -------------------------------------------------------
        ax = self.ax_map = fig.add_subplot(gs_map[0])
        major, minor, hub_xy = net_lines
        ax.add_collection(LineCollection(minor, colors="0.82", linewidths=0.5, zorder=1))
        ax.add_collection(LineCollection(major, colors="0.55", linewidths=1.1, zorder=1))
        pts = np.array([hub_xy[h] for h in self.hubs])
        lo, hi = pts.min(0) - 1500.0, pts.max(0) + 1500.0
        c, span = (lo + hi) / 2, hi - lo
        pos = ax.get_position()
        aspect = (pos.height * fig.get_figheight()) / (pos.width * fig.get_figwidth())
        w = max(span[0], span[1] / aspect)
        ax.set_xlim(c[0] - w / 2, c[0] + w / 2)
        ax.set_ylim(c[1] - w * aspect / 2, c[1] + w * aspect / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]), ax.set_yticks([])
        self.veh = ax.scatter([], [], s=6, zorder=3)
        self.hub_txt = {}
        for h in self.hubs:
            x, y = hub_xy[h]
            ax.scatter(x, y, s=260 if not small else 180, marker="*", color=HUB_COLOR[h], edgecolor="k", zorder=5)
            dx, dy, ha = {"H1": (10, 6, "left"), "H2": (10, 6, "left"), "H3": (-10, -16, "right")}.get(h, (8, 8, "left"))
            self.hub_txt[h] = ax.annotate("", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=self.fs,
                                          ha=ha,
                                          fontweight="bold", color=HUB_COLOR[h], zorder=6,
                                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        self.map_info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=self.fs - 1,
                                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9), zorder=7)
        ax.text(0.99, 0.01, "TRANSPORT (SUMO)", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=self.fs, fontweight="bold", color="0.3", zorder=7)

        # --- charging hubs ----------------------------------------------------
        ax = self.ax_hub = fig.add_subplot(gs_map[1])
        ax.set_xlim(0, 3), ax.set_ylim(-0.75, 1.15)
        ax.axis("off")
        self.port_kw = header["port_kw"]
        self.tiles, self.bar, self.allow_mk, self.cap_mk, self.hub_info = {}, {}, {}, {}, {}
        cmap = plt.get_cmap("YlGn")
        self.cmap = cmap
        for i, h in enumerate(self.hubs):
            n = header["n_evse"][h]
            cols = 5
            rows = int(np.ceil(n / cols))
            x0, w = i + 0.1, 0.8
            tw, th = w / cols, 0.62 / rows
            rects = []
            for j in range(n):
                r, cidx = divmod(j, cols)
                rect = plt.Rectangle((x0 + cidx * tw + 0.01, 0.98 - (r + 1) * th + 0.01), tw - 0.02, th - 0.02,
                                     fc="white", ec="0.6", lw=0.6)
                ax.add_patch(rect)
                rects.append(rect)
            self.tiles[h] = rects
            ax.text(i + 0.5, 1.05, HUB_NAME[h], ha="center", fontsize=self.fs + 1, fontweight="bold",
                    color=HUB_COLOR[h])
            # power bar: 0 .. nameplate allowance (p_allow_kw)
            p_max = next(hub for hub in header["hubs"] if hub["id"] == h).get("p_allow_kw", 105.0)
            self.p_max = p_max
            ax.add_patch(plt.Rectangle((x0, 0.12), w, 0.12, fc="0.93", ec="0.6", lw=0.6))
            self.bar[h] = plt.Rectangle((x0, 0.12), 0, 0.12, fc=HUB_COLOR[h], alpha=0.85)
            ax.add_patch(self.bar[h])
            self.allow_mk[h] = ax.plot([], [], color="crimson", lw=3, solid_capstyle="butt")[0]
            self.cap_mk[h] = ax.plot([], [], color="purple", lw=3, ls=(0, (1, 0.6)), solid_capstyle="butt")[0]
            self.hub_info[h] = ax.text(i + 0.5, 0.06, "", ha="center", va="top", fontsize=self.fs - 1, family="monospace")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(0.75), ec="0.6", label="charging port"),
                   plt.Rectangle((0, 0), 1, 1, fc=IDLE_COLOR, ec="0.45", label="plugged, not charging"),
                   plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.6", label="free"),
                   plt.Line2D([], [], color="crimson", lw=3, label="grid allowance (derating)")]
        if header["warrant"]:
            handles.append(plt.Line2D([], [], color="purple", lw=3, ls=(0, (1, 0.6)), label="WARRANT certified cap"))
        ax.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=self.fs - 2, frameon=False,
                  bbox_to_anchor=(0.5, -0.06), handlelength=1.4, columnspacing=0.7)
        ax.text(0.99, -0.72, "CHARGING (ACN-Sim)", ha="right", fontsize=self.fs, fontweight="bold", color="0.3",
                transform=ax.transData)

        # --- feeder -----------------------------------------------------------
        from matplotlib.colors import TwoSlopeNorm
        ax = self.ax_grid = fig.add_subplot(gs_map[2])
        xy = feeder_coords(header["feeder"])
        buses = header["buses"]
        self.bus_keep = np.array([b in xy for b in buses])
        bxy = np.array([xy[b] for b in buses if b in xy])
        segs, self.seg_ends = [], []
        bi = {b: i for i, b in enumerate(buses)}
        for a, b, kind, ph in header["edges"]:
            if a in xy and b in xy and a != b:
                segs.append([xy[a], xy[b]])
                self.seg_ends.append((bi[a], bi[b]))
        self.seg_ends = np.array(self.seg_ends)
        self.norm = TwoSlopeNorm(vmin=0.92, vcenter=header["phi"]["v_min"] + 0.02, vmax=1.06)
        self.vmap = plt.get_cmap("RdYlGn")
        self.lines = LineCollection(segs, cmap=self.vmap, norm=self.norm, linewidths=2.4, zorder=1)
        self.lines.set_array(np.ones(len(segs)))
        ax.add_collection(self.lines)
        self.bus_sc = ax.scatter(bxy[:, 0], bxy[:, 1], s=14, c=np.ones(len(bxy)), cmap=self.vmap, norm=self.norm,
                                 edgecolor="0.3", linewidths=0.3, zorder=2)
        pad = 0.06 * (bxy.max(0) - bxy.min(0))
        ax.set_xlim(bxy[:, 0].min() - pad[0], bxy[:, 0].max() + pad[0])
        ax.set_ylim(bxy[:, 1].min() - 3 * pad[1], bxy[:, 1].max() + 2 * pad[1])
        ax.axis("off")
        self.grid_txt = {}
        offs = {"H1": (-16, 4), "H2": (-14, 22), "H3": (14, -8)}
        has = {"H1": "right", "H2": "right", "H3": "left"}
        for h, hub in zip(self.hubs, header["hubs"]):
            b = str(hub["bus"]).lower()
            if b not in xy:
                continue
            ax.scatter(*xy[b], s=240 if not small else 170, marker="*", color=HUB_COLOR[h], edgecolor="k", zorder=5)
            self.grid_txt[h] = ax.annotate("", xy[b], xytext=offs.get(h, (0, 14)), textcoords="offset points",
                                           ha=has.get(h, "center"), va="center" if h == "H1" else "baseline",
                                           fontsize=self.fs - 1,
                                           family="monospace", zorder=6,
                                           bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9))
        sub = [b for b in ("sourcebus", "800", "150") if b in xy]
        if sub:
            ax.scatter(*xy[sub[0]], s=110, marker="s", color="k", zorder=4)
        cax = ax.inset_axes([0.04, 0.05, 0.35, 0.035])
        cb = fig.colorbar(self.bus_sc, cax=cax, orientation="horizontal")
        cb.set_ticks([0.92, 0.95, 1.0, 1.05])
        cb.set_label("bus voltage [pu]", fontsize=self.fs - 2)
        cb.ax.tick_params(labelsize=self.fs - 3)
        ax.text(0.01, 0.99, f"GRID (OpenDSS, {header['feeder'].upper().replace('IEEE', 'IEEE ')})",
                transform=ax.transAxes, ha="left", va="top", fontsize=self.fs, fontweight="bold", color="0.3")

        # --- time series ------------------------------------------------------
        t0, t1 = window[0] / 3600, window[1] / 3600
        self.ts_axes = [fig.add_subplot(gs_ts[i]) for i in range(3)]
        ylabels = ("vehicles queued", "hub power [kW]", "hub voltage [pu]")
        for ax, yl in zip(self.ts_axes, ylabels):
            ax.set_xlim(t0, t1)
            ax.grid(True, alpha=0.4)
            ax.set_ylabel(yl, fontsize=self.fs - 1)
            ax.tick_params(labelsize=self.fs - 2)
            step = next(m for m in (15, 30, 60, 120, 180, 240) if (t1 - t0) * 60 / m <= 8) / 60
            ticks = np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step)
            ax.set_xticks(ticks)
            ax.set_xticklabels([hhmm(t * 3600) for t in ticks])
            for on, off in sorted({(a["onset_s"], a["onset_s"] + a["duration_s"]) for a in header["attacks"]}):
                ax.axvspan(on / 3600, off / 3600, color="red", alpha=0.08, lw=0)
            ax.margins(x=0)
        self.ts_axes[0].set_ylim(0, 10)
        self.ts_axes[1].set_ylim(0, 135)
        self.ts_axes[2].set_ylim(0.875, 1.07)
        self.ts_axes[2].axhline(header["phi"]["v_min"], color="crimson", lw=1.2, ls="--")
        self.ts_axes[2].text(t0, header["phi"]["v_min"] - 0.004, " ANSI limit 0.95", color="crimson",
                             fontsize=self.fs - 2, va="top")
        self.series = {"t": []}
        self.ts_lines = {}
        for h in self.hubs:
            col = HUB_COLOR[h]
            self.series.update({(h, k): [] for k in ("queue", "P", "lim", "V", "Vrep")})
            self.ts_lines[(h, "queue")] = self.ts_axes[0].plot([], [], color=col, lw=1.6, label=h)[0]
            self.ts_lines[(h, "P")] = self.ts_axes[1].plot([], [], color=col, lw=1.6, label=f"{h} power")[0]
            self.ts_lines[(h, "lim")] = self.ts_axes[1].plot([], [], color=col, lw=1.0, ls="--")[0]
            self.ts_lines[(h, "V")] = self.ts_axes[2].plot([], [], color=col, lw=1.6, label=f"{h} true")[0]
            self.ts_lines[(h, "Vrep")] = self.ts_axes[2].plot([], [], color=col, lw=1.2, ls=":")[0]
        self.ts_axes[0].legend(fontsize=self.fs - 3, loc="upper left", ncol=3, frameon=False)
        self.ts_axes[1].plot([], [], color="0.3", lw=1.0, ls="--", label="limit in force")
        self.ts_axes[1].legend(fontsize=self.fs - 3, loc="upper left", ncol=4, frameon=False)
        self.ts_axes[2].plot([], [], color="0.3", lw=1.2, ls=":", label="reported")
        self.ts_axes[2].legend(fontsize=self.fs - 3, loc="lower right", ncol=4, frameon=False)
        self.now = [ax.axvline(t0, color="0.2", lw=0.8) for ax in self.ts_axes]
        self.events, self.seen = [], {}
        self.phi_min, self.phi_hubs = 0, set()
        self.max_status = 175 if small else 165
        self.status = None  # set by the dashboard
        self.last = None

    # --- updates ------------------------------------------------------------
    def draw_vehicles(self, snap):
        _, xy, code = snap
        colors = np.empty((len(code), 4))
        colors[:] = (0.45, 0.45, 0.45, 0.55)
        sizes = np.full(len(code), 5.0)
        for i, h in enumerate(self.hubs):
            rgb = HUB_COLOR[h]
            seek, qd = code == SEEK0 + i, code == QUEUE0 + i
            colors[seek] = to_rgba(rgb, 0.95)
            colors[qd] = to_rgba("red", 0.95)
            sizes[seek], sizes[qd] = 16, 26
        self.veh.set_offsets(xy if len(xy) else np.empty((0, 2)))
        self.veh.set_facecolors(colors)
        self.veh.set_sizes(sizes)
        n_seek = int(((code >= SEEK0) & (code < SEEK0 + 3)).sum())
        n_q = sum(s["queue"] for s in self.last["hubs"].values()) if self.last else 0
        self.map_info.set_text(f"{len(code)} vehicles on the road\n"
                               f"{n_seek} driving to a hub   {n_q} queued at hubs")

    def update(self, fr):
        self.last = fr
        phi = self.h["phi"]
        bad = [h for h, s in fr["hubs"].items()
               if (s["V_true"] is not None and not phi["v_min"] <= s["V_true"] <= phi["v_max"])
               or (s["loading"] is not None and s["loading"] > phi["loading_max"])]
        if bad:
            self.phi_min += 1
            self.phi_hubs.update(bad)
        t_h = fr["t_s"] / 3600
        self.series["t"].append(t_h)
        for h in self.hubs:
            s = fr["hubs"][h]
            lim = s["allow"] if s["allow"] is not None else np.nan
            if s["cap"] is not None:
                lim = min(lim, s["cap"])
            for k, v in (("queue", s["queue"]), ("P", s["P"]), ("lim", lim), ("V", s["V_true"]), ("Vrep", s["V_rep"])):
                self.series[(h, k)].append(np.nan if v is None else v)
        merged = {}  # same message at several hubs in one step -> one line
        for _, e, hub in fr["events"]:
            desc = describe(e, hub)
            if desc:
                merged.setdefault(desc, [])
                if " @" in desc and hub not in merged[desc]:
                    merged[desc].append(str(hub))
        for desc, hubs in merged.items():
            desc = desc.replace(" @", " @ " + ", ".join(hubs), 1) if hubs else desc
            if fr["t_s"] - self.seen.get(desc, -1e9) > 900:
                self.events.append((fr["t_s"], desc))
            self.seen[desc] = fr["t_s"]
        if fr["bus_v"] is None:
            return
        t = self.series["t"]
        for key, line in self.ts_lines.items():
            line.set_data(t, self.series[key])
        self.ts_axes[0].set_ylim(0, max(10, self.queue_max() * 1.5))
        for ln in self.now:
            ln.set_xdata([t_h, t_h])

        for h in self.hubs:
            s = fr["hubs"][h]
            busy = s["port_busy"]
            for j, rect in enumerate(self.tiles[h]):
                kw = float(s["port_kw"][j]) if j < len(s["port_kw"]) else 0.0
                if not busy[j]:
                    rect.set_facecolor("white")
                elif kw < 0.05:
                    rect.set_facecolor(IDLE_COLOR)
                else:
                    rect.set_facecolor(self.cmap(0.35 + 0.6 * min(1.0, kw / self.port_kw)))
            i = self.hubs.index(h)
            x0, w = i + 0.1, 0.8
            scale = w / self.p_max
            self.bar[h].set_width(min(w, s["P"] * scale))
            if s["allow"] is not None:
                xa = x0 + min(w, s["allow"] * scale)
                self.allow_mk[h].set_data([xa, xa], [0.09, 0.27])
            if s["cap"] is not None:
                xc = x0 + min(w, s["cap"] * scale)
                self.cap_mk[h].set_data([xc, xc], [0.09, 0.27])
            n_chg = int(sum(1 for j, b in enumerate(busy) if b and j < len(s["port_kw"]) and s["port_kw"][j] >= 0.05))
            parts = [f"{s['occupied']}/{len(busy)} plugged", f"{n_chg} charging, {s['P']:.1f} kW"]
            lim = [f"allow {s['allow']:.0f}" if s["allow"] is not None else None,
                   f"cap {s['cap']:.0f}" if s["cap"] is not None else None]
            parts.append("  ".join(x for x in lim if x) + " kW")
            if s["offline"]:
                parts.append("BREAKER OPEN")
            self.hub_info[h].set_text("\n".join(parts))
            self.hub_info[h].set_color("crimson" if s["offline"] else "0.15")
            atk = " ⚠" if s["attack"] else ""
            self.hub_txt[h].set_text(f"{h}{atk}  {s['occupied']}/{len(busy)}" + (f"  q{s['queue']}" if s["queue"] else ""))

        v = fr["bus_v"]
        vk = v[self.bus_keep]
        self.bus_sc.set_array(np.nan_to_num(vk, nan=1.0))
        if len(self.seg_ends):
            self.lines.set_array(np.nan_to_num(np.fmin(v[self.seg_ends[:, 0]], v[self.seg_ends[:, 1]]), nan=1.0))
        vmin = self.h["phi"]["v_min"]
        for h, txt in self.grid_txt.items():
            s = fr["hubs"][h]
            if s["V_true"] is None:
                continue
            lines = [f"{h} V {s['V_true']:.3f} pu"]
            bad = s["V_true"] < vmin
            if s["V_rep"] is not None and abs(s["V_rep"] - s["V_true"]) > 0.004:
                lines.append(f"reported {s['V_rep']:.3f} !")
            if s["V_der"] is not None and s["V_rep"] is not None and abs(s["V_der"] - s["V_rep"]) > 0.004:
                lines.append(f"WARRANT est. {s['V_der']:.3f}")
            if s["loading"] is not None:
                lines.append(f"xfmr {s['loading']:.0f} %")
                bad |= s["loading"] > self.h["phi"]["loading_max"]
            txt.set_text("\n".join(lines))
            fake = len(lines) > 2 and "reported" in lines[1]
            txt.get_bbox_patch().set_edgecolor("crimson" if bad or fake else "0.6")
            txt.get_bbox_patch().set_linewidth(2.0 if bad or fake else 0.8)

    def queue_max(self):
        return max([1] + [np.nanmax(self.series[(h, "queue")] or [0]) for h in self.hubs])

    def status_text(self):
        fr = self.last
        if fr is None:
            return ""
        parts = []
        atk = sorted({s["attack"] for s in fr["hubs"].values() if s["attack"]})
        if atk:
            parts.append("ATTACK ACTIVE: " + ", ".join(map(str, atk)))
        parts.append(f"GRID SAFETY VIOLATED: {self.phi_min} min at {', '.join(sorted(self.phi_hubs))}"
                     if self.phi_min else "grid safety (ANSI band, xfmr) OK")
        if self.h["warrant"]:
            parts.append("WARRANT: " + ("degraded modes " + ", ".join(fr["modes"]) if fr["modes"] else "nominal contract"))
        recent = [f"[{hhmm(t)}] {d}" for t, d in self.events[-3:]]
        while recent and len("   |   ".join(parts + recent)) > self.max_status:
            recent.pop(0)
        return "   |   ".join(parts + recent)


def feeder_coords(feeder):
    base = PROJECT_ROOT / "seams/scenarios/grid" / feeder
    xy = {}
    for f in (base / "IEEE34_BusXY.csv", base / "BusCoords.dat"):
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.replace(",", " ").split()
                if len(p) >= 3:
                    try:
                        xy[p[0].lower()] = (float(p[1]), float(p[2]))
                    except ValueError:
                        pass
    return xy


def road_lines(net_file, hubs):
    sys.path.insert(0, str(Path(os.environ.get("SUMO_HOME", "~/sumo")).expanduser() / "tools"))
    import sumolib
    net = sumolib.net.readNet(str(net_file))
    major, minor = [], []
    for e in net.getEdges():
        if e.getFunction() == "internal":
            continue
        (major if e.getType() in MAJOR else minor).append(np.asarray(e.getShape()))
    hub_xy = {h["id"]: np.asarray(net.getEdge(h["edge"]).getShape()).mean(axis=0) for h in hubs}
    return major, minor, hub_xy


class Dashboard:
    def __init__(self, headers, window, scenario, seed, electrification, caption=None):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        n = len(headers)
        self.fig = fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("white")
        net = road_lines(headers[0]["net"], headers[0]["hubs"])
        heights = [0.35 if n > 1 else 0.25]
        for _ in range(n):
            heights += [0.32, 3.0 if n == 1 else 2.25, 1.1 if n == 1 else 0.85]
        outer = GridSpec(len(heights), 1, figure=fig, height_ratios=heights, hspace=0.18 if n == 1 else 0.22,
                         left=0.035, right=0.99, top=0.985, bottom=0.04 if n == 1 else 0.03)
        hax = fig.add_subplot(outer[0])
        hax.axis("off")
        self.clock = hax.text(0.0, 0.5, "", fontsize=24, fontweight="bold", va="center", family="monospace")
        hax.text(0.11, 0.72 if caption else 0.5, f"{ATTACK_TEXT.get(scenario, scenario)}   ·   seed {seed}   ·   "
                 f"{headers[0]['feeder'].upper()} feeder, {electrification} electrification",
                 fontsize=14, va="center")
        if caption:
            hax.text(0.11, 0.1, caption, fontsize=11.5, va="center", color="0.3", style="italic")
        self.banner = hax.text(1.0, 0.5, "", fontsize=13, va="center", ha="right", color="0.3")
        self.bands = []
        for i, h in enumerate(headers):
            r = 1 + 3 * i
            lax = fig.add_subplot(outer[r])
            lax.axis("off")
            warrant = h["condition"] == "warrant"
            label = lax.text(0.0, 0.5, h["label"], fontsize=15 if n == 1 else 13, fontweight="bold", va="center",
                     color="white", bbox=dict(boxstyle="round,pad=0.3", fc="#2a7f3f" if warrant else "#a33",
                                              ec="none"))
            status = lax.annotate("", xy=(1, 0.5), xycoords=label, xytext=(18, 0), textcoords="offset points",
                                  fontsize=11 if n == 1 else 10, va="center", color="0.15")
            gm = outer[r + 1].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.04)
            gt = outer[r + 2].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.17)
            b = Band(fig, gm, gt, h, net, window, n)
            b.status = status
            self.bands.append(b)

    def set_clock(self, t_s, note=""):
        self.clock.set_text(hhmm(t_s))
        self.banner.set_text(note)
        for b in self.bands:
            b.status.set_text(b.status_text())
            b.status.set_color("crimson" if b.phi_min else "0.15")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--conditions", help="comma list of " + ",".join(CONDITIONS) +
                    " (default: warrant for normal, protection_only,warrant for attacks)")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--strength", type=float, help="attack strength (bias_pu for FDI, fraction otherwise)")
    ap.add_argument("--scheduler", default="EDF", choices=("EDF", "LLF", "RR"))
    ap.add_argument("--feeder", default="ieee34", choices=("ieee34", "ieee123"))
    ap.add_argument("--electrification", default="2035", choices=("today", "2035", "2045"))
    ap.add_argument("--start", help="first time shown, HH:MM (the day is still simulated from 00:00)")
    ap.add_argument("--end", help="last time shown, HH:MM")
    ap.add_argument("--sub-s", type=float, default=10.0, help="vehicle snapshot interval [SUMO s]")
    ap.add_argument("--save", help="record an MP4 (or .gif) here")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--headless", action="store_true", help="no window (with --save)")
    ap.add_argument("--caption", help="one line under the header, e.g. the setting and how typical the day is")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last frame in the video")
    args = ap.parse_args(argv)

    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from seams.viz.stream import build_config
    conds = (args.conditions.split(",") if args.conditions else
             ["warrant"] if args.scenario == "normal" else ["protection_only", "warrant"])
    probe = build_config({"condition": conds[0], "scenario": args.scenario, "seed": args.seed,
                          "scheduler": args.scheduler, "feeder": args.feeder,
                          "electrification": args.electrification, "strength": args.strength})
    attacks = probe.get("attacks", [])
    if args.start:
        start = parse_time(args.start)
    elif attacks:
        start = max(0.0, min(a["onset_s"] for a in attacks) - 3600)
    else:
        start = 6 * 3600.0
    if args.end:
        end = parse_time(args.end)
    elif attacks:
        end = min(86400.0, max(a["onset_s"] + a["duration_s"] for a in attacks) + 3600)
    else:
        end = 22 * 3600.0

    runs_root = Path("~/seams_runs/viz").expanduser()
    ctx = mp.get_context("spawn")
    procs, queues = [], []
    for c in conds:
        spec = {"condition": c, "scenario": args.scenario, "seed": args.seed, "scheduler": args.scheduler,
                "feeder": args.feeder, "electrification": args.electrification, "strength": args.strength,
                "start_s": start, "end_s": end, "sub_s": args.sub_s,
                "run_dir": str(runs_root / f"{c}_{args.scenario}_{args.feeder}_{args.electrification}_s{args.seed}")}
        q = ctx.Queue(maxsize=40)
        p = ctx.Process(target=run_stream, args=(spec, q), daemon=True)
        p.start()
        procs.append(p)
        queues.append(q)

    def get(q):
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except queue.Empty:
                if not any(p.is_alive() for p in procs):
                    raise SystemExit("simulation process exited without output")
                if not args.headless:
                    plt.pause(0.01)
                continue
            if kind == "error":
                raise SystemExit(payload)
            if kind == "status":
                print(payload, flush=True)
                continue
            return kind, payload

    try:
        headers = []
        for q in queues:
            kind, h = get(q)
            assert kind == "header", kind
            headers.append(h)
        print(f"simulating from 00:00; showing {hhmm(start)}-{hhmm(end)}", flush=True)
        dash = Dashboard(headers, (start, end), args.scenario, args.seed, args.electrification, args.caption)
        fig = dash.fig
        if not args.headless:
            plt.show(block=False)
        writer = None
        if args.save:
            from matplotlib import animation
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            out = Path(args.save).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix == ".gif":
                writer = animation.PillowWriter(fps=args.fps)
            else:
                writer = animation.FFMpegWriter(fps=args.fps, codec="libx264", bitrate=-1,
                                                extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
            writer.setup(fig, str(out), dpi=100)

        def show():
            if writer:
                writer.grab_frame()
            if not args.headless:
                fig.canvas.draw_idle()
                plt.pause(0.001)
                if not plt.fignum_exists(fig.number):
                    raise KeyboardInterrupt

        t_wall, last_ff = time.time(), 0.0
        done = False
        while not done:
            frames = []
            for q in queues:
                kind, fr = get(q)
                if kind == "done":
                    done = True
                    break
                frames.append(fr)
            if done:
                break
            t_s = frames[0]["t_s"]
            for b, fr in zip(dash.bands, frames):
                b.update(fr)
            if len(dash.bands) > 1 and frames[0]["bus_v"] is not None:  # same queue scale in every run
                top = max(10, max(b.queue_max() for b in dash.bands) * 1.5)
                for b in dash.bands:
                    b.ts_axes[0].set_ylim(0, top)
            if frames[0]["bus_v"] is None:  # before the shown window: fast-forward
                if time.time() - last_ff > 0.5:
                    last_ff = time.time()
                    dash.set_clock(t_s, f"fast-forwarding to {hhmm(start)} ...")
                    if not args.headless:
                        fig.canvas.draw_idle()
                        plt.pause(0.001)
                    print(f"  sim {hhmm(t_s)} (fast-forward)", end="\r", flush=True)
                continue
            n_sub = max(len(fr["snaps"]) for fr in frames) or 1
            for i in range(n_sub):
                for b, fr in zip(dash.bands, frames):
                    if fr["snaps"]:
                        b.draw_vehicles(fr["snaps"][min(i, len(fr["snaps"]) - 1)])
                snap_t = frames[0]["snaps"][min(i, len(frames[0]["snaps"]) - 1)][0] if frames[0]["snaps"] else t_s
                dash.set_clock(snap_t)
                show()
            print(f"  sim {hhmm(t_s)}  (wall {time.time() - t_wall:.0f} s)        ", end="\r", flush=True)
        print()
        if writer:
            for _ in range(int(args.hold * args.fps)):
                writer.grab_frame()
            writer.finish()
            print(f"saved {args.save}")
        if not args.headless:
            dash.set_clock(end, "done - close the window to exit")
            fig.canvas.draw_idle()
            plt.show()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    main()
