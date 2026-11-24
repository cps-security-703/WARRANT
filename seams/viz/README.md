# Live three-domain view (`seams.viz.live`)

One window, three sections, driven by a real co-simulation (no replay, no LLM):

| Section | Simulator | Top (spatial) | Bottom (time series) |
|---|---|---|---|
| Transport | SUMO | Pasadena streets; grey = through traffic, hub colour = driving to that hub, red = queued at a hub | vehicles queued per hub |
| Charging | ACN-Sim | 20 port tiles per hub (green = charging, shade = kW; slate grey = plugged but not charging; white = free; text: plugged / charging counts); power bar with the grid's derating allowance (red) and WARRANT's certified cap (purple, dotted) | hub power (solid) and the limit in force (dashed) |
| Grid | OpenDSS | IEEE 34 feeder, buses and lines coloured by voltage; at each hub: true V, the *reported* V when falsified, WARRANT's own estimate when it distrusts telemetry, transformer loading | hub voltage, true (solid) vs reported (dotted), ANSI 0.95 limit |

The header shows the clock and scenario. Each run's strip shows the active attack, WARRANT's state, and the latest events: attack onset, IDS alarm, blame, distrust mode, contract changes and transformer trips. Attack windows are shaded red.

## Commands

From the project root:

```zsh
export PYTHONPATH="$PWD" SUMO_HOME=~/sumo; PY=python
$PY -m seams.viz.live normal                     # attack-free day, WARRANT, window pops up
$PY -m seams.viz.live fdi   --save demofdi.mp4                      # FDI day: protection only (top) vs WARRANT (bottom)
$PY -m seams.viz.live aggregator --save agg.mp4  # watch live and record at the same time
$PY -m seams.viz.live navigation --headless --save seams/results/videos/navigation.mp4   # video only
```

Options:
- **Scenarios:** `normal`, `fdi`, `fdi_morning`, `navigation`, `aggregator` (the RQ1 attacks, at the strongest RQ1 strength).
- **Demo scenario** `navigation_redirect` (not in the paper's campaigns; `seams/config/attacks/navigation_redirect.yaml`): from 15:00, a compromised navigation app redirects cars already heading to H2/H3 to H1, and lures passing EVs there with a free top-up. Seeds 3001/3002 were used to design it; it was then run unchanged on the 10 evaluation seeds. Results:
  - WARRANT blames transport on 9/10 days (median 10 min after onset).
  - H1's peak queue is 18 vs 35 cars on average.
  - The contrast is clear on 7/10 days and absent on seed 2006.
- **Contrast settings used for the videos:** `fdi --electrification today`, and `aggregator --electrification 2045 --seed 2007`.
- **Conditions:** an attack scenario compares `protection_only` with `warrant` by default. Use `--conditions protection_only,static_contracts,warrant` for any 1–3 runs, or `--conditions protection_only,warrant` on `normal`.
- **Window:** `--start HH:MM --end HH:MM`. The default is the attack window with an hour either side, or 06:00–22:00 for `normal`. The day is always simulated from 00:00; the hours before the window are fast-forwarded without drawing.
- **Run setup:** `--seed` (default 2000), `--strength`, `--scheduler EDF|LLF|RR`, `--feeder ieee34|ieee123`, `--electrification today|2035|2045`.
- **Video:**
  - `--save x.mp4` (or `.gif`); `--fps` (default 30).
  - `--sub-s`: a vehicle snapshot every N SUMO seconds (default 10, i.e. 6 frames per simulated minute, 5 simulated minutes per video second at 30 fps). Larger values give a shorter video.
  - `--hold`: seconds to hold the final frame.

`--save` uses the ffmpeg bundled with `imageio-ffmpeg`, so no system ffmpeg is needed.

## How it works

`stream.py` runs each condition in its own process. OpenDSS is a single engine per process, so two runs can't share one. It unrolls `Master.advance()` and adds read-only probes:
- vehicle positions every `--sub-s` seconds;
- every bus voltage, read right after the fine step's own power flow (before any contract-boundary sensitivity probe);
- the logged events;
- the contract in force.

Nothing the probes read feeds back into the simulation, so a viewed run follows the same trajectory as a campaign run with the same seed and config. Run logs go to `~/seams_runs/viz/`. `live.py` draws whatever the runs stream and keeps several runs in lockstep, minute by minute.

Timing on an M-series Mac: about 1–2 min to set up and fast-forward to the afternoon, then about 0.2 s per frame for two runs.
