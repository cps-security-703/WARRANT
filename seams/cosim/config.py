"""Experiment config: one YAML file fully describes a run; overrides use dotted keys."""
import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path, overrides=None):
    """Load a YAML config. A `base: other.yaml` key (relative to this file) is loaded first
    and this file is deep-merged over it; lists replace rather than merge."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for key, value in (overrides or {}).items():
        set_dotted(cfg, key, value)
    return derive_timing(cfg)


def derive_timing(cfg):
    """Contract timing is configured in seconds; derive the period counts for the configured
    contract period Delta (idempotent, so it can run after any override):
      contracts.windows_s          -> contracts.windows  (always includes the 1-period window)
      verifier.horizon_s           -> verifier.horizon_periods
      contracts.epsilon.drift_horizon_s -> contracts.epsilon.drift_horizon
      renegotiation.exit_quiet_s   -> renegotiation.exit_quiet_periods
      '{delta_s}' in theta file names -> the contract period in seconds
      '{elec}', '{feeder}' in file names -> the electrification level, the feeder
    and applies the scenario presets (scenario.feeder, scenario.electrification) to the grid; a
    preset wins over the keys it sets, so change the preset, not those keys.
    """
    apply_scenario(cfg)
    delta = float(cfg.get("clock", {}).get("contract_period_s", 0) or 0)
    if not delta:
        return cfg
    sc = cfg.get("scenario", {})
    elec, feeder = str(sc.get("electrification", "")), str(sc.get("feeder", ""))
    if str(sc.get("hub_kind", "l2")) != "l2":
        feeder = f"{feeder}-{sc['hub_kind']}"
    fmt = lambda pattern: pattern.format(delta_s=int(delta), elec=elec, feeder=feeder)
    per = lambda s: max(1, round(float(s) / delta))
    c = cfg.get("contracts")
    if c and "windows_s" in c:
        c["windows"] = sorted({1} | {per(w) for w in c["windows_s"]})
        for key in ("theta_file",):
            if key in c and "{" in c.get(f"{key}_pattern", c[key]):
                c[f"{key}_pattern"] = c.get(f"{key}_pattern", c[key])
                c[key] = fmt(c[f"{key}_pattern"])
        e = c.get("epsilon")
        if e and "drift_horizon_s" in e:
            e["drift_horizon"] = per(e["drift_horizon_s"])
    ids = cfg.get("ids")
    if ids and "{" in ids.get("bdd_file_pattern", ids.get("bdd_file", "")):
        ids["bdd_file_pattern"] = ids.get("bdd_file_pattern", ids["bdd_file"])
        ids["bdd_file"] = fmt(ids["bdd_file_pattern"])
    v = cfg.get("verifier")
    if v and "horizon_s" in v:
        v["horizon_periods"] = per(v["horizon_s"])
    r = cfg.get("renegotiation")
    if r:
        if "exit_quiet_s" in r:
            r["exit_quiet_periods"] = per(r["exit_quiet_s"])
        if "{" in r.get("theta_safe_file_pattern", r.get("theta_safe_file", "")):
            r["theta_safe_file_pattern"] = r.get("theta_safe_file_pattern", r["theta_safe_file"])
            r["theta_safe_file"] = fmt(r["theta_safe_file_pattern"])
    return cfg


def apply_scenario(cfg):
    """scenario.feeder names an entry of scenario.feeders: {file, voltage_bases, mv_kv,
    households, hub_buses} -> grid.feeder / voltage_bases / mv_kv and each hub's bus.
    scenario.electrification names a level of scenario.electrification_levels: {peak_mult,
    home_share, peak_quantile_max (optional)} -> grid.background and grid.home_ev.chargers
    (= home_share x the feeder's households)."""
    sc = cfg.get("scenario") or {}
    if "grid" not in cfg:
        return
    kind = str(sc.get("hub_kind", "l2"))
    if kind != "l2":
        kinds = sc.get("hub_kinds", {})
        if kind not in kinds:
            raise ValueError(f"unknown hub kind {kind!r}; known: l2, {sorted(kinds)}")
        preset = copy.deepcopy(kinds[kind])
        while "extends" in preset:  # a kind may extend another and override some of its keys
            preset = deep_merge(copy.deepcopy(kinds[preset.pop("extends")]), preset)
        overrides = preset.pop("hub_overrides", {})
        for hub in cfg.get("hubs", []):
            hub.update(preset.get("hubs", {}))
            hub.update(copy.deepcopy(overrides.get(hub["id"], {})))
        for key, value in preset.items():
            if key != "hubs":
                cfg[key] = deep_merge(cfg.get(key, {}), value)
    households = None
    if sc.get("feeder") is not None:
        feeders = sc.get("feeders", {})
        if str(sc["feeder"]) not in feeders:
            raise ValueError(f"unknown feeder {sc['feeder']!r}; known: {sorted(feeders)}")
        fd = feeders[str(sc["feeder"])]
        g = cfg["grid"]
        g["feeder"], g["voltage_bases"], g["mv_kv"] = fd["file"], fd["voltage_bases"], float(fd["mv_kv"])
        for hub in cfg.get("hubs", []):
            hub["bus"] = str(fd["hub_buses"][hub["id"]])
        households = float(fd["households"])
    level = sc.get("electrification")
    if level is None:
        return
    levels = sc.get("electrification_levels", {})
    if str(level) not in levels:
        raise ValueError(f"unknown electrification level {level!r}; known: {sorted(levels)}")
    lv = levels[str(level)]
    bg = cfg["grid"].setdefault("background", {})
    bg["peak_mult"] = float(lv["peak_mult"])
    bg["peak_quantile_max"] = lv.get("peak_quantile_max")
    if "home_ev" in cfg["grid"] and households is not None:
        cfg["grid"]["home_ev"]["chargers"] = int(float(lv["home_share"]) * households + 0.5)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if isinstance(node, dict) and node.get(p) is None:  # e.g. agents.byzantine: null
            node[p] = {}
        node = node[int(p)] if isinstance(node, list) else node[p]
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def with_overrides(cfg, overrides):
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_dotted(out, key, value)
    return derive_timing(out)


def resolve_path(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
