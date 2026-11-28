"""Baseline configurations. Each is a pure config transform applied before the run.

- protection_only (primary): honest scheduler with infrastructure limits + voltage-aware
  derating + transformer protection (trip at sustained overload) + IDS (detection only).
- no_protection (lower bound only): Uncontrolled scheduler, no derating, no protection.
  Shows the raw cascade; never the main comparison.
- oracle (upper bound on availability): see baselines/oracle.py.
"""
import copy


def apply_baseline(cfg):
    cfg = copy.deepcopy(cfg)
    kind = cfg.get("baseline")
    if kind == "no_protection":
        cfg["evse"]["scheduler"] = "UNCONTROLLED"
        cfg["grid"]["derating"]["enabled"] = False
        cfg["grid"]["protection"]["enabled"] = False
    elif kind == "oracle":
        cfg.setdefault("oracle", {})
        sc = cfg.get("scenario", {})
        # transformer limits (P_load_max) and the home portfolio's range come from the static
        # contract calibrated for the same feeder and electrification level (default Delta)
        name = f"contracts_static_{sc.get('feeder', 'ieee34')}_{sc.get('electrification', '2035')}_120s.yaml"
        cfg["oracle"] = {"theta_file": f"seams/config/{name}", "margin_pu": 0.005,
                         "sensitivity_every_s": 300, **cfg["oracle"]}
    elif kind not in (None, "protection_only"):
        raise ValueError(f"unknown baseline {kind!r}")
    return cfg
