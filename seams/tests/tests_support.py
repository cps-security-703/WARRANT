HUBS = ["H1", "H2", "H3"]
PHI = {"v_min": 0.95, "v_max": 1.05}
K_MAX = 60
BASE_THETA = {
    "version": "t", "lam_bar": {h: {1: 17, 6: 29} for h in HUBS}, "P_bar": {h: 105 for h in HUBS},
    "V_lo": 0.95, "V_hi": 1.05, "beta": {h: {1: 0.007, 6: 0.016} for h in HUBS},
    "epsilon": {
        "D": {h: {k: 0.005 + 0.0005 * k for k in range(1, K_MAX + 1)} for h in HUBS},
        "lin": {h: 0.002 for h in HUBS}, "P_load_max": {h: 109.5 for h in HUBS},
        "V_ceiling": {h: 1.045 for h in HUBS},
    },
}
S = {"H1": {"H1": -2.2e-4, "H2": -0.7e-4, "H3": -0.7e-4},
     "H2": {"H1": -0.8e-4, "H2": -2.6e-4, "H3": -1.2e-4},
     "H3": {"H1": -0.8e-4, "H2": -1.2e-4, "H3": -2.6e-4}}
