"""Interface signals per seam, aggregated from fine-step records into one value per contract
period (the coarse view monitors and verifier see). Period value = mean over the period.

Transport -> charging : lambda_h  attempted arrivals (reached the hub: queued or parked),
                                  counted before any admission shield
Charging  -> grid     : P_req_h   power requested at the PCC limiter input (pre-shield)
                        P_h       power at the PCC, trusted meter
                        (and the same two for the aggregator's home portfolio, party HOME:
                        AMI-metered total and the pre-cap request)
Grid      -> charging : V_rep_h   reported voltage;  d_h  allowance in force
                        V_mv_rep_h  MV voltage at the hub's bus (feeder RTU, an independent meter)
Grid (published each Delta, for the next period's physics check):
                        S[h][j]   dV_h/dP_j,  T[h][r]  dV_h/dtap_r,  taps  mean tap positions
"""
from dataclasses import dataclass, field

import numpy as np

HOME = "HOME"


@dataclass
class PeriodSignals:
    k: int
    lam: dict
    P_req: dict
    P: dict
    V_rep: dict
    d_applied: dict
    taps: tuple       # mean regulator tap positions over the period
    V_mv_rep: dict = field(default_factory=dict)


def aggregate(k, period_records, hubs, voltage_v, home=False):
    """period_records: the period's fine-step log records (all hubs). home: add the aggregator's
    home portfolio (HOME) to the power seam: P from trusted AMI aggregates, P_req pre-cap."""
    by_hub = {h: [r for r in period_records if r["hub"] == h] for h in hubs}
    taps = np.mean([[float(t) for t in r["taps"].split(",")] for r in by_hub[hubs[0]]], axis=0)
    sig = PeriodSignals(
        k=k,
        lam={h: int(sum(r["arrivals"] for r in rs)) for h, rs in by_hub.items()},
        P_req={h: float(np.mean([r["P_req"] for r in rs])) for h, rs in by_hub.items()},
        P={h: float(np.mean([r["P_pcc"] for r in rs])) for h, rs in by_hub.items()},
        V_rep={h: float(np.mean([r["V_reported"] for r in rs])) for h, rs in by_hub.items()},
        d_applied={h: float(np.mean([r["limit_a"] * voltage_v / 1000.0 for r in rs])) for h, rs in by_hub.items()},
        taps=tuple(taps),
        V_mv_rep={h: float(np.mean([r["V_mv_reported"] for r in rs])) for h, rs in by_hub.items()
                  if all(r.get("V_mv_reported") is not None for r in rs)},
    )
    if home:
        rs = by_hub[hubs[0]]  # the home totals are repeated on every hub's record
        sig.P[HOME] = float(np.mean([r["home_kw"] for r in rs]))
        sig.P_req[HOME] = float(np.mean([r["home_req_kw"] for r in rs]))
    return sig
