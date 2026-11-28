"""Availability-aware hub selection (our own TraCI logic, not device.stationfinder).

score(h) = travel_time(veh -> h) + expected_wait(h), with expected_wait driven by the
availability a_h the charging domain publishes. These inputs are exactly what the
navigation attack corrupts, so they are passed in explicitly.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HubView:
    """What routing believes about a hub (possibly spoofed)."""
    availability: float  # a_h: free EVSEs, as published
    queue: int           # vehicles waiting for a space
    n_evse: int


def expected_wait_s(view, mean_dwell_s):
    """a_h = free EVSEs - queue, so a_h <= 0 means 1 - a_h vehicles (including this one) are
    ahead for the next free port; ports free up at n_evse / mean_dwell per second."""
    if view.availability > 0:
        return 0.0
    return (1 - view.availability) * mean_dwell_s / max(view.n_evse, 1)


def select_hub(travel_times, views, mean_dwell_s, excluded=()):
    """travel_times: hub -> seconds (None = unreachable). Returns (hub, score) or (None, None)."""
    best = (None, None)
    for hub in sorted(travel_times):  # sorted for deterministic tie-breaking
        tt = travel_times[hub]
        if tt is None or hub in excluded:
            continue
        score = tt + expected_wait_s(views[hub], mean_dwell_s)
        if best[1] is None or score < best[1]:
            best = (hub, score)
    return best
