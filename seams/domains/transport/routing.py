from dataclasses import dataclass


@dataclass(frozen=True)
class HubView:
    availability: float  # a_h: free EVSEs, as published
    queue: int           # vehicles waiting for a space
    n_evse: int


def expected_wait_s(view, mean_dwell_s):
    if view.availability > 0:
        return 0.0
    return (1 - view.availability) * mean_dwell_s / max(view.n_evse, 1)


def select_hub(travel_times, views, mean_dwell_s, excluded=()):
    best = (None, None)
    for hub in sorted(travel_times):  # sorted for deterministic tie-breaking
        tt = travel_times[hub]
        if tt is None or hub in excluded:
            continue
        score = tt + expected_wait_s(views[hub], mean_dwell_s)
        if best[1] is None or score < best[1]:
            best = (hub, score)
    return best
