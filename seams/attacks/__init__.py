from seams.attacks.scripted.aggregator import AggregatorAttack
from seams.attacks.scripted.dispatch import DispatchAttack
from seams.attacks.scripted.disturbance import CapacitorTrip, LoadStep, MeterStuck
from seams.attacks.scripted.fdi import FdiAttack
from seams.attacks.scripted.hub_outage import HubOutage
from seams.attacks.scripted.navigation import NavigationAttack

SCRIPTED = {"navigation": NavigationAttack, "dispatch": DispatchAttack, "fdi": FdiAttack, "aggregator": AggregatorAttack, "hub_outage": HubOutage,
            "capacitor_trip": CapacitorTrip, "load_step": LoadStep, "meter_stuck": MeterStuck}


def make_attacks(cfg, master):
    return [SCRIPTED[spec["type"]](spec, master) for spec in cfg.get("attacks") or []]
