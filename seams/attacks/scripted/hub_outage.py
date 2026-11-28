"""Hub lost entirely (RQ4 unforeseen fault, not an attacker): the target hub's service-transformer
breaker opens at onset and stays open for duration_s, as after an equipment failure. It uses the
protection model's offline state, so every domain sees exactly what a trip produces: no power at
the PCC, the charging operator reports the hub down, routing sends vehicles elsewhere. Ground
truth: grid."""
from seams.attacks.scripted.base import ScriptedAttack


class HubOutage(ScriptedAttack):
    origin = "grid"

    def start(self, t_s):
        m = self.master
        if m.protection is None:
            raise RuntimeError("hub_outage needs grid.protection (the breaker model)")
        k_end = m.clock.seconds_to_fine(self.offset_s)
        m.protection.offline_until[self.target] = max(m.protection.offline_until.get(self.target, -1), k_end)
