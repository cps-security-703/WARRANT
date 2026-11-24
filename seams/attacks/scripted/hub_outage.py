from seams.attacks.scripted.base import ScriptedAttack


class HubOutage(ScriptedAttack):
    origin = "grid"

    def start(self, t_s):
        m = self.master
        if m.protection is None:
            raise RuntimeError("hub_outage needs grid.protection (the breaker model)")
        k_end = m.clock.seconds_to_fine(self.offset_s)
        m.protection.offline_until[self.target] = max(m.protection.offline_until.get(self.target, -1), k_end)
