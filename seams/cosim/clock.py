"""Two clocks: fine period delta drives physics, contract period Delta drives monitoring."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    sumo_step_s: float
    fine_period_s: float
    contract_period_s: float

    def __post_init__(self):
        for name, num, den in [
            ("fine_period_s / sumo_step_s", self.fine_period_s, self.sumo_step_s),
            ("contract_period_s / fine_period_s", self.contract_period_s, self.fine_period_s),
        ]:
            ratio = num / den
            if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
                raise ValueError(f"{name} must be a positive integer, got {ratio}")

    @classmethod
    def from_config(cls, cfg):
        c = cfg["clock"]
        return cls(float(c["sumo_step_s"]), float(c["fine_period_s"]), float(c["contract_period_s"]))

    @property
    def sumo_steps_per_fine(self):
        return round(self.fine_period_s / self.sumo_step_s)

    @property
    def fine_per_contract(self):
        return round(self.contract_period_s / self.fine_period_s)

    @property
    def acn_period_min(self):
        return self.fine_period_s / 60.0

    def fine_steps(self, horizon_s):
        return int(horizon_s // self.fine_period_s)

    def seconds_to_fine(self, t_s):
        """Fine period index containing simulation time t_s."""
        return int(t_s // self.fine_period_s)

    def duration_to_fine(self, d_s):
        """Number of whole fine periods in a duration (rounded down, at least 1)."""
        return max(1, int(d_s // self.fine_period_s))

    def is_contract_boundary(self, k):
        """True after fine step k closes a contract period."""
        return (k + 1) % self.fine_per_contract == 0

    def contract_index(self, k):
        return k // self.fine_per_contract
