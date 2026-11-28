"""Scripted attack base: an injection point, a strength, an onset and a duration, and the
ground-truth origin domain used to score attribution."""


class ScriptedAttack:
    origin = None  # "transport" | "charging" | "grid"

    def __init__(self, spec, master):
        self.spec = spec
        self.target = spec["target"]
        self.onset_s = float(spec["onset_s"])
        self.offset_s = self.onset_s + float(spec["duration_s"])
        self.master = master
        self.active = False

    def is_active(self, t_s):
        return self.onset_s <= t_s < self.offset_s

    def step(self, t_s):
        """Called once per fine step with the start time of the *next* fine period."""
        now = self.is_active(t_s)
        if now and not self.active:
            self.start(t_s)
        elif not now and self.active:
            self.stop(t_s)
        self.active = now
        if now:
            self.act(t_s)

    def start(self, t_s):
        pass

    def stop(self, t_s):
        pass

    def act(self, t_s):
        pass
