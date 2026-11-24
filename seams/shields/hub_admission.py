from collections import deque


class HubAdmission:
    def __init__(self, hubs, windows):
        self.windows = sorted(int(w) for w in windows)
        self.lam_bar = None  # hub -> {w: quota}
        self.history = {h: deque([0] * (max(self.windows) - 1), maxlen=max(self.windows) - 1) for h in hubs}
        self.current = {h: 0 for h in hubs}

    def set_quota(self, lam_bar):
        self.lam_bar = {h: {int(w): float(v) for w, v in ws.items()} for h, ws in lam_bar.items()}

    def request(self, hub):
        if self.lam_bar is None:
            return True
        past = list(self.history[hub])
        for w in self.windows:
            used = self.current[hub] + sum(past[len(past) - (w - 1):]) if w > 1 else self.current[hub]
            if used + 1 > self.lam_bar[hub][w]:
                return False
        self.current[hub] += 1
        return True

    def roll(self):
        for h in self.current:
            self.history[h].append(self.current[h])
            self.current[h] = 0
