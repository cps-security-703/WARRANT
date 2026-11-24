import hashlib
import json
import sqlite3
import time
from pathlib import Path


def request_key(model, temperature, messages, schema):
    blob = json.dumps({"model": model, "temperature": temperature, "messages": messages, "schema": schema},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ReplayCache:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, model TEXT, request TEXT, response TEXT, latency_s REAL,
            prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL, created REAL)""")
        self.db.commit()

    def get(self, key):
        row = self.db.execute("SELECT response, latency_s, prompt_tokens, completion_tokens, cost_usd "
                              "FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return dict(zip(("response", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd"), row))

    def put(self, key, model, request, response, latency_s, prompt_tokens, completion_tokens, cost_usd):
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (key, model, json.dumps(request, sort_keys=True), response, latency_s,
                         prompt_tokens, completion_tokens, cost_usd, time.time()))
        self.db.commit()

    def stats(self):
        n, cost, p, c = self.db.execute("SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(prompt_tokens), 0), "
                                        "COALESCE(SUM(completion_tokens), 0) FROM calls").fetchone()
        return {"calls": n, "cost_usd": cost, "prompt_tokens": p, "completion_tokens": c}
