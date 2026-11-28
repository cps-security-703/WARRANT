"""Run logger: one Parquet file per record type (fine, contract, session, events, trips, home) per run."""
from pathlib import Path

import pandas as pd
import yaml

TABLES = ("fine", "contract", "session", "events", "trips", "home", "negotiation")


class RunLogger:
    def __init__(self, run_dir, run_id, cfg):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rows = {t: [] for t in TABLES}
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def log(self, table, **fields):
        self.rows[table].append({"run_id": self.run_id, **fields})

    def frames(self):
        return {t: pd.DataFrame(rows) for t, rows in self.rows.items()}

    def close(self):
        for table, df in self.frames().items():
            df.to_parquet(self.run_dir / f"{table}.parquet", index=False)
