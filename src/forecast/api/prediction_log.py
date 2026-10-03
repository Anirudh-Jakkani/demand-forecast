"""Append-only log of every forecast made (API requests and batch runs).

Monitoring joins this to actual sales once they arrive. `cutoff` is the last day of
data the model had seen, so `date - cutoff` is the forecast horizon in days.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    series_id     TEXT NOT NULL,
    store_id      TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    date          TEXT NOT NULL,
    yhat          REAL NOT NULL,
    cutoff        TEXT
);
CREATE INDEX IF NOT EXISTS ix_predictions_series_date ON predictions (series_id, date);
"""


class PredictionLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            if "cutoff" not in cols:  # logs created before the column existed
                conn.execute("ALTER TABLE predictions ADD COLUMN cutoff TEXT")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def write(self, request_id: str, created_at: str, model_version: str,
              forecasts: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        cutoff_s = pd.Timestamp(cutoff).date().isoformat()
        rows = [
            (request_id, created_at, model_version, r.id, r.store_id, r.item_id,
             r.date.date().isoformat(), float(r.yhat), cutoff_s)
            for r in forecasts.itertuples(index=False)
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )

    def has_request(self, request_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM predictions WHERE request_id = ? LIMIT 1",
                               (request_id,)).fetchone()
        return row is not None

    def read(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM predictions", conn, parse_dates=["date", "cutoff"])
