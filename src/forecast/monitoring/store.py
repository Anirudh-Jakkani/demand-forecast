"""Persist one row per monitoring run; the dashboard plots these over time."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_runs (
    as_of            TEXT NOT NULL,
    run_at           TEXT NOT NULL,
    model_version    TEXT,
    model_type       TEXT,
    model_data_end   TEXT,
    model_age_days   INTEGER,
    n_points         INTEGER,
    live_wape        REAL,
    live_bias        REAL,
    wape_h1_7        REAL,
    wape_h8_28       REAL,
    cv_wape          REAL,
    wape_ratio       REAL,
    drift_share      REAL,
    drifted_columns  TEXT,
    drift_scores     TEXT,
    decision         TEXT NOT NULL,
    reasons          TEXT,
    report_path      TEXT
);
"""


class MonitoringStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def write(self, row: dict) -> None:
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with sqlite3.connect(self.path) as conn:
            conn.execute(f"INSERT INTO monitoring_runs ({cols}) VALUES ({marks})",
                         list(row.values()))

    def read(self) -> pd.DataFrame:
        with sqlite3.connect(self.path) as conn:
            return pd.read_sql("SELECT * FROM monitoring_runs ORDER BY as_of, run_at", conn,
                               parse_dates=["as_of", "model_data_end"])
