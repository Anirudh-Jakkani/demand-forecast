"""Shared helpers for the Prefect flows."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from forecast.api.prediction_log import PredictionLog
from forecast.config import Config, load_config
from forecast.data.ingest import read_processed


def get_config(config_path: str | None) -> Config:
    return load_config(config_path) if config_path else load_config()


def load_until(cfg: Config, as_of: str | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """All processed data visible on `as_of` (default: everything). In production as_of is
    today; in a replay it lets us pretend we are standing on an earlier day."""
    df = read_processed(cfg.resolve(cfg.data.processed_path))
    as_of_ts = pd.Timestamp(as_of) if as_of else df["date"].max()
    return df[df["date"] <= as_of_ts].reset_index(drop=True), as_of_ts


def batch_forecast_champion(cfg: Config) -> dict:
    """Forecast every series with the current champion and log it for monitoring.
    Idempotent per model version."""
    from forecast.registry import load_champion

    bundle, version = load_champion(cfg)
    request_id = f"batch-v{version.version}"
    pred_log = PredictionLog(cfg.resolve(cfg.serving.prediction_log))
    if pred_log.has_request(request_id):
        return {"version": str(version.version), "rows": 0, "skipped": True}

    ids = bundle.future_frame["id"].astype(str).unique().tolist()
    fc = bundle.forecast(ids, bundle.max_horizon)
    pred_log.write(request_id, dt.datetime.now(dt.UTC).isoformat(), str(version.version),
                   fc, bundle.cutoff)
    return {"version": str(version.version), "rows": len(fc), "skipped": False}
