"""Data shaping for the dashboard. Kept free of Streamlit so it can be unit-tested."""

from __future__ import annotations

import json

import pandas as pd

from forecast.api.prediction_log import PredictionLog
from forecast.config import Config
from forecast.data.ingest import read_processed
from forecast.monitoring.performance import join_forecasts_to_actuals
from forecast.monitoring.store import MonitoringStore


def monitoring_runs(cfg: Config) -> pd.DataFrame:
    path = cfg.resolve(cfg.monitoring.db_path)
    if not path.exists():
        return pd.DataFrame()
    runs = MonitoringStore(path).read()
    if runs.empty:
        return runs
    # Re-runs of the same day (e.g. a manual run after the scheduled one): keep the latest.
    runs = runs.sort_values("run_at").drop_duplicates("as_of", keep="last").sort_values("as_of")
    runs["reasons"] = runs["reasons"].map(lambda s: json.loads(s) if s else [])
    runs["drift_scores"] = runs["drift_scores"].map(lambda s: json.loads(s) if s else {})
    runs["wape_limit"] = runs["cv_wape"] * cfg.monitoring.max_wape_ratio
    return runs.reset_index(drop=True)


def drift_long(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (day, column) drift score."""
    rows = [{"as_of": r.as_of, "column": col, "score": score}
            for r in runs.itertuples() for col, score in (r.drift_scores or {}).items()]
    return pd.DataFrame(rows, columns=["as_of", "column", "score"])


def version_changes(runs: pd.DataFrame) -> pd.DataFrame:
    """Days on which a different model version started serving."""
    if runs.empty:
        return runs
    changed = runs["model_version"] != runs["model_version"].shift()
    out = runs.loc[changed, ["as_of", "model_version", "model_type", "model_data_end"]]
    return out.iloc[1:]  # the first row is the initial model, not a change


def scored_forecasts(cfg: Config) -> pd.DataFrame:
    """Every logged forecast that has an actual to compare against."""
    log_path = cfg.resolve(cfg.serving.prediction_log)
    data_path = cfg.resolve(cfg.data.processed_path)
    if not log_path.exists() or not data_path.exists():
        return pd.DataFrame()
    preds = PredictionLog(log_path).read()
    if preds.empty:
        return pd.DataFrame()
    actuals = read_processed(data_path, ["id", "date", "sales"])
    end = min(preds["date"].max(), actuals["date"].max())
    window = (end - preds["date"].min()).days + 1
    return join_forecasts_to_actuals(preds, actuals, end, window)


def daily_totals(scored: pd.DataFrame) -> pd.DataFrame:
    """Total units per day across all series: actual vs the forecast in use that day."""
    if scored.empty:
        return pd.DataFrame(columns=["date", "actual", "forecast"])
    totals = scored.groupby("date")[["sales", "yhat"]].sum().reset_index()
    return totals.rename(columns={"sales": "actual", "yhat": "forecast"})


def series_view(cfg: Config, series_id: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Actual sales plus each model version's forecast for one series, long format."""
    actuals = read_processed(cfg.resolve(cfg.data.processed_path), ["id", "date", "sales"])
    actuals = actuals[actuals["id"].astype(str) == series_id]
    preds = PredictionLog(cfg.resolve(cfg.serving.prediction_log)).read()
    preds = preds[preds["series_id"] == series_id]
    preds = preds.sort_values("created_at").drop_duplicates(["model_version", "date"], keep="last")
    if since is not None:
        actuals = actuals[actuals["date"] >= since]
    out = pd.concat([
        actuals.assign(line="actual", value=actuals["sales"])[["date", "line", "value"]],
        preds.assign(line="forecast v" + preds["model_version"], value=preds["yhat"])
             [["date", "line", "value"]],
    ], ignore_index=True)
    return out.sort_values(["line", "date"]).reset_index(drop=True)


def series_ids(cfg: Config) -> list[str]:
    log_path = cfg.resolve(cfg.serving.prediction_log)
    if not log_path.exists():
        return []
    ids = PredictionLog(log_path).read()["series_id"].unique()
    return sorted(ids)
