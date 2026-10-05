"""Monitoring pipeline: score live forecasts against actuals, check drift, record the
result, and kick off retraining when a trigger fires."""

from __future__ import annotations

import datetime as dt

import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from forecast.api.prediction_log import PredictionLog
from forecast.flows.common import get_config, load_until
from forecast.flows.training import training_flow
from forecast.monitoring.drift import drift_windows, run_drift
from forecast.monitoring.performance import join_forecasts_to_actuals, performance_summary
from forecast.monitoring.policy import retrain_reasons
from forecast.monitoring.store import MonitoringStore
from forecast.registry import get_champion
from forecast.tracking import setup_mlflow

TASK_OPTS = {"cache_policy": NO_CACHE}


@task(**TASK_OPTS)
def champion_info(cfg) -> dict | None:
    setup_mlflow(cfg)
    champ = get_champion(MlflowClient(), cfg.registry.model_name)
    if champ is None:
        return None
    return {
        "version": str(champ.version),
        "model": champ.tags.get("model"),
        "data_end": pd.Timestamp(champ.tags["data_end"]),
        "cv_wape": float(champ.tags["cv_wape"]) if "cv_wape" in champ.tags else None,
    }


@task(**TASK_OPTS)
def live_performance(cfg, df, as_of) -> dict:
    preds = PredictionLog(cfg.resolve(cfg.serving.prediction_log)).read()
    joined = join_forecasts_to_actuals(preds, df, as_of, cfg.monitoring.window_days)
    return performance_summary(joined)


@task(**TASK_OPTS)
def data_drift(cfg, df, reference_end, as_of):
    m = cfg.monitoring
    reference, current = drift_windows(df, reference_end, as_of, m.drift_window_days)
    html = cfg.resolve(m.reports_dir) / f"drift_{as_of.date()}.html"
    return run_drift(reference, current, html)


@flow(name="monitoring", log_prints=True)
def monitoring_flow(as_of: str | None = None, config_path: str | None = None,
                    auto_retrain: bool = True) -> dict:
    logger = get_run_logger()
    cfg = get_config(config_path)
    df, as_of_ts = load_until(cfg, as_of)

    champ = champion_info(cfg)
    if champ is None:
        logger.warning("No champion yet: running the training flow")
        result = training_flow(as_of=as_of, config_path=config_path)
        return {"as_of": str(as_of_ts.date()), "decision": "retrain",
                "reasons": ["no champion"], "training": result}

    perf = live_performance(cfg, df, as_of_ts)
    age = (as_of_ts - champ["data_end"]).days
    # With no new data since training, drift is meaningless (identical windows).
    drift = data_drift(cfg, df, champ["data_end"], as_of_ts) if age > 0 else None

    reasons = retrain_reasons(
        cfg.monitoring, perf["live_wape"], champ["cv_wape"],
        drift.share if drift else None, age, perf["n_points"],
        drift.drifted_columns if drift else None, drift.scores if drift else None,
        live_bias=perf["live_bias"] if perf["n_points"] else None,
    )
    decision = "retrain" if reasons else "ok"
    ratio = (perf["live_wape"] / champ["cv_wape"]
             if champ["cv_wape"] and perf["n_points"] else None)

    row = {
        "as_of": str(as_of_ts.date()),
        "run_at": dt.datetime.now(dt.UTC).isoformat(),
        "model_version": champ["version"],
        "model_type": champ["model"],
        "model_data_end": str(champ["data_end"].date()),
        "model_age_days": age,
        **{k: perf[k] for k in ("n_points", "live_wape", "live_bias", "wape_h1_7",
                                "wape_h8_28")},
        "cv_wape": champ["cv_wape"],
        "wape_ratio": ratio,
        "drift_share": drift.share if drift else None,
        "drifted_columns": drift.drifted_columns if drift else [],
        "drift_scores": drift.scores if drift else {},
        "decision": decision,
        "reasons": reasons,
        "report_path": str(drift.report_path) if drift else None,
    }
    MonitoringStore(cfg.resolve(cfg.monitoring.db_path)).write(row)
    logger.info("as_of %s | v%s age %dd | live WAPE %s | drift %s | %s %s",
                row["as_of"], champ["version"], age,
                f"{perf['live_wape']:.3f}" if perf["n_points"] else "n/a",
                f"{drift.share:.0%}" if drift else "n/a", decision.upper(), reasons)

    if reasons and auto_retrain:
        row["training"] = training_flow(as_of=str(as_of_ts.date()), model=champ["model"],
                                        config_path=config_path)
    return row
