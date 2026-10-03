"""Training pipeline: load -> validate -> train/register/promote -> batch forecast."""

from __future__ import annotations

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from forecast.data.validate import validate_long
from forecast.flows.common import batch_forecast_champion, get_config, load_until
from forecast.registry import train_and_register

# DataFrames are large and change every run: never hash them for caching.
TASK_OPTS = {"cache_policy": NO_CACHE}


@task(**TASK_OPTS)
def load_data(cfg, as_of):
    df, as_of_ts = load_until(cfg, as_of)
    get_run_logger().info("Loaded %d rows, %d series, up to %s",
                          len(df), df["id"].nunique(), as_of_ts.date())
    return df


@task(**TASK_OPTS)
def validate(df):
    validate_long(df)


@task(**TASK_OPTS, retries=1, retry_delay_seconds=30)
def train_register_promote(cfg, df, model):
    out = train_and_register(cfg, df, model)
    get_run_logger().info("v%s %s: %s", out.version, "PROMOTED" if out.promoted else
                          "not promoted", out.reason)
    return out


@task(**TASK_OPTS)
def batch_forecast(cfg):
    return batch_forecast_champion(cfg)


@flow(name="training", log_prints=True)
def training_flow(as_of: str | None = None, model: str = "lightgbm",
                  config_path: str | None = None) -> dict:
    cfg = get_config(config_path)
    df = load_data(cfg, as_of)
    validate(df)
    outcome = train_register_promote(cfg, df, model)
    batch = batch_forecast(cfg) if outcome.promoted else {"skipped": True}
    return {
        "as_of": str(df["date"].max().date()),
        "model": model,
        "version": outcome.version,
        "cv_wape": outcome.cv_wape,
        "promoted": outcome.promoted,
        "reason": outcome.reason,
        "batch_forecast": batch,
    }
