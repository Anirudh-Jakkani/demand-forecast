"""Replay history one day at a time to show the MLOps loop working end to end.

    train on day 0 -> every day: monitor (score forecasts, check drift, maybe retrain)

A shock (demand and/or price change) can be injected from a given date so you can
watch accuracy degrade, drift fire, a retrain trigger, and accuracy recover.

Everything is written to a separate workspace (default `sim/`) with its own MLflow
registry, prediction log and monitoring store, so the real project state is untouched.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml
from prefect import flow, get_run_logger

from forecast.config import PROJECT_ROOT, Config
from forecast.flows.monitoring import monitoring_flow
from forecast.flows.training import training_flow

META_FILE = "sim_meta.json"
DEFAULT_WORKDIR = PROJECT_ROOT / "sim"


@dataclass
class Shock:
    date: str
    demand_factor: float = 1.0          # multiply sales from `date` onward
    price_factor: float = 1.0           # multiply sell_price from `date` onward
    dept: str | None = None             # limit to one department, e.g. FOODS_3

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df["date"] >= pd.Timestamp(self.date)
        if self.dept:
            mask &= df["dept_id"].astype(str) == self.dept
        if self.demand_factor != 1.0:
            df.loc[mask, "sales"] = (df.loc[mask, "sales"] * self.demand_factor).round()
        if self.price_factor != 1.0:
            df.loc[mask, "sell_price"] = df.loc[mask, "sell_price"] * self.price_factor
        return df


def sim_config(base: Config, workdir: Path) -> Config:
    """Same model/backtest/monitoring settings, every output redirected into `workdir`."""
    cfg = base.model_copy(deep=True)
    cfg.data.raw_dir = base.resolve(base.data.raw_dir)
    cfg.data.processed_path = workdir / "data.parquet"
    cfg.mlflow.tracking_uri = f"sqlite:///{(workdir / 'mlflow.db').as_posix()}"
    cfg.mlflow.artifact_root = workdir / "mlruns"
    cfg.mlflow.experiment = f"{base.mlflow.experiment}-sim"
    cfg.registry.model_name = f"{base.registry.model_name}-sim"
    cfg.serving.prediction_log = workdir / "predictions.sqlite"
    cfg.monitoring.db_path = workdir / "monitoring.sqlite"
    cfg.monitoring.reports_dir = workdir / "reports"
    # Without the weekly schedule, a model older than its horizon has no forecasts left
    # to score, so cap the age trigger at the horizon to avoid blind days.
    cfg.monitoring.max_model_age_days = min(base.monitoring.max_model_age_days,
                                            base.backtest.horizon)
    return cfg


def prepare_workspace(base: Config, workdir: Path, start: str, days: int,
                      shock: Shock | None, model: str, weekly_retrain: bool) -> Path:
    """Wipe and rebuild the sim workspace; returns the path of its config.yaml."""
    workdir = Path(workdir)
    if workdir.exists():
        # Only ever delete a folder this module created.
        if any(workdir.iterdir()) and not (workdir / META_FILE).exists():
            raise FileExistsError(f"{workdir} exists and is not a simulation workspace")
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    cfg = sim_config(base, workdir)
    df = pd.read_parquet(base.resolve(base.data.processed_path))
    if shock:
        df = shock.apply(df)
    df.to_parquet(cfg.data.processed_path, index=False)

    cfg_path = workdir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
                        encoding="utf-8")
    meta = {"start": start, "days": days, "model": model, "weekly_retrain": weekly_retrain,
            "shock": asdict(shock) if shock else None}
    (workdir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cfg_path


@flow(name="replay", log_prints=True)
def replay_flow(config_path: str, start: str, days: int, model: str = "lightgbm",
                weekly_retrain: bool = False) -> list[dict]:
    logger = get_run_logger()
    start_ts = pd.Timestamp(start)
    training_flow(as_of=start, model=model, config_path=config_path)

    results = []
    for offset in range(1, days + 1):
        as_of = start_ts + pd.Timedelta(days=offset)
        if weekly_retrain and as_of.dayofweek == 0:  # the scheduled Monday retrain
            training_flow(as_of=str(as_of.date()), model=model, config_path=config_path)
        row = monitoring_flow(as_of=str(as_of.date()), config_path=config_path)
        results.append({k: row.get(k) for k in
                        ("as_of", "model_version", "live_wape", "drift_share", "decision")})
        logger.info("day %d/%d done", offset, days)
    return results


def run_replay(base: Config, start: str, days: int, shock: Shock | None = None,
               model: str = "lightgbm", weekly_retrain: bool = False,
               workdir: Path = DEFAULT_WORKDIR) -> list[dict]:
    dates = pd.read_parquet(base.resolve(base.data.processed_path), columns=["date"])["date"]
    data_end = dates.max()
    if pd.Timestamp(start) + pd.Timedelta(days=days) > data_end:
        raise ValueError(f"start + days goes past the last day of data ({data_end.date()})")
    cfg_path = prepare_workspace(base, workdir, start, days, shock, model, weekly_retrain)
    return replay_flow(config_path=str(cfg_path), start=start, days=days, model=model,
                       weekly_retrain=weekly_retrain)


def load_meta(workdir: Path = DEFAULT_WORKDIR) -> dict | None:
    path = Path(workdir) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
