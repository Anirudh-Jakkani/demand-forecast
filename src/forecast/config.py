"""Typed project configuration loaded from configs/config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
TRACKING_URI_ENV = "FORECAST_TRACKING_URI"


class DataConfig(BaseModel):
    raw_dir: Path
    processed_path: Path
    state: str = "CA"
    category: str = "FOODS"
    start_date: str | None = None


class BacktestConfig(BaseModel):
    horizon: int = 28
    n_folds: int = 4
    step: int = 28


class MlflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment: str = "m5-demand-forecast"
    artifact_root: Path = Path("mlruns")


class RegistryConfig(BaseModel):
    model_name: str = "m5-demand-forecaster"
    # A different model recipe must beat the champion's backtest WAPE by this fraction.
    min_improvement: float = 0.01


class MonitoringConfig(BaseModel):
    window_days: int = 7               # score live forecasts over the last N days of actuals
    min_points: int = 100              # don't judge accuracy on fewer forecast/actual pairs
    max_wape_ratio: float = 1.25       # retrain if live WAPE > ratio x backtest WAPE
    drift_window_days: int = 28        # size of reference and current windows for drift
    drift_share_threshold: float = 0.5 # retrain if this share of columns drifted
    target_drift_trigger: bool = True  # also retrain if the target column alone drifts
    target_column: str = "sales"
    drift_cooldown_days: int = 7       # ignore drift for N days after a retrain (no flapping)
    max_model_age_days: int = 35       # retrain if the champion's data is older than this
    db_path: Path = Path("data/monitoring.sqlite")
    reports_dir: Path = Path("reports")


class ScheduleConfig(BaseModel):
    training_cron: str = "0 3 * * 1"   # Mondays 03:00
    monitoring_cron: str = "0 6 * * *" # daily 06:00
    timezone: str = "UTC"


class ServingConfig(BaseModel):
    prediction_log: Path = Path("data/predictions.sqlite")
    reload_interval_s: int = 60


class Config(BaseModel):
    data: DataConfig
    backtest: BacktestConfig = BacktestConfig()
    models: dict[str, dict[str, Any]] = {}
    mlflow: MlflowConfig = MlflowConfig()
    registry: RegistryConfig = RegistryConfig()
    serving: ServingConfig = ServingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    schedule: ScheduleConfig = ScheduleConfig()

    def resolve(self, path: Path) -> Path:
        """Relative paths in the config are relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def model_params(self, name: str) -> dict[str, Any]:
        return dict(self.models.get(name) or {})


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    # Lets containers point at a tracking server (e.g. http://mlflow:5000) while sharing
    # the same YAML as local runs. Deliberately NOT MLFLOW_TRACKING_URI: mlflow writes that
    # variable into os.environ on every set_tracking_uri(), so honouring it would make one
    # workspace's registry leak into every config loaded later in the same process.
    if os.environ.get(TRACKING_URI_ENV):
        cfg.mlflow.tracking_uri = os.environ[TRACKING_URI_ENV]
    return cfg
