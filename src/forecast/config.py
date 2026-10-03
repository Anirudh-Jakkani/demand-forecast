"""Typed project configuration loaded from configs/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


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

    def resolve(self, path: Path) -> Path:
        """Relative paths in the config are relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def model_params(self, name: str) -> dict[str, Any]:
        return dict(self.models.get(name) or {})


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
