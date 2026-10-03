"""Typed project configuration loaded from configs/config.yaml."""

from __future__ import annotations

from pathlib import Path

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


class Config(BaseModel):
    data: DataConfig
    backtest: BacktestConfig = BacktestConfig()
    mlflow: MlflowConfig = MlflowConfig()

    def resolve(self, path: Path) -> Path:
        """Relative paths in the config are relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
