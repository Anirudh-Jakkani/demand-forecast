import pytest

from forecast.config import (
    BacktestConfig,
    Config,
    DataConfig,
    MlflowConfig,
    MonitoringConfig,
    RegistryConfig,
    ServingConfig,
)
from forecast.data.ingest import load_m5_long
from forecast.data.synthetic import make_synthetic_m5

FAST_LGBM = {"num_boost_round": 30, "num_leaves": 15, "min_data_in_leaf": 20, "train_days": None}


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return make_synthetic_m5(tmp_path_factory.mktemp("raw"), n_days=300)


@pytest.fixture(scope="session")
def long_df(raw_dir):
    return load_m5_long(raw_dir, state="CA", category="FOODS")


def make_cfg(tmp_path, raw_dir) -> Config:
    """Config whose every output (MLflow, logs, reports) lives under tmp_path."""
    return Config(
        data=DataConfig(raw_dir=raw_dir, processed_path=tmp_path / "data.parquet"),
        backtest=BacktestConfig(horizon=28, n_folds=2, step=28),
        models={"lightgbm": FAST_LGBM},
        mlflow=MlflowConfig(tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                            experiment="test", artifact_root=tmp_path / "mlruns"),
        registry=RegistryConfig(model_name="test-forecaster", min_improvement=0.01),
        serving=ServingConfig(prediction_log=tmp_path / "pred.sqlite", reload_interval_s=3600),
        monitoring=MonitoringConfig(db_path=tmp_path / "mon.sqlite",
                                    reports_dir=tmp_path / "reports", min_points=50),
    )
