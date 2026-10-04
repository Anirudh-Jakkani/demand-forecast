"""MLflow setup and logging for backtest and training runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from forecast.config import PROJECT_ROOT, Config  # noqa: E402
from forecast.evaluation.backtest import BacktestResult  # noqa: E402


def tracking_uri(cfg: Config) -> str:
    uri = cfg.mlflow.tracking_uri
    # Anchor relative sqlite paths to the project root so runs land in one place.
    if uri.startswith("sqlite:///") and not Path(uri.removeprefix("sqlite:///")).is_absolute():
        uri = f"sqlite:///{(PROJECT_ROOT / uri.removeprefix('sqlite:///')).as_posix()}"
    return uri


def is_local_store(uri: str) -> bool:
    return uri.startswith(("sqlite:", "file:")) or "://" not in uri


def setup_mlflow(cfg: Config) -> None:
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(cfg.mlflow.experiment) is None:
        if is_local_store(uri):
            # Pin artifacts to a fixed folder instead of "wherever the process was started".
            root = cfg.resolve(cfg.mlflow.artifact_root)
            root.mkdir(parents=True, exist_ok=True)
            client.create_experiment(cfg.mlflow.experiment, artifact_location=root.as_uri())
        else:
            # A tracking server (e.g. in Docker) decides where artifacts live and proxies them.
            client.create_experiment(cfg.mlflow.experiment)
    mlflow.set_experiment(cfg.mlflow.experiment)


def log_backtest(
    model_name: str, params: dict, result: BacktestResult, cfg: Config, data_info: dict
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("stage", "backtest")
        mlflow.log_params({**params, **cfg.backtest.model_dump(), **data_info})
        log_backtest_details(result)
        return run.info.run_id


def log_backtest_details(result: BacktestResult) -> None:
    """Per-fold + mean metrics and diagnostic artifacts, into the active run."""
    for row in result.fold_metrics.itertuples(index=False):
        for metric in ("wape", "mae", "rmse", "bias", "rmsse"):
            mlflow.log_metric(f"fold_{metric}", getattr(row, metric), step=row.fold)
    mlflow.log_metrics({f"cv_{k}": v for k, v in result.summary.items()})
    for row in result.by_horizon().itertuples(index=False):
        key = row.horizon.replace("days ", "h").replace("-", "_")   # e.g. h1_7
        mlflow.log_metrics({f"cv_wape_{key}": row.wape, f"cv_bias_{key}": row.bias})

    with tempfile.TemporaryDirectory() as tmp:
        fold_csv = Path(tmp) / "fold_metrics.csv"
        result.fold_metrics.to_csv(fold_csv, index=False)
        mlflow.log_artifact(str(fold_csv))
        mlflow.log_figure(plot_backtest(result), "backtest_total_sales.png")

        if hasattr(result.last_model, "feature_importance"):
            imp = result.last_model.feature_importance()
            imp_csv = Path(tmp) / "feature_importance.csv"
            imp.to_csv(imp_csv, index=False)
            mlflow.log_artifact(str(imp_csv))
            mlflow.log_figure(plot_importance(imp), "feature_importance.png")


def plot_backtest(result: BacktestResult):
    """Total daily units across all series: actual vs forecast, per fold."""
    daily = result.predictions.groupby(["fold", "date"])[["sales", "yhat"]].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    for fold, g in daily.groupby("fold"):
        ax.plot(g["date"], g["sales"], color="0.3", lw=1.2, label="actual" if fold == 0 else None)
        ax.plot(g["date"], g["yhat"], color="C0", lw=1.2, label="forecast" if fold == 0 else None)
        ax.axvline(g["date"].min(), color="0.8", ls="--", lw=0.8)
    ax.set_ylabel("units / day (all series)")
    ax.set_title("Backtest: actual vs forecast per fold")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_importance(imp, top: int = 20):
    top_imp = imp.head(top).iloc[::-1]
    labels = top_imp["feature"]
    if "bucket" in top_imp and top_imp["bucket"].nunique() > 1:
        labels = labels + "  (" + top_imp["bucket"] + ")"
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(top_imp) + 1))
    ax.barh(labels, top_imp["gain"], color="C0")
    ax.set_xlabel("total gain (last fold)")
    ax.set_title(f"Top {len(top_imp)} features")
    fig.tight_layout()
    plt.close(fig)
    return fig
