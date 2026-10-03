"""Train -> register -> promote.

A *recipe* is a model type plus its hyperparameters. Promotion rules:

1. No champion yet                         -> promote.
2. Same recipe as the champion             -> promote (same method, fresher data).
3. Different recipe                        -> backtest the champion's recipe on the same
   data and folds; promote only if the candidate's WAPE is lower by `min_improvement`.
   Otherwise the candidate gets the `challenger` alias so it stays easy to find.

Comparing recipes on identical folds keeps the comparison fair even when the champion
was trained on an older slice of data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from forecast.config import PROJECT_ROOT, Config
from forecast.data.future import build_future_frame
from forecast.evaluation.backtest import BacktestResult, run_backtest
from forecast.models import MODELS
from forecast.serving_model import ForecastBundle
from forecast.tracking import log_backtest_details, setup_mlflow

log = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"


@dataclass
class TrainOutcome:
    run_id: str
    version: str
    cv_wape: float
    promoted: bool
    reason: str
    champion_wape: float | None = None


def recipe(model_name: str, params: dict) -> dict:
    return {"model": model_name, "params": params}


def recipe_hash(r: dict) -> str:
    return hashlib.sha1(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:12]


def backtest_recipe(r: dict, df: pd.DataFrame, cfg: Config) -> BacktestResult:
    bt = cfg.backtest
    model_cls, params = MODELS[r["model"]], r["params"]
    return run_backtest(lambda: model_cls(**params), df, bt.horizon, bt.n_folds, bt.step)


def get_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None


def train_and_register(
    cfg: Config, df: pd.DataFrame, model_name: str, promote: bool = True
) -> TrainOutcome:
    setup_mlflow(cfg)
    client = MlflowClient()
    registered_name = cfg.registry.model_name
    rec = recipe(model_name, cfg.model_params(model_name))
    rhash = recipe_hash(rec)
    data_end = df["date"].max()

    log.info("Backtesting candidate %s", rec)
    result = backtest_recipe(rec, df, cfg)
    cv_wape = result.summary["wape"]

    log.info("Fitting %s on all data up to %s", model_name, data_end.date())
    model = MODELS[model_name](**rec["params"]).fit(df)
    future = build_future_frame(df, cfg.resolve(cfg.data.raw_dir), cfg.backtest.horizon,
                                cfg.data.state)
    bundle = ForecastBundle(model, future, model_name)

    with mlflow.start_run(run_name=f"train-{model_name}") as run:
        mlflow.set_tags({"model": model_name, "stage": "train", "recipe_hash": rhash})
        mlflow.log_params({**rec["params"], **cfg.backtest.model_dump(),
                           "data_end": str(data_end.date()), "n_series": df["id"].nunique()})
        mlflow.log_dict(rec, "recipe.json")
        log_backtest_details(result)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=bundle,
            code_paths=[str(PROJECT_ROOT / "src" / "forecast")],
            registered_model_name=registered_name,
        )
    version = str(info.registered_model_version)
    for key, value in {
        "model": model_name, "recipe_hash": rhash, "recipe": json.dumps(rec, default=str),
        "cv_wape": f"{cv_wape:.6f}", "data_end": str(data_end.date()),
    }.items():
        client.set_model_version_tag(registered_name, version, key, value)

    outcome = TrainOutcome(run.info.run_id, version, cv_wape, promoted=False, reason="")
    if promote:
        _decide_promotion(client, cfg, df, outcome, rhash)
    else:
        outcome.reason = "promotion skipped (--no-promote)"
    log.info("Version %s: %s", version, outcome.reason)
    return outcome


def _decide_promotion(
    client: MlflowClient, cfg: Config, df: pd.DataFrame, outcome: TrainOutcome, rhash: str
) -> None:
    name = cfg.registry.model_name
    champ = get_champion(client, name)

    if champ is None:
        outcome.promoted, outcome.reason = True, "no champion yet"
    elif champ.tags.get("recipe_hash") == rhash:
        outcome.promoted = True
        outcome.reason = f"same recipe as champion v{champ.version}, retrained on newer data"
    else:
        champ_recipe = json.loads(champ.tags["recipe"])
        log.info("Re-backtesting champion v%s recipe on the same folds", champ.version)
        champ_wape = backtest_recipe(champ_recipe, df, cfg).summary["wape"]
        outcome.champion_wape = champ_wape
        threshold = champ_wape * (1 - cfg.registry.min_improvement)
        outcome.promoted = outcome.cv_wape < threshold
        verdict = "beats" if outcome.promoted else "does not beat"
        outcome.reason = (f"WAPE {outcome.cv_wape:.4f} {verdict} champion v{champ.version} "
                          f"{champ_wape:.4f} by the required {cfg.registry.min_improvement:.0%}")

    alias = CHAMPION if outcome.promoted else CHALLENGER
    client.set_registered_model_alias(name, alias, outcome.version)
    client.set_model_version_tag(name, outcome.version, "promotion_reason", outcome.reason)


def promote_version(cfg: Config, version: str) -> None:
    """Manual override / rollback: point the champion alias at any version."""
    setup_mlflow(cfg)
    client = MlflowClient()
    client.set_registered_model_alias(cfg.registry.model_name, CHAMPION, version)
    client.set_model_version_tag(cfg.registry.model_name, version, "promotion_reason",
                                 "manually promoted")


def list_versions(cfg: Config) -> pd.DataFrame:
    setup_mlflow(cfg)
    client = MlflowClient()
    name = cfg.registry.model_name
    versions = client.search_model_versions(f"name='{name}'")
    rows = []
    for v in versions:
        full = client.get_model_version(name, v.version)
        rows.append({
            "version": int(v.version),
            "aliases": ",".join(full.aliases),
            "model": full.tags.get("model"),
            "cv_wape": float(full.tags.get("cv_wape", "nan")),
            "data_end": full.tags.get("data_end"),
            "reason": full.tags.get("promotion_reason", ""),
        })
    return pd.DataFrame(rows).sort_values("version") if rows else pd.DataFrame()


def load_champion(cfg: Config) -> tuple[ForecastBundle, ModelVersion]:
    setup_mlflow(cfg)
    client = MlflowClient()
    champ = get_champion(client, cfg.registry.model_name)
    if champ is None:
        raise LookupError(f"No '{CHAMPION}' alias on model '{cfg.registry.model_name}'. "
                          "Run `forecast train` first.")
    pyfunc = mlflow.pyfunc.load_model(f"models:/{cfg.registry.model_name}/{champ.version}")
    return pyfunc.unwrap_python_model(), champ
