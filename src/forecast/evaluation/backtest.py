"""Rolling-origin backtesting that strictly respects time order.

For fold k the model sees data up to and including `cutoff_k` and must forecast
(cutoff_k, cutoff_k + horizon]. The future frame handed to `predict` has the target
column removed, so a model physically cannot peek at the answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from forecast.evaluation.metrics import score
from forecast.models.base import ForecastModel


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: pd.Timestamp
    horizon: int

    @property
    def test_end(self) -> pd.Timestamp:
        return self.cutoff + pd.Timedelta(days=self.horizon)


def make_folds(last_date: pd.Timestamp, horizon: int, n_folds: int, step: int) -> list[Fold]:
    """Oldest fold first; the last fold's test window ends exactly at `last_date`."""
    last_date = pd.Timestamp(last_date)
    cutoffs = [last_date - pd.Timedelta(days=horizon + step * k) for k in range(n_folds)]
    return [Fold(i, c, horizon) for i, c in enumerate(sorted(cutoffs))]


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= fold.cutoff]
    test = df[(df["date"] > fold.cutoff) & (df["date"] <= fold.test_end)]
    # Only score series that exist in training; a model can't forecast an unseen launch.
    test = test[test["id"].isin(train["id"].unique())]
    return train, test


def iter_splits(df, horizon, n_folds, step) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    for fold in make_folds(df["date"].max(), horizon, n_folds, step):
        train, test = split(df, fold)
        yield fold, train, test


@dataclass
class BacktestResult:
    fold_metrics: pd.DataFrame      # one row per fold
    predictions: pd.DataFrame       # id, date, fold, sales, yhat
    last_model: ForecastModel       # model from the most recent fold, for inspection

    @property
    def summary(self) -> dict[str, float]:
        return self.fold_metrics.drop(columns=["fold", "cutoff"]).mean().to_dict()


def run_backtest(
    model_factory, df: pd.DataFrame, horizon: int = 28, n_folds: int = 4, step: int = 28
) -> BacktestResult:
    """`model_factory` builds a fresh model per fold so no state leaks across folds."""
    rows, preds = [], []
    for fold, train, test in iter_splits(df, horizon, n_folds, step):
        model: ForecastModel = model_factory()
        model.fit(train)
        future = test.drop(columns=["sales"])
        yhat = model.predict(future)

        scored = test[["id", "date", "sales"]].copy()
        scored["yhat"] = yhat.to_numpy()
        scored["fold"] = fold.index
        rows.append({"fold": fold.index, "cutoff": fold.cutoff, **score(train, scored)})
        preds.append(scored)

    return BacktestResult(pd.DataFrame(rows), pd.concat(preds, ignore_index=True), model)
