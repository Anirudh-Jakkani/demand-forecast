"""Forecast accuracy metrics.

WAPE is the headline metric: it is scale-free across series, handles zeros, and
reads as "% of units we got wrong". RMSSE is the M5 competition metric (unweighted here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else float("nan")


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Positive = over-forecasting, as a share of actual volume."""
    denom = np.sum(y)
    return float((np.sum(yhat) - np.sum(y)) / denom) if denom > 0 else float("nan")


def rmsse(train: pd.DataFrame, test: pd.DataFrame, pred_col: str = "yhat") -> float:
    """Mean over series of RMSE scaled by the in-sample one-step naive RMSE.

    Following M5, the scale only uses history from each series' first non-zero sale.
    Series with a zero scale (constant history) are skipped.
    """
    train = train.sort_values(["id", "date"])
    active = train[train.groupby("id", observed=True)["sales"].cummax() > 0]
    diffs = active.groupby("id", observed=True)["sales"].diff()
    scale = diffs.pow(2).groupby(active["id"], observed=True).mean()

    err = (test["sales"] - test[pred_col]).pow(2).groupby(test["id"], observed=True).mean()
    ratio = (err / scale).replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.sqrt(ratio).mean())


def score(train: pd.DataFrame, test: pd.DataFrame, pred_col: str = "yhat") -> dict[str, float]:
    y = test["sales"].to_numpy(dtype=float)
    yhat = test[pred_col].to_numpy(dtype=float)
    return {
        "wape": wape(y, yhat),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "bias": bias(y, yhat),
        "rmsse": rmsse(train, test, pred_col),
    }
