"""Global LightGBM model: one model across all store x item series.

Tweedie loss fits retail demand well: non-negative, many zeros, long right tail.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features.build import CATEGORICAL, FEATURES, LOOKBACK_DAYS, MIN_LAG, add_features
from forecast.models.base import ForecastModel

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": 42,
}


class LGBMForecaster(ForecastModel):
    name = "lightgbm"

    def __init__(self, num_boost_round: int = 600, train_days: int | None = 730,
                 chunk_series: int = 1000, **params):
        """`train_days` limits fitting to the most recent N days (features still use
        older history), trading a little accuracy for much faster training.
        `chunk_series` bounds peak memory while building features (no effect on results)."""
        self.num_boost_round = num_boost_round
        self.train_days = train_days
        self.chunk_series = chunk_series
        self.params = {**DEFAULT_PARAMS, **params}
        self.booster: lgb.Booster | None = None

    def fit(self, train: pd.DataFrame) -> LGBMForecaster:
        self._cutoff = train["date"].max()
        self._categories = {c: train[c].astype("category").cat.categories for c in CATEGORICAL}

        # Only build features for rows we train on, plus the history they look back into,
        # and do it in chunks of series so the full-width feature frame never exists at once.
        # Feature values are identical either way (features never cross series).
        source = train
        first_day = None
        if self.train_days:
            first_day = self._cutoff - pd.Timedelta(days=self.train_days)
            source = train[train["date"] > first_day - pd.Timedelta(days=LOOKBACK_DAYS)]
        # Count training rows up front (a row trains once it has MIN_LAG days of history) so
        # the float32 design matrix is allocated once and filled chunk by chunk.
        position = source.groupby("id", observed=True, sort=False).cumcount().to_numpy()
        trains = position >= MIN_LAG
        if first_day is not None:
            trains &= (source["date"] > first_day).to_numpy()
        X = np.empty((int(trains.sum()), len(FEATURES)), dtype="float32")
        y = np.empty(len(X), dtype="float32")

        ids = source["id"].unique()
        filled = 0
        for i in range(0, len(ids), self.chunk_series):
            part = add_features(source[source["id"].isin(ids[i:i + self.chunk_series])])
            keep = part[f"lag_{MIN_LAG}"].notna()
            if first_day is not None:
                keep &= part["date"] > first_day
            part = part[keep]
            X[filled:filled + len(part)] = self._matrix(part)
            y[filled:filled + len(part)] = part["sales"].to_numpy(dtype="float32")
            filled += len(part)
            del part
        assert filled == len(X), "training row count mismatch"
        del source, position, trains

        dataset = lgb.Dataset(X, label=y, feature_name=FEATURES,
                              categorical_feature=CATEGORICAL, free_raw_data=True).construct()
        del X, y  # LightGBM now holds its own compact binned copy
        self.booster = lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

        # Keep only the tail of history that predict() needs to rebuild features.
        self._history = train[train["date"] > self._cutoff - pd.Timedelta(days=LOOKBACK_DAYS)]
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        horizon = (future["date"].max() - self._cutoff).days
        if horizon > MIN_LAG:
            raise ValueError(f"Horizon {horizon}d exceeds MIN_LAG={MIN_LAG}d: features would leak")

        # Only rebuild features for the requested series: keeps single-item API calls fast.
        history = self._history[self._history["id"].isin(future["id"].unique())]
        combined = pd.concat(
            [history, future.assign(sales=float("nan"), _row=future.index)],
            ignore_index=True,
        )
        for c in CATEGORICAL:  # concat can widen categories; keep the training set
            combined[c] = combined[c].astype("object")
        feats = add_features(combined)
        feats = feats[feats["date"] > self._cutoff]

        yhat = self.booster.predict(self._matrix(feats))
        rows = feats["_row"].astype("int64").to_numpy()
        return pd.Series(yhat, index=rows, name="yhat").reindex(future.index)

    def _matrix(self, feats: pd.DataFrame) -> np.ndarray:
        """float32 design matrix in FEATURES order. Categoricals become their code in the
        training categories; unseen values become NaN, which LightGBM treats as missing."""
        X = np.empty((len(feats), len(FEATURES)), dtype="float32")
        for j, col in enumerate(FEATURES):
            if col in self._categories:
                codes = pd.Categorical(feats[col].astype("object"),
                                       categories=self._categories[col]).codes
                X[:, j] = np.where(codes < 0, np.nan, codes)
            else:
                X[:, j] = feats[col].to_numpy(dtype="float32", na_value=np.nan)
        return X

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": self.booster.feature_name(),
                "gain": self.booster.feature_importance("gain"),
                "split": self.booster.feature_importance("split"),
            })
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    def get_params(self) -> dict[str, Any]:
        return {"num_boost_round": self.num_boost_round, "train_days": self.train_days,
                **self.params}
