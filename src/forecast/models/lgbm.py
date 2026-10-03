"""Global LightGBM model: one model across all store x item series.

Tweedie loss fits retail demand well: non-negative, many zeros, long right tail.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
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

    def __init__(self, num_boost_round: int = 600, train_days: int | None = 730, **params):
        """`train_days` limits fitting to the most recent N days (features still use
        older history), trading a little accuracy for much faster training."""
        self.num_boost_round = num_boost_round
        self.train_days = train_days
        self.params = {**DEFAULT_PARAMS, **params}
        self.booster: lgb.Booster | None = None

    def fit(self, train: pd.DataFrame) -> LGBMForecaster:
        self._cutoff = train["date"].max()
        self._categories = {c: train[c].astype("category").cat.categories for c in CATEGORICAL}

        feats = add_features(train)
        feats = feats[feats[f"lag_{MIN_LAG}"].notna()]
        if self.train_days:
            feats = feats[feats["date"] > self._cutoff - pd.Timedelta(days=self.train_days)]

        dataset = lgb.Dataset(
            self._encode(feats[FEATURES]), label=feats["sales"],
            categorical_feature=CATEGORICAL, free_raw_data=True,
        )
        self.booster = lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

        # Keep only the tail of history that predict() needs to rebuild features.
        self._history = train[train["date"] > self._cutoff - pd.Timedelta(days=LOOKBACK_DAYS)]
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        horizon = (future["date"].max() - self._cutoff).days
        if horizon > MIN_LAG:
            raise ValueError(f"Horizon {horizon}d exceeds MIN_LAG={MIN_LAG}d: features would leak")

        combined = pd.concat(
            [self._history, future.assign(sales=float("nan"), _row=future.index)],
            ignore_index=True,
        )
        for c in CATEGORICAL:  # concat can widen categories; keep the training set
            combined[c] = combined[c].astype("object")
        feats = add_features(combined)
        feats = feats[feats["date"] > self._cutoff]

        yhat = self.booster.predict(self._encode(feats[FEATURES]))
        rows = feats["_row"].astype("int64").to_numpy()
        return pd.Series(yhat, index=rows, name="yhat").reindex(future.index)

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype("object"), categories=self._categories[c])
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
