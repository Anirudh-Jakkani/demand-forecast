"""Global LightGBM model: one model across all store x item series.

Tweedie loss fits retail demand well: non-negative, many zeros, long right tail.

Horizon buckets: with `horizon_buckets=(7, 14, 28)` three boosters are trained. The one for
forecast days 1-7 may use sales lagged >= 7 days, the one for days 8-14 lags >= 14, and the
one for days 15-28 lags >= 28. Each is leak-free for its own days, and near days get much
fresher signal than a single lag-28 model allows. `(28,)` is the single-model setup.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.features.build import CATEGORICAL, LOOKBACK_DAYS, add_features, feature_names
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
                 horizon_buckets: tuple[int, ...] | list[int] = (28,),
                 chunk_series: int = 1000, **params):
        """`train_days` limits fitting to the most recent N days (features still use
        older history), trading a little accuracy for much faster training.
        `horizon_buckets` are the last forecast day of each bucket (= its minimum lag).
        `chunk_series` bounds peak memory while building features (no effect on results)."""
        self.num_boost_round = num_boost_round
        self.train_days = train_days
        self.horizon_buckets = tuple(sorted(int(b) for b in horizon_buckets))
        self.chunk_series = chunk_series
        self.params = {**DEFAULT_PARAMS, **params}
        self.boosters: dict[int, lgb.Booster] = {}

    @property
    def max_horizon(self) -> int:
        return self.horizon_buckets[-1]

    def fit(self, train: pd.DataFrame) -> LGBMForecaster:
        self._cutoff = train["date"].max()
        self._categories = {c: train[c].astype("category").cat.categories for c in CATEGORICAL}

        # Only build features for rows we train on, plus the history they look back into.
        source = train
        first_day = None
        if self.train_days:
            first_day = self._cutoff - pd.Timedelta(days=self.train_days)
            source = train[train["date"] > first_day - pd.Timedelta(days=LOOKBACK_DAYS)]

        # One booster per bucket, trained one after another so only one design matrix
        # exists at a time.
        for min_lag in self.horizon_buckets:
            self.boosters[min_lag] = self._fit_bucket(source, min_lag, first_day)

        # Keep only the tail of history that predict() needs to rebuild features.
        self._history = train[train["date"] > self._cutoff - pd.Timedelta(days=LOOKBACK_DAYS)]
        return self

    def _fit_bucket(self, source: pd.DataFrame, min_lag: int,
                    first_day: pd.Timestamp | None) -> lgb.Booster:
        names = feature_names(min_lag)
        # Count training rows up front (a row trains once it has min_lag days of history) so
        # the float32 design matrix is allocated once and filled chunk by chunk of series:
        # the full-width feature frame never exists at once (features never cross series).
        position = source.groupby("id", observed=True, sort=False).cumcount().to_numpy()
        trains = position >= min_lag
        if first_day is not None:
            trains &= (source["date"] > first_day).to_numpy()
        X = np.empty((int(trains.sum()), len(names)), dtype="float32")
        y = np.empty(len(X), dtype="float32")
        del position, trains

        ids = source["id"].unique()
        filled = 0
        for i in range(0, len(ids), self.chunk_series):
            part = add_features(source[source["id"].isin(ids[i:i + self.chunk_series])],
                                min_lag)
            keep = part[f"lag_{min_lag}"].notna()
            if first_day is not None:
                keep &= part["date"] > first_day
            part = part[keep]
            X[filled:filled + len(part)] = self._matrix(part, names)
            y[filled:filled + len(part)] = part["sales"].to_numpy(dtype="float32")
            filled += len(part)
            del part
        assert filled == len(X), "training row count mismatch"

        dataset = lgb.Dataset(X, label=y, feature_name=names,
                              categorical_feature=CATEGORICAL, free_raw_data=True).construct()
        del X, y  # LightGBM now holds its own compact binned copy
        return lgb.train(self.params, dataset, num_boost_round=self.num_boost_round)

    def predict(self, future: pd.DataFrame) -> pd.Series:
        horizon = (future["date"] - self._cutoff).dt.days
        if horizon.max() > self.max_horizon:
            raise ValueError(f"Horizon {horizon.max()}d exceeds the largest bucket "
                             f"({self.max_horizon}d): features would leak")

        # Only rebuild features for the requested series: keeps single-item API calls fast.
        history = self._history[self._history["id"].isin(future["id"].unique())]
        combined = pd.concat(
            [history, future.assign(sales=float("nan"), _row=future.index)],
            ignore_index=True,
        )
        for c in CATEGORICAL:  # concat can widen categories; keep the training set
            combined[c] = combined[c].astype("object")

        # Each forecast day goes to the first bucket that reaches it.
        bucket_of = np.searchsorted(self.horizon_buckets, horizon.to_numpy())
        out = pd.Series(np.nan, index=future.index, name="yhat")
        for b, min_lag in enumerate(self.horizon_buckets):
            rows = future.index[bucket_of == b]
            if len(rows) == 0:
                continue
            feats = add_features(combined, min_lag)
            feats = feats[feats["_row"].isin(rows)]
            yhat = self.boosters[min_lag].predict(self._matrix(feats, feature_names(min_lag)))
            out.loc[feats["_row"].astype("int64").to_numpy()] = yhat
        return out

    def _matrix(self, feats: pd.DataFrame, names: list[str]) -> np.ndarray:
        """float32 design matrix in `names` order. Categoricals become their code in the
        training categories; unseen values become NaN, which LightGBM treats as missing."""
        X = np.empty((len(feats), len(names)), dtype="float32")
        for j, col in enumerate(names):
            if col in self._categories:
                codes = pd.Categorical(feats[col].astype("object"),
                                       categories=self._categories[col]).codes
                X[:, j] = np.where(codes < 0, np.nan, codes)
            else:
                X[:, j] = feats[col].to_numpy(dtype="float32", na_value=np.nan)
        return X

    def feature_importance(self) -> pd.DataFrame:
        frames = [
            pd.DataFrame({
                "bucket": f"days 1-{b}" if len(self.horizon_buckets) == 1 or i == 0
                else f"days {self.horizon_buckets[i - 1] + 1}-{b}",
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            })
            for i, (b, booster) in enumerate(sorted(self.boosters.items()))
        ]
        return pd.concat(frames).sort_values("gain", ascending=False).reset_index(drop=True)

    def get_params(self) -> dict[str, Any]:
        return {"num_boost_round": self.num_boost_round, "train_days": self.train_days,
                "horizon_buckets": list(self.horizon_buckets), **self.params}
