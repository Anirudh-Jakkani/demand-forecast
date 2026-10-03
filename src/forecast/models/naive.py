"""Baselines. Every fancier model has to beat these to earn its complexity."""

from __future__ import annotations

import pandas as pd

from forecast.models.base import ForecastModel


class SeasonalNaive(ForecastModel):
    """Forecast = the value on the same weekday in the last `season` days of history."""

    name = "seasonal_naive"

    def __init__(self, season: int = 7):
        self.season = season
        self._last: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        cutoff = train["date"].max()
        recent = train[train["date"] > cutoff - pd.Timedelta(days=self.season)]
        self._last = (
            recent.assign(slot=self._slot(recent["date"], cutoff))
            [["id", "slot", "sales"]]
            .rename(columns={"sales": "yhat"})
        )
        self._cutoff = cutoff
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        keys = pd.DataFrame({
            "id": future["id"].to_numpy(),
            "slot": self._slot(future["date"], self._cutoff).to_numpy(),
        })
        out = keys.merge(self._last, on=["id", "slot"], how="left")
        return pd.Series(out["yhat"].fillna(0.0).to_numpy(), index=future.index, name="yhat")

    def _slot(self, dates: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
        return (dates - cutoff).dt.days % self.season

    def get_params(self):
        return {"season": self.season}


class MovingAverage(ForecastModel):
    """Forecast = flat mean of each series over the last `window` days."""

    name = "moving_average"

    def __init__(self, window: int = 28):
        self.window = window
        self._means: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> MovingAverage:
        cutoff = train["date"].max()
        recent = train[train["date"] > cutoff - pd.Timedelta(days=self.window)]
        self._means = recent.groupby("id", observed=True)["sales"].mean()
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        yhat = future["id"].map(self._means).astype(float).fillna(0.0)
        return yhat.rename("yhat")

    def get_params(self):
        return {"window": self.window}
