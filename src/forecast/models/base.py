"""Common interface every forecaster implements, so all models share one backtest harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ForecastModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> ForecastModel:
        """Train on the long table (id, date, sales, + exogenous columns)."""

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Return one forecast per row of `future`, aligned to its index.

        `future` has every column of the training table except `sales`; known-ahead
        features like price, calendar and events are allowed.
        """

    def get_params(self) -> dict[str, Any]:
        return {}
