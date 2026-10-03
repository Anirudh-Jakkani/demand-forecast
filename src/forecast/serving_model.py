"""The deployable artifact: a fitted ForecastModel bundled with its future frame.

Logged to MLflow as a pyfunc so it can be loaded by registry alias
(`models:/<name>@champion`) without the caller knowing which model type it is.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd

from forecast.models.base import ForecastModel


class ForecastBundle(mlflow.pyfunc.PythonModel):
    def __init__(self, model: ForecastModel, future_frame: pd.DataFrame, model_name: str):
        self.model = model
        self.future_frame = future_frame
        self.model_name = model_name
        self.cutoff: pd.Timestamp = future_frame["date"].min() - pd.Timedelta(days=1)
        self.max_horizon: int = int(future_frame["date"].nunique())
        self._keys = (
            future_frame[["store_id", "item_id", "id"]].drop_duplicates()
            .astype(str).set_index(["store_id", "item_id"])["id"]
        )

    @property
    def n_series(self) -> int:
        return len(self._keys)

    def series_id(self, store_id: str, item_id: str) -> str | None:
        return self._keys.get((store_id, item_id))

    def forecast(self, ids: list[str], horizon: int) -> pd.DataFrame:
        if not 1 <= horizon <= self.max_horizon:
            raise ValueError(f"horizon must be between 1 and {self.max_horizon}")
        last_day = self.cutoff + pd.Timedelta(days=horizon)
        fut = self.future_frame
        fut = fut[fut["id"].isin(ids) & (fut["date"] <= last_day)]
        yhat = self.model.predict(fut)
        out = fut[["id", "store_id", "item_id", "date"]].astype({
            "id": str, "store_id": str, "item_id": str,
        })
        return out.assign(yhat=yhat.clip(lower=0).to_numpy()).reset_index(drop=True)

    def predict(self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None):
        """pyfunc entry point. Input columns: store_id, item_id; params: {"horizon": int}."""
        horizon = int((params or {}).get("horizon", self.max_horizon))
        keys = zip(model_input["store_id"], model_input["item_id"], strict=True)
        ids = [self.series_id(s, i) for s, i in keys]
        return self.forecast([i for i in ids if i is not None], horizon)
