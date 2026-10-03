from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class SeriesKey(BaseModel):
    store_id: str = Field(examples=["CA_1"])
    item_id: str = Field(examples=["FOODS_3_090"])


class PredictRequest(BaseModel):
    items: list[SeriesKey] = Field(min_length=1, max_length=1000)
    horizon: int = Field(28, ge=1, description="days ahead, up to the model's max horizon")


class DailyForecast(BaseModel):
    date: dt.date
    yhat: float


class SeriesForecast(BaseModel):
    store_id: str
    item_id: str
    forecast: list[DailyForecast]


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    model_type: str
    forecast_start: dt.date
    forecasts: list[SeriesForecast]


class ModelInfo(BaseModel):
    registered_name: str
    version: str
    model_type: str
    data_end: dt.date
    max_horizon: int
    n_series: int
    cv_wape: float | None
    promotion_reason: str | None
    loaded_at: dt.datetime
