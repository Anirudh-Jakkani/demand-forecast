"""Live accuracy: forecasts that were logged, scored against sales that arrived since."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.evaluation.metrics import bias, mae, wape


def join_forecasts_to_actuals(
    predictions: pd.DataFrame, actuals: pd.DataFrame, as_of: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    """Forecast vs actual for days in (as_of - window_days, as_of].

    When several forecasts exist for the same series and day, keep the freshest one made
    before that day, i.e. the forecast a planner would actually have been using.
    """
    start = as_of - pd.Timedelta(days=window_days - 1)
    p = predictions[
        (predictions["date"] >= start)
        & (predictions["date"] <= as_of)
        & (predictions["cutoff"] < predictions["date"])
    ]
    p = (p.sort_values(["cutoff", "created_at"])
          .drop_duplicates(["series_id", "date"], keep="last"))

    a = actuals.loc[(actuals["date"] >= start) & (actuals["date"] <= as_of),
                    ["id", "date", "sales"]]
    a = a.assign(series_id=a["id"].astype(str)).drop(columns="id")
    joined = p.merge(a, on=["series_id", "date"], how="inner")
    joined["horizon"] = (joined["date"] - joined["cutoff"]).dt.days
    return joined


def performance_summary(joined: pd.DataFrame) -> dict[str, float]:
    if joined.empty:
        return {"n_points": 0, "live_wape": np.nan, "live_bias": np.nan, "live_mae": np.nan,
                "wape_h1_7": np.nan, "wape_h8_28": np.nan}
    y, yhat = joined["sales"].to_numpy(float), joined["yhat"].to_numpy(float)
    short = joined["horizon"] <= 7
    return {
        "n_points": int(len(joined)),
        "live_wape": wape(y, yhat),
        "live_bias": bias(y, yhat),
        "live_mae": mae(y, yhat),
        "wape_h1_7": wape(y[short], yhat[short]) if short.any() else np.nan,
        "wape_h8_28": wape(y[~short], yhat[~short]) if (~short).any() else np.nan,
    }
