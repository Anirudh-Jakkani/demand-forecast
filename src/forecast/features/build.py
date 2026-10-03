"""Feature engineering for the global LightGBM model.

Leakage rule: every sales-derived feature uses lags >= MIN_LAG days. As long as the
forecast horizon is <= MIN_LAG, a row in the forecast window only ever reads sales
from on or before the cutoff, so one model can predict the whole horizon directly
(no recursive feeding of its own forecasts).

Price, calendar and event features are known ahead of time and need no lag.
Assumes each series is a contiguous daily run (enforced by data.validate).
"""

from __future__ import annotations

import pandas as pd

MIN_LAG = 28
LAGS = (28, 35, 42, 49, 56, 364)        # 364 = same weekday last year
ROLL_WINDOWS = (7, 28, 56, 112)         # rolling stats over the lag-28 series

CATEGORICAL = ["item_id", "dept_id", "store_id", "event_name_1", "event_type_1"]
CALENDAR = ["wday", "month", "year", "dayofmonth", "weekofyear", "snap"]
PRICE = ["sell_price", "price_norm", "price_change_7", "price_momentum_28"]
SALES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"rmean_{MIN_LAG}_{w}" for w in ROLL_WINDOWS]
    + [f"rstd_{MIN_LAG}_28", f"zero_share_{MIN_LAG}_28"]
)
FEATURES = CATEGORICAL + CALENDAR + PRICE + SALES

PRICE_WINDOW = 364                      # price_norm = price / max price over the last year

# Days of history to keep before the cutoff so every in-horizon feature is computable
# exactly as it was in training (+ MIN_LAG margin for the horizon itself).
LOOKBACK_DAYS = max(max(LAGS), MIN_LAG + max(ROLL_WINDOWS), PRICE_WINDOW) + MIN_LAG


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` (sorted by id, date) with all FEATURES columns added.

    Future rows may carry NaN sales; they only feed lags that no in-horizon row reads.
    """
    df = df.sort_values(["id", "date"]).copy()
    by_id = df.groupby("id", observed=True, sort=False)

    # Sales lags and rolling stats, all anchored at MIN_LAG.
    for lag in LAGS:
        df[f"lag_{lag}"] = by_id["sales"].shift(lag)
    base = df[f"lag_{MIN_LAG}"].groupby(df["id"], observed=True, sort=False)
    for w in ROLL_WINDOWS:
        df[f"rmean_{MIN_LAG}_{w}"] = base.transform(lambda s, w=w: s.rolling(w, 1).mean())
    df[f"rstd_{MIN_LAG}_28"] = base.transform(lambda s: s.rolling(28, 2).std())
    is_zero = (df[f"lag_{MIN_LAG}"] == 0).astype("float32").where(df[f"lag_{MIN_LAG}"].notna())
    df[f"zero_share_{MIN_LAG}_28"] = is_zero.groupby(df["id"], observed=True, sort=False) \
        .transform(lambda s: s.rolling(28, 1).mean())

    # Calendar.
    df["dayofmonth"] = df["date"].dt.day.astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int8")

    # Price: level relative to the item's own range, and recent changes (promos show up here).
    price = by_id["sell_price"]
    # Rolling (not all-time) max so training and inference see the same definition.
    df["price_norm"] = df["sell_price"] / price.transform(
        lambda s: s.rolling(PRICE_WINDOW, 1).max()
    )
    df["price_change_7"] = df["sell_price"] / price.shift(7) - 1
    df["price_momentum_28"] = df["sell_price"] / price.transform(
        lambda s: s.rolling(28, 1).mean()
    )

    for col in SALES + PRICE[1:]:
        df[col] = df[col].astype("float32")
    return df
