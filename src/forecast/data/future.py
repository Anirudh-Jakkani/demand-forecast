"""Build the "future frame" a deployed model forecasts on: one row per (series, day)
for the days after the last observed sale, carrying the known-ahead inputs
(calendar, events, SNAP, planned prices) but no sales.

M5's calendar.csv and sell_prices.csv extend 28 days past the sales data, so on real
data these are the true planned values. Where they are missing (e.g. the synthetic
sample) calendar fields are derived from the date and prices are carried forward.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecast.data.ingest import CALENDAR_FILE, PRICES_FILE

SERIES_COLS = ["id", "item_id", "dept_id", "store_id"]


def build_future_frame(
    df: pd.DataFrame, raw_dir: Path | None, horizon: int, state: str
) -> pd.DataFrame:
    cutoff = df["date"].max()
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Only series still selling at the cutoff get a forecast.
    last = df.loc[df["date"] == cutoff, SERIES_COLS + ["sell_price"]]
    last = last.rename(columns={"sell_price": "last_price"})
    grid = last.merge(pd.DataFrame({"date": dates}), how="cross")

    calendar = _future_calendar(raw_dir, dates, state)
    grid = grid.merge(calendar.drop(columns="wm_yr_wk"), on="date", how="left")
    grid = _attach_prices(grid, calendar, raw_dir)

    grid = grid.drop(columns="last_price")
    for col in SERIES_COLS + ["event_name_1", "event_type_1"]:
        grid[col] = grid[col].astype("category")
    for col in ["wday", "month", "snap"]:
        grid[col] = grid[col].astype("int8")
    grid["year"] = grid["year"].astype("int16")
    grid["sell_price"] = grid["sell_price"].astype("float32")
    return grid.sort_values(["id", "date"]).reset_index(drop=True)


def _future_calendar(raw_dir: Path | None, dates: pd.DatetimeIndex, state: str) -> pd.DataFrame:
    derived = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": pd.NA,
        "wday": (dates.dayofweek + 2) % 7 + 1,  # M5: Saturday=1 ... Friday=7
        "month": dates.month,
        "year": dates.year,
        "event_name_1": None,
        "event_type_1": None,
        "snap": 0,
    })
    path = Path(raw_dir) / CALENDAR_FILE if raw_dir else None
    if path is None or not path.exists():
        return derived

    cal = pd.read_csv(path, parse_dates=["date"])
    cal = cal[cal["date"].isin(dates)].rename(columns={f"snap_{state}": "snap"})
    cal = cal[derived.columns]
    # Prefer the real calendar, fall back to derived values for days it doesn't cover.
    missing = derived[~derived["date"].isin(cal["date"])]
    return pd.concat([cal, missing], ignore_index=True).sort_values("date")


def _attach_prices(
    grid: pd.DataFrame, calendar: pd.DataFrame, raw_dir: Path | None
) -> pd.DataFrame:
    grid["sell_price"] = float("nan")
    path = Path(raw_dir) / PRICES_FILE if raw_dir else None
    weeks = calendar.dropna(subset=["wm_yr_wk"])
    if path is not None and path.exists() and len(weeks):
        prices = pd.read_csv(path)
        prices = prices[prices["wm_yr_wk"].isin(weeks["wm_yr_wk"].astype(int))]
        planned = (
            weeks[["date", "wm_yr_wk"]].astype({"wm_yr_wk": int})
            .merge(prices, on="wm_yr_wk")
            .drop(columns="wm_yr_wk")
        )
        keys = ["store_id", "item_id", "date"]
        grid = grid.astype({"store_id": "object", "item_id": "object"})
        grid = grid.drop(columns="sell_price").merge(planned, on=keys, how="left")
    grid["sell_price"] = grid["sell_price"].fillna(grid["last_price"])
    return grid
