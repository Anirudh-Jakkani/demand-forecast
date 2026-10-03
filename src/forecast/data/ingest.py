"""Turn raw M5 CSVs into one long, typed Parquet table: one row per (series, day).

Output columns:
    id, item_id, dept_id, store_id, date, sales,
    wday, month, year, event_name_1, event_type_1, snap, sell_price
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SALES_FILE = "sales_train_evaluation.csv"
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5_long(
    raw_dir: Path,
    state: str = "CA",
    category: str = "FOODS",
    start_date: str | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    calendar = pd.read_csv(raw_dir / CALENDAR_FILE, parse_dates=["date"])
    if start_date is not None:
        calendar = calendar[calendar["date"] >= pd.Timestamp(start_date)]
    day_cols = calendar["d"].tolist()

    # Only read the day columns we keep; the full file is ~120 MB wide.
    header = pd.read_csv(raw_dir / SALES_FILE, nrows=0).columns
    usecols = ID_COLS + [d for d in day_cols if d in header]
    sales = pd.read_csv(raw_dir / SALES_FILE, usecols=usecols)
    sales = sales[(sales["state_id"] == state) & (sales["cat_id"] == category)]
    if sales.empty:
        raise ValueError(f"No series found for state={state!r}, category={category!r}")
    log.info("Selected %d series", len(sales))

    long = sales.melt(id_vars=ID_COLS, var_name="d", value_name="sales")
    long = long.drop(columns=["cat_id", "state_id"])

    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                "event_name_1", "event_type_1", f"snap_{state}"]
    long = long.merge(calendar[cal_cols], on="d", how="left")
    long = long.rename(columns={f"snap_{state}": "snap"})

    prices = pd.read_csv(raw_dir / PRICES_FILE)
    prices = prices[prices["store_id"].isin(long["store_id"].unique())]
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # No price means the item was not on the shelf yet: those zeros are not real demand.
    before = len(long)
    long = long.dropna(subset=["sell_price"])
    log.info("Dropped %d pre-launch rows", before - len(long))

    long = long.drop(columns=["d", "wm_yr_wk"])
    return _optimize_dtypes(long).sort_values(["id", "date"]).reset_index(drop=True)


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["id", "item_id", "dept_id", "store_id", "event_name_1", "event_type_1"]:
        df[col] = df[col].astype("category")
    df["sales"] = df["sales"].astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")
    for col in ["wday", "month", "snap"]:
        df[col] = df[col].astype("int8")
    df["year"] = df["year"].astype("int16")
    return df


def run_ingest(raw_dir: Path, out_path: Path, **kwargs) -> pd.DataFrame:
    df = load_m5_long(raw_dir, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Wrote %s rows x %s series to %s", len(df), df["id"].nunique(), out_path)
    return df
