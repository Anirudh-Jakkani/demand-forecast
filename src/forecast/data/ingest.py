"""Turn raw M5 CSVs into one long, typed Parquet table: one row per (series, day).

Output columns:
    id, item_id, dept_id, store_id, date, sales,
    wday, month, year, event_name_1, event_type_1, snap, sell_price
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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
    """Built straight from arrays (no melt/merge) so the full M5 slice fits in a few
    hundred MB: the wide file is read in filtered int16 chunks, calendar fields are tiled
    per series, and prices come from a (series x week) lookup table."""
    raw_dir = Path(raw_dir)
    calendar = pd.read_csv(raw_dir / CALENDAR_FILE, parse_dates=["date"])
    if start_date is not None:
        calendar = calendar[calendar["date"] >= pd.Timestamp(start_date)]

    # Only read the day columns we keep, as int16, filtering rows chunk by chunk.
    header = pd.read_csv(raw_dir / SALES_FILE, nrows=0).columns
    calendar = calendar[calendar["d"].isin(header)].reset_index(drop=True)
    day_cols = calendar["d"].tolist()
    chunks = pd.read_csv(raw_dir / SALES_FILE, usecols=ID_COLS + day_cols,
                         dtype={d: "int16" for d in day_cols}, chunksize=5000)
    sales = pd.concat(
        [c[(c["state_id"] == state) & (c["cat_id"] == category)] for c in chunks],
        ignore_index=True,
    )
    if sales.empty:
        raise ValueError(f"No series found for state={state!r}, category={category!r}")
    sales = sales.sort_values("id", ignore_index=True)
    n_series, n_days = len(sales), len(day_cols)
    log.info("Selected %d series x %d days", n_series, n_days)

    def per_series(col: str) -> pd.Categorical:
        cats = pd.Categorical(sales[col])
        return pd.Categorical.from_codes(np.repeat(cats.codes, n_days), cats.categories)

    def per_day(values) -> np.ndarray:
        return np.tile(np.asarray(values), n_series)

    def per_day_cat(col: str) -> pd.Categorical:
        cats = pd.Categorical(calendar[col])
        return pd.Categorical.from_codes(per_day(cats.codes), cats.categories)

    long = pd.DataFrame({
        "id": per_series("id"),
        "item_id": per_series("item_id"),
        "dept_id": per_series("dept_id"),
        "store_id": per_series("store_id"),
        "date": per_day(calendar["date"].to_numpy()),
        "sales": sales[day_cols].to_numpy(dtype="float32").ravel(),
        "wday": per_day(calendar["wday"].to_numpy(dtype="int8")),
        "month": per_day(calendar["month"].to_numpy(dtype="int8")),
        "year": per_day(calendar["year"].to_numpy(dtype="int16")),
        "event_name_1": per_day_cat("event_name_1"),
        "event_type_1": per_day_cat("event_type_1"),
        "snap": per_day(calendar[f"snap_{state}"].to_numpy(dtype="int8")),
        "sell_price": _price_matrix(raw_dir, sales, calendar).ravel(),
    })

    # No price means the item was not on the shelf yet: those zeros are not real demand.
    on_shelf = long["sell_price"].notna()
    log.info("Dropped %d pre-launch rows", int((~on_shelf).sum()))
    return long[on_shelf].reset_index(drop=True)


def _price_matrix(raw_dir: Path, sales: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """(n_series, n_days) float32 sell prices; NaN where the item had no price that week."""
    weeks = pd.Index(calendar["wm_yr_wk"].unique())
    series = pd.MultiIndex.from_arrays([sales["store_id"], sales["item_id"]])
    prices = pd.read_csv(raw_dir / PRICES_FILE,
                         dtype={"store_id": "category", "item_id": "category",
                                "wm_yr_wk": "int32", "sell_price": "float32"})
    prices = prices[prices["wm_yr_wk"].isin(weeks)]
    row = series.get_indexer(pd.MultiIndex.from_arrays(
        [prices["store_id"].astype(str), prices["item_id"].astype(str)]))
    found = row >= 0
    by_week = np.full((len(sales), len(weeks)), np.nan, dtype="float32")
    by_week[row[found], weeks.get_indexer(prices["wm_yr_wk"])[found]] = \
        prices["sell_price"].to_numpy()[found]
    return by_week[:, weeks.get_indexer(calendar["wm_yr_wk"])]


def read_processed(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the processed table without Arrow keeping a second copy alive.

    Plain `pd.read_parquet` leaves the Arrow buffers and allocator slack resident
    (~3x the frame size); on an 8 GB machine that headroom matters for training.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    pa.default_memory_pool().release_unused()
    return df


def run_ingest(raw_dir: Path, out_path: Path, **kwargs) -> pd.DataFrame:
    df = load_m5_long(raw_dir, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Wrote %s rows x %s series to %s", len(df), df["id"].nunique(), out_path)
    return df
