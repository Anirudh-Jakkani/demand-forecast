"""Lightweight data-quality checks for the processed long table."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"id", "item_id", "store_id", "date", "sales", "sell_price"}


class DataValidationError(ValueError):
    pass


def validate_long(df: pd.DataFrame) -> None:
    """Raise DataValidationError listing every problem found."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    problems = []
    if (df["sales"] < 0).any():
        problems.append(f"{int((df['sales'] < 0).sum())} rows with negative sales")
    if df["sales"].isna().any():
        problems.append(f"{int(df['sales'].isna().sum())} rows with null sales")
    if df.duplicated(["id", "date"]).any():
        problems.append(f"{int(df.duplicated(['id', 'date']).sum())} duplicate (id, date) rows")

    # Each series must be a contiguous daily run from its first to last date.
    span = df.groupby("id", observed=True)["date"].agg(["min", "max", "count"])
    expected = (span["max"] - span["min"]).dt.days + 1
    gappy = span.index[span["count"] != expected]
    if len(gappy):
        problems.append(f"{len(gappy)} series with date gaps, e.g. {list(gappy[:3])}")

    if problems:
        raise DataValidationError("; ".join(problems))
