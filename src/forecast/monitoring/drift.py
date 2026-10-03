"""Data drift with Evidently: compare the window the champion was trained on with the
most recent window of data.

Columns are row-level (one row per series-day): demand, price level and price changes.
Calendar fields are left out on purpose: they "drift" every season by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DRIFT_COLUMNS = ["sales", "sell_price", "price_change_7"]


@dataclass
class DriftResult:
    share: float                                   # fraction of columns that drifted
    drifted_columns: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    report_path: Path | None = None


def drift_windows(
    df: pd.DataFrame, reference_end: pd.Timestamp, current_end: pd.Timestamp, window_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["date"] <= current_end].sort_values(["id", "date"])
    df = df.assign(
        price_change_7=df["sell_price"]
        / df.groupby("id", observed=True)["sell_price"].shift(7) - 1
    )

    def window(end):
        start = end - pd.Timedelta(days=window_days - 1)
        return df.loc[(df["date"] >= start) & (df["date"] <= end), DRIFT_COLUMNS] \
                 .astype("float64").reset_index(drop=True)

    return window(reference_end), window(current_end)


def run_drift(reference: pd.DataFrame, current: pd.DataFrame,
              html_path: Path | None = None) -> DriftResult:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=DRIFT_COLUMNS)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    scores, drifted = {}, []
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        column = cfg.get("column")
        if column is None or "threshold" not in cfg:
            continue
        value, threshold = float(m["value"]), float(cfg["threshold"])
        scores[column] = value
        # Statistical tests report p-values (drift when small); distances drift when large.
        is_p_value = "p_value" in str(cfg.get("method", "")).lower() or "test" in str(
            cfg.get("method", "")).lower()
        if (value < threshold) if is_p_value else (value >= threshold):
            drifted.append(column)

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))

    share = len(drifted) / len(scores) if scores else 0.0
    return DriftResult(share, drifted, scores, html_path)
