"""Generate tiny M5-shaped raw CSVs so the pipeline runs without the real dataset.

Includes a non-CA store and a non-FOODS item so the ingest filters get exercised,
plus late-launched items to exercise the pre-launch drop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast.data.ingest import CALENDAR_FILE, PRICES_FILE, SALES_FILE


def make_synthetic_m5(
    out_dir: Path,
    n_days: int = 500,
    items_per_dept: int = 5,
    start: str = "2015-01-01",
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start, periods=n_days, freq="D")
    calendar = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": (11101 + np.arange(n_days) // 7),
        "weekday": dates.day_name(),
        "wday": (dates.dayofweek + 2) % 7 + 1,  # M5: Saturday=1 ... Friday=7
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(n_days)],
        "event_name_1": np.where((dates.month == 12) & (dates.day == 25), "Christmas", None),
        "event_type_1": np.where((dates.month == 12) & (dates.day == 25), "National", None),
        "event_name_2": None,
        "event_type_2": None,
        "snap_CA": (dates.day <= 10).astype(int),
        "snap_TX": ((dates.day >= 5) & (dates.day <= 15)).astype(int),
        "snap_WI": ((dates.day >= 10) & (dates.day <= 20)).astype(int),
    })

    stores = [("CA_1", "CA"), ("CA_2", "CA"), ("TX_1", "TX")]
    depts = [("FOODS_1", "FOODS"), ("FOODS_2", "FOODS"), ("HOBBIES_1", "HOBBIES")]
    weekly = np.array([1.0, 0.9, 0.85, 0.9, 1.1, 1.4, 1.5])  # Mon..Sun multipliers
    dow = dates.dayofweek.to_numpy()

    sales_rows, price_rows = [], []
    for store_id, state_id in stores:
        for dept_id, cat_id in depts:
            for k in range(1, items_per_dept + 1):
                item_id = f"{dept_id}_{k:03d}"
                base = rng.gamma(2.0, 1.5)
                trend = 1 + 0.0004 * np.arange(n_days)
                launch = int(rng.integers(0, n_days // 4)) if k == items_per_dept else 0
                y = rng.poisson(base * weekly[dow] * trend).astype(int)
                y[:launch] = 0
                sales_rows.append({
                    "id": f"{item_id}_{store_id}_evaluation", "item_id": item_id,
                    "dept_id": dept_id, "cat_id": cat_id, "store_id": store_id,
                    "state_id": state_id, **{f"d_{i + 1}": v for i, v in enumerate(y)},
                })
                price = round(float(rng.uniform(1, 10)), 2)
                for wk in calendar["wm_yr_wk"].unique()[launch // 7:]:
                    price_rows.append({"store_id": store_id, "item_id": item_id,
                                       "wm_yr_wk": wk, "sell_price": price})

    pd.DataFrame(sales_rows).to_csv(out_dir / SALES_FILE, index=False)
    calendar.to_csv(out_dir / CALENDAR_FILE, index=False)
    pd.DataFrame(price_rows).to_csv(out_dir / PRICES_FILE, index=False)
    return out_dir
