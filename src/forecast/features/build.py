"""Feature engineering for the global LightGBM model.

Leakage rule: every sales-derived feature uses lags >= `min_lag` days. A row whose forecast
horizon is <= min_lag then only ever reads sales from on or before the cutoff, so a model
can predict that whole horizon directly (no recursive feeding of its own forecasts).
Horizon-bucketed models use a smaller min_lag for near days (fresher signal) and the full
28 for far days; see models.lgbm.

Price, calendar and event features are known ahead of time and need no lag.
Assumes each series is a contiguous daily run (enforced by data.validate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_LAG = 28                            # default: one model for the whole 28-day horizon
ROLL_WINDOWS = (7, 28, 56, 112)         # rolling stats over the min-lag series
YEAR_LAG = 364                          # same weekday last year

CATEGORICAL = ["item_id", "dept_id", "store_id", "event_name_1", "event_type_1"]
CALENDAR = ["wday", "month", "year", "dayofmonth", "weekofyear", "snap"]
PRICE = ["sell_price", "price_norm", "price_change_7", "price_momentum_28"]


def sales_lags(min_lag: int = MIN_LAG) -> tuple[int, ...]:
    """Weekly-aligned lags starting at min_lag, plus last year."""
    return tuple(min_lag + 7 * k for k in range(5)) + (YEAR_LAG,)


def sales_features(min_lag: int = MIN_LAG) -> list[str]:
    return (
        [f"lag_{lag}" for lag in sales_lags(min_lag)]
        + [f"rmean_{min_lag}_{w}" for w in ROLL_WINDOWS]
        + [f"rstd_{min_lag}_28", f"zero_share_{min_lag}_28"]
    )


def feature_names(min_lag: int = MIN_LAG) -> list[str]:
    return CATEGORICAL + CALENDAR + PRICE + sales_features(min_lag)


LAGS = sales_lags()
SALES = sales_features()
FEATURES = feature_names()

PRICE_WINDOW = 364                      # price_norm = price / max price over the last year

# Days of history to keep before the cutoff so every in-horizon feature is computable
# exactly as it was in training (+ MIN_LAG margin for the horizon itself). The default
# (largest) min_lag needs the most history, so this covers every bucket.
LOOKBACK_DAYS = max(max(LAGS), MIN_LAG + max(ROLL_WINDOWS), PRICE_WINDOW) + MIN_LAG


def add_features(df: pd.DataFrame, min_lag: int = MIN_LAG) -> pd.DataFrame:
    """Return a copy of `df` (sorted by id, date) with `feature_names(min_lag)` added.

    Future rows may carry NaN sales; they only feed lags that no in-horizon row reads.
    Every series is processed in one vectorized pass (no per-series Python), and features
    are produced directly as float32 to keep peak memory low on the full M5 slice.
    """
    df = df.sort_values(["id", "date"], ignore_index=True)
    start = _group_starts(df["id"])

    sales = df["sales"].to_numpy(dtype="float64", na_value=np.nan)
    for lag in sales_lags(min_lag):
        df[f"lag_{lag}"] = _lag(sales, lag, start)
    base = df[f"lag_{min_lag}"].to_numpy(dtype="float64")
    for w in ROLL_WINDOWS:
        df[f"rmean_{min_lag}_{w}"] = _rolling_mean(base, w, 1, start)
    df[f"rstd_{min_lag}_28"] = _rolling_std(base, 28, 2, start)
    is_zero = np.where(np.isnan(base), np.nan, (base == 0).astype("float64"))
    df[f"zero_share_{min_lag}_28"] = _rolling_mean(is_zero, 28, 1, start)

    # Calendar.
    df["dayofmonth"] = df["date"].dt.day.astype("int8")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int8").to_numpy()

    # Price: level relative to the item's recent range, and recent changes (promos show here).
    price = df["sell_price"].to_numpy(dtype="float64", na_value=np.nan)
    # Rolling (not all-time) max so training and inference see the same definition.
    df["price_norm"] = (price / _rolling_max(price, PRICE_WINDOW, start)).astype("float32")
    df["price_change_7"] = (price / _lag(price, 7, start).astype("float64") - 1) \
        .astype("float32")
    df["price_momentum_28"] = (price / _rolling_mean(price, 28, 1, start).astype("float64")) \
        .astype("float32")
    return df


# ---------- vectorized per-series window helpers (rows sorted by id, contiguous days) ----------

def _group_starts(ids: pd.Series) -> np.ndarray:
    """Index of the first row of each row's series."""
    codes = pd.factorize(ids)[0] if not isinstance(ids.dtype, pd.CategoricalDtype) \
        else ids.cat.codes.to_numpy()
    idx = np.arange(len(codes))
    new = np.ones(len(codes), dtype=bool)
    new[1:] = codes[1:] != codes[:-1]
    return np.maximum.accumulate(np.where(new, idx, 0))


def _lag(x: np.ndarray, k: int, start: np.ndarray) -> np.ndarray:
    src = np.arange(len(x)) - k
    ok = src >= start
    out = np.full(len(x), np.nan, dtype="float32")
    out[ok] = x[src[ok]]
    return out


def _window_bounds(n: int, w: int, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = np.arange(1, n + 1)
    return np.maximum(hi - w, start), hi


def _rolling_mean(x: np.ndarray, w: int, min_periods: int, start: np.ndarray) -> np.ndarray:
    """pandas `rolling(w, min_periods).mean()` within each series, via running sums."""
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    lo, hi = _window_bounds(len(x), w, start)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count < min_periods] = np.nan
    return out.astype("float32")


def _rolling_std(x: np.ndarray, w: int, min_periods: int, start: np.ndarray) -> np.ndarray:
    """pandas `rolling(w, min_periods).std()` (ddof=1) within each series."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(v)))
    csq = np.concatenate(([0.0], np.cumsum(v * v)))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    lo, hi = _window_bounds(len(x), w, start)
    s, sq, n = csum[hi] - csum[lo], csq[hi] - csq[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sq - s * s / n) / (n - 1)
    out = np.sqrt(np.clip(var, 0.0, None))
    out[n < min_periods] = np.nan
    return out.astype("float32")


def _rolling_max(x: np.ndarray, w: int, start: np.ndarray) -> np.ndarray:
    """pandas `rolling(w, 1).max()` within each series, NaN-ignoring.

    Max has no running-sum trick, so use doubling: after the loop, m[i] is the max over the
    last `k` rows (k = largest power of two <= w, clipped at the series start). Any window of
    length w is covered by two overlapping length-k windows. O(n log w) time, O(n) memory.
    """
    n = len(x)
    idx = np.arange(n)
    m = x.astype("float64", copy=True)
    k = 1
    while k * 2 <= w:
        prev = idx - k
        ok = prev >= start
        shifted = np.full(n, np.nan)
        shifted[ok] = m[prev[ok]]
        m = np.fmax(m, shifted)
        k *= 2
    lo = np.maximum(idx - w + 1, start)              # first row of each window
    j = np.minimum(idx, lo + k - 1)                   # window of length k starting at lo
    return np.fmax(m, m[j])
