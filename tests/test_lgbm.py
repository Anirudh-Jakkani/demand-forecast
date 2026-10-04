import numpy as np
import pandas as pd
import pytest

from forecast.evaluation.backtest import make_folds, run_backtest, split
from forecast.features.build import FEATURES, MIN_LAG, add_features
from forecast.models import LGBMForecaster, SeasonalNaive

FAST = {"num_boost_round": 50, "num_leaves": 15, "min_data_in_leaf": 20, "train_days": None}


def _reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """The original, obviously-correct pandas implementation (slow, memory-hungry)."""
    from forecast.features.build import LAGS, PRICE_WINDOW, ROLL_WINDOWS

    df = df.sort_values(["id", "date"], ignore_index=True)
    by_id = df.groupby("id", observed=True, sort=False)
    out = pd.DataFrame(index=df.index)
    for lag in LAGS:
        out[f"lag_{lag}"] = by_id["sales"].shift(lag)
    base = out[f"lag_{MIN_LAG}"].groupby(df["id"], observed=True, sort=False)
    for w in ROLL_WINDOWS:
        out[f"rmean_{MIN_LAG}_{w}"] = base.transform(lambda s, w=w: s.rolling(w, 1).mean())
    out[f"rstd_{MIN_LAG}_28"] = base.transform(lambda s: s.rolling(28, 2).std())
    is_zero = (out[f"lag_{MIN_LAG}"] == 0).astype(float).where(out[f"lag_{MIN_LAG}"].notna())
    out[f"zero_share_{MIN_LAG}_28"] = is_zero.groupby(df["id"], observed=True, sort=False) \
        .transform(lambda s: s.rolling(28, 1).mean())
    price = by_id["sell_price"]
    out["price_norm"] = df["sell_price"] / price.transform(
        lambda s: s.rolling(PRICE_WINDOW, 1).max())
    out["price_change_7"] = df["sell_price"] / price.shift(7) - 1
    out["price_momentum_28"] = df["sell_price"] / price.transform(
        lambda s: s.rolling(28, 1).mean())
    return out


def test_vectorized_features_match_reference(long_df):
    """The vectorized builder must reproduce the reference pandas features exactly
    (up to float32 rounding), including series that launch late and NaN sales rows."""
    df = long_df.copy()
    df.loc[df.sample(frac=0.02, random_state=0).index, "sales"] = np.nan  # future-like gaps
    fast = add_features(df)
    ref = _reference_features(df)
    for col in ref.columns:
        np.testing.assert_allclose(fast[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float),
                                   rtol=1e-5, atol=1e-5, equal_nan=True, err_msg=col)


def test_features_need_only_lookback_days_of_history(tmp_path):
    """LGBMForecaster.fit builds features on a trimmed window; rows at least LOOKBACK_DAYS
    into that window must match features built on the full history."""
    from forecast.data.ingest import load_m5_long
    from forecast.data.synthetic import make_synthetic_m5
    from forecast.features.build import LOOKBACK_DAYS

    long_df = load_m5_long(make_synthetic_m5(tmp_path, n_days=520, items_per_dept=2))
    window_start = long_df["date"].min() + pd.Timedelta(days=60)
    trimmed = long_df[long_df["date"] >= window_start]
    full = add_features(long_df).set_index(["id", "date"])
    part = add_features(trimmed).set_index(["id", "date"])
    rows = part.index.get_level_values("date") >= window_start + pd.Timedelta(days=LOOKBACK_DAYS)
    assert rows.any()
    pd.testing.assert_frame_equal(part.loc[rows, FEATURES], full.loc[part.index[rows], FEATURES])


def test_features_ignore_sales_after_cutoff(long_df):
    """Scrambling every sale after the cutoff must not change any in-horizon feature."""
    cutoff = long_df["date"].max() - pd.Timedelta(days=MIN_LAG)
    scrambled = long_df.copy()
    after = scrambled["date"] > cutoff
    scrambled.loc[after, "sales"] = np.random.default_rng(0).integers(0, 100, after.sum())

    a = add_features(long_df).set_index(["id", "date"])
    b = add_features(scrambled).set_index(["id", "date"])
    horizon = a.index.get_level_values("date") > cutoff
    pd.testing.assert_frame_equal(a.loc[horizon, FEATURES], b.loc[horizon, FEATURES])


def test_lgbm_predicts_every_future_row(long_df):
    fold = make_folds(long_df["date"].max(), horizon=28, n_folds=1, step=28)[0]
    train, test = split(long_df, fold)
    model = LGBMForecaster(**FAST).fit(train)
    yhat = model.predict(test.drop(columns="sales").sample(frac=1, random_state=0))
    assert yhat.notna().all() and (yhat >= 0).all()
    assert len(yhat) == len(test)


def test_lgbm_rejects_horizon_longer_than_min_lag(long_df):
    fold = make_folds(long_df["date"].max(), horizon=MIN_LAG + 7, n_folds=1, step=7)[0]
    train, test = split(long_df, fold)
    model = LGBMForecaster(**FAST).fit(train)
    with pytest.raises(ValueError, match="leak"):
        model.predict(test.drop(columns="sales"))


def test_lgbm_beats_seasonal_naive_on_synthetic(long_df):
    kw = dict(horizon=28, n_folds=2, step=28)
    naive = run_backtest(SeasonalNaive, long_df, **kw).summary["wape"]
    lgbm = run_backtest(lambda: LGBMForecaster(**FAST), long_df, **kw)
    assert lgbm.summary["wape"] < naive
    assert set(lgbm.last_model.feature_importance()["feature"]) == set(FEATURES)


def test_chunked_feature_building_gives_identical_model(long_df):
    fold = make_folds(long_df["date"].max(), horizon=28, n_folds=1, step=28)[0]
    train, test = split(long_df, fold)
    future = test.drop(columns="sales")
    whole = LGBMForecaster(**FAST, chunk_series=10_000).fit(train).predict(future)
    chunked = LGBMForecaster(**FAST, chunk_series=3).fit(train).predict(future)
    np.testing.assert_allclose(chunked.to_numpy(), whole.to_numpy())


BUCKETS = (7, 14, 28)


@pytest.mark.parametrize("min_lag", BUCKETS)
def test_bucket_features_ignore_sales_after_cutoff(long_df, min_lag):
    """For each bucket: scrambling sales after the cutoff must not change any feature on
    the days that bucket forecasts (cutoff+1 .. cutoff+min_lag)."""
    from forecast.features.build import feature_names

    cutoff = long_df["date"].max() - pd.Timedelta(days=35)  # room to look past the bucket
    scrambled = long_df.copy()
    after = scrambled["date"] > cutoff
    scrambled.loc[after, "sales"] = np.random.default_rng(1).integers(0, 100, after.sum())

    a = add_features(long_df, min_lag).set_index(["id", "date"])
    b = add_features(scrambled, min_lag).set_index(["id", "date"])
    dates = a.index.get_level_values("date")
    window = (dates > cutoff) & (dates <= cutoff + pd.Timedelta(days=min_lag))
    names = feature_names(min_lag)
    pd.testing.assert_frame_equal(a.loc[window, names], b.loc[window, names])
    # ...and one day further the bucket WOULD leak, which is why buckets cap their horizon.
    beyond = dates == cutoff + pd.Timedelta(days=min_lag + 1)
    assert not a.loc[beyond, names].equals(b.loc[beyond, names])


def test_bucketed_model_routes_each_day_to_its_bucket(long_df):
    fold = make_folds(long_df["date"].max(), horizon=28, n_folds=1, step=28)[0]
    train, test = split(long_df, fold)
    future = test.drop(columns="sales")
    model = LGBMForecaster(**FAST, horizon_buckets=BUCKETS).fit(train)
    assert sorted(model.boosters) == list(BUCKETS)

    yhat = model.predict(future)
    assert yhat.notna().all() and (yhat >= 0).all() and len(yhat) == len(future)

    # Predicting only days 1-7 must give the same numbers: those rows use the 7-day booster
    # whatever else is in the request.
    first_week = future[(future["date"] - fold.cutoff).dt.days <= 7]
    np.testing.assert_allclose(model.predict(first_week).to_numpy(),
                               yhat.loc[first_week.index].to_numpy(), rtol=1e-6)
    imp = model.feature_importance()
    assert set(imp["bucket"]) == {"days 1-7", "days 8-14", "days 15-28"}
    assert model.get_params()["horizon_buckets"] == [7, 14, 28]
