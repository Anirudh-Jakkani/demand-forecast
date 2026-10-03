import numpy as np
import pandas as pd
import pytest

from forecast.evaluation.backtest import make_folds, run_backtest, split
from forecast.features.build import FEATURES, MIN_LAG, add_features
from forecast.models import LGBMForecaster, SeasonalNaive

FAST = {"num_boost_round": 50, "num_leaves": 15, "min_data_in_leaf": 20, "train_days": None}


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
