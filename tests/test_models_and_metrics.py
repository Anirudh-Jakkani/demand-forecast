import numpy as np
import pandas as pd
import pytest

from forecast.evaluation.backtest import run_backtest
from forecast.evaluation.metrics import bias, rmsse, wape
from forecast.models import MovingAverage, SeasonalNaive


def _toy(n_days=21):
    dates = pd.date_range("2020-01-01", periods=n_days)
    pattern = np.tile([1, 2, 3, 4, 5, 6, 7], n_days // 7 + 1)[:n_days]
    return pd.DataFrame({"id": "A", "date": dates, "sales": pattern.astype(float)})


def test_seasonal_naive_repeats_last_week():
    df = _toy(21)
    train, future = df.iloc[:14], df.iloc[14:].drop(columns="sales")
    yhat = SeasonalNaive().fit(train).predict(future)
    # Perfectly periodic series -> perfect forecast.
    np.testing.assert_array_equal(yhat.to_numpy(), df["sales"].iloc[14:].to_numpy())


def test_moving_average_is_flat_mean():
    df = _toy(14)
    yhat = MovingAverage(window=7).fit(df).predict(df.drop(columns="sales"))
    assert np.allclose(yhat, 4.0)


def test_unknown_series_forecasts_zero():
    train = _toy(14)
    future = pd.DataFrame({"id": ["B"], "date": [pd.Timestamp("2020-01-15")]})
    assert SeasonalNaive().fit(train).predict(future).iloc[0] == 0.0


def test_wape_and_bias():
    y, yhat = np.array([10.0, 0.0, 10.0]), np.array([8.0, 1.0, 13.0])
    assert wape(y, yhat) == pytest.approx(6 / 20)
    assert bias(y, yhat) == pytest.approx(2 / 20)


def test_rmsse_is_one_for_naive_on_random_walk_scale():
    train = pd.DataFrame({"id": "A", "date": pd.date_range("2020-01-01", periods=4),
                          "sales": [0.0, 2.0, 4.0, 6.0]})
    test = pd.DataFrame({"id": ["A"], "sales": [10.0], "yhat": [8.0]})
    # scale = mean((2,2)^2) = 4 from first non-zero onward; err = 4 -> rmsse = 1
    assert rmsse(train, test) == pytest.approx(1.0)


def test_seasonal_naive_backtest_on_synthetic(long_df):
    result = run_backtest(SeasonalNaive, long_df, horizon=28, n_folds=3, step=28)
    assert len(result.fold_metrics) == 3
    assert 0 < result.summary["wape"] < 1.5
