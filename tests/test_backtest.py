import pandas as pd

from forecast.evaluation.backtest import iter_splits, make_folds, run_backtest
from forecast.models.base import ForecastModel


def test_folds_are_ordered_and_end_at_last_date():
    last = pd.Timestamp("2016-05-22")
    folds = make_folds(last, horizon=28, n_folds=3, step=28)
    assert [f.index for f in folds] == [0, 1, 2]
    assert folds[0].cutoff < folds[1].cutoff < folds[2].cutoff
    assert folds[-1].test_end == last


def test_no_leakage_between_train_and_test(long_df):
    for fold, train, test in iter_splits(long_df, horizon=14, n_folds=3, step=14):
        assert train["date"].max() == fold.cutoff
        assert test["date"].min() > fold.cutoff
        assert test["date"].nunique() == 14


class _SpyModel(ForecastModel):
    """Records what it was given so the test can check the harness hides the target."""

    seen_future_cols: list = []

    def fit(self, train):
        self.max_train_date = train["date"].max()
        return self

    def predict(self, future):
        _SpyModel.seen_future_cols.append(set(future.columns))
        assert future["date"].min() > self.max_train_date
        return pd.Series(0.0, index=future.index)


def test_predict_never_sees_target(long_df):
    run_backtest(_SpyModel, long_df, horizon=7, n_folds=2, step=7)
    assert all("sales" not in cols for cols in _SpyModel.seen_future_cols)
