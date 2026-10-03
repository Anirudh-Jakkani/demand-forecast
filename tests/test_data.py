import pandas as pd
import pytest

from forecast.data.ingest import load_m5_long
from forecast.data.validate import DataValidationError, validate_long


def test_ingest_filters_state_and_category(long_df):
    assert set(long_df["store_id"].unique()) == {"CA_1", "CA_2"}
    assert set(long_df["dept_id"].unique()) == {"FOODS_1", "FOODS_2"}


def test_ingest_drops_prelaunch_rows(long_df):
    # Every remaining row has a price, i.e. the item was on the shelf.
    assert long_df["sell_price"].notna().all()


def test_ingest_start_date(raw_dir):
    df = load_m5_long(raw_dir, start_date="2015-06-01")
    assert df["date"].min() >= pd.Timestamp("2015-06-01")


def test_ingest_unknown_state_raises(raw_dir):
    with pytest.raises(ValueError, match="No series"):
        load_m5_long(raw_dir, state="NY")


def test_validate_passes_on_clean_data(long_df):
    validate_long(long_df)


def test_validate_catches_problems(long_df):
    bad = long_df.copy()
    bad.loc[bad.index[0], "sales"] = -1
    bad = bad.drop(index=bad.index[10])  # creates a date gap
    with pytest.raises(DataValidationError) as exc:
        validate_long(bad)
    assert "negative" in str(exc.value) and "gaps" in str(exc.value)
