import pandas as pd
import pytest

from forecast.data.ingest import load_m5_long
from forecast.data.validate import DataValidationError, validate_long


def _reference_load(raw_dir, state="CA", category="FOODS"):
    """The original melt/merge implementation: simple, but needs GBs on the real file."""
    from forecast.data.ingest import CALENDAR_FILE, ID_COLS, PRICES_FILE, SALES_FILE

    calendar = pd.read_csv(raw_dir / CALENDAR_FILE, parse_dates=["date"])
    sales = pd.read_csv(raw_dir / SALES_FILE)
    sales = sales[(sales["state_id"] == state) & (sales["cat_id"] == category)]
    long = sales.melt(id_vars=ID_COLS, var_name="d", value_name="sales")
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                "event_name_1", "event_type_1", f"snap_{state}"]
    long = long.merge(calendar[cal_cols], on="d").rename(columns={f"snap_{state}": "snap"})
    long = long.merge(pd.read_csv(raw_dir / PRICES_FILE),
                      on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.dropna(subset=["sell_price"]).sort_values(["id", "date"], ignore_index=True)
    return long


def test_ingest_matches_reference_implementation(raw_dir, long_df):
    ref = _reference_load(raw_dir)
    assert len(long_df) == len(ref)
    for col in ["id", "item_id", "dept_id", "store_id", "event_name_1", "event_type_1"]:
        assert long_df[col].astype(object).fillna("").tolist() == \
            ref[col].astype(object).fillna("").tolist(), col
    for col in ["date", "sales", "wday", "month", "year", "snap"]:
        assert (long_df[col].to_numpy() == ref[col].to_numpy()).all(), col
    # The pipeline has always stored prices as float32.
    assert (long_df["sell_price"].to_numpy() == ref["sell_price"].to_numpy("float32")).all()
    assert long_df["sales"].dtype == "float32" and long_df["sell_price"].dtype == "float32"
    assert long_df["id"].dtype == "category" and long_df["wday"].dtype == "int8"


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
