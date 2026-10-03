import pandas as pd
import pytest

from forecast.config import PROJECT_ROOT
from forecast.monitoring import views
from forecast.simulation.replay import Shock, load_meta, prepare_workspace, run_replay
from tests.conftest import make_cfg


def test_shock_scales_demand_and_price_from_date_for_one_dept(long_df):
    date = long_df["date"].max() - pd.Timedelta(days=10)
    shock = Shock(str(date.date()), demand_factor=2.0, price_factor=0.5, dept="FOODS_1")
    out = shock.apply(long_df)
    hit = (long_df["date"] >= date) & (long_df["dept_id"] == "FOODS_1")
    assert (out.loc[hit, "sales"] == (long_df.loc[hit, "sales"] * 2).round()).all()
    assert (out.loc[hit, "sell_price"] == long_df.loc[hit, "sell_price"] * 0.5).all()
    pd.testing.assert_frame_equal(out[~hit], long_df[~hit])


def test_workspace_refuses_to_wipe_unrelated_folder(tmp_path, raw_dir, long_df):
    cfg = make_cfg(tmp_path, raw_dir)
    long_df.to_parquet(cfg.data.processed_path, index=False)
    precious = tmp_path / "not_a_sim"
    precious.mkdir()
    (precious / "important.txt").write_text("keep me")
    with pytest.raises(FileExistsError):
        prepare_workspace(cfg, precious, "2015-06-01", 10, None, "moving_average", False)
    assert (precious / "important.txt").exists()


def test_workspace_redirects_every_output(tmp_path, raw_dir, long_df):
    cfg = make_cfg(tmp_path, raw_dir)
    long_df.to_parquet(cfg.data.processed_path, index=False)
    work = tmp_path / "sim"
    shock = Shock("2015-09-01", demand_factor=1.5)
    cfg_path = prepare_workspace(cfg, work, "2015-08-01", 30, shock, "moving_average", False)

    from forecast.config import load_config
    sim = load_config(cfg_path)
    for p in (sim.data.processed_path, sim.serving.prediction_log, sim.monitoring.db_path,
              sim.monitoring.reports_dir, sim.mlflow.artifact_root):
        assert work in sim.resolve(p).parents
    assert str(work.as_posix()) in sim.mlflow.tracking_uri
    assert sim.registry.model_name.endswith("-sim")
    assert load_meta(work)["shock"]["demand_factor"] == 1.5
    # Re-preparing an existing sim workspace is allowed (it is ours).
    prepare_workspace(cfg, work, "2015-08-01", 30, None, "moving_average", False)


@pytest.mark.slow
def test_replay_end_to_end_and_dashboard_views(tmp_path, raw_dir, long_df):
    from prefect.testing.utilities import prefect_test_harness

    cfg = make_cfg(tmp_path, raw_dir)
    end = long_df["date"].max()
    start = end - pd.Timedelta(days=40)
    shock_day = start + pd.Timedelta(days=10)
    long_df.to_parquet(cfg.data.processed_path, index=False)
    work = tmp_path / "sim"

    with prefect_test_harness():
        results = run_replay(cfg, str(start.date()), 30, Shock(str(shock_day.date()), 3.0),
                             model="moving_average", workdir=work)

    assert len(results) == 30
    assert any(r["decision"] == "retrain" for r in results)

    from forecast.config import load_config
    sim = load_config(work / "config.yaml")
    runs = views.monitoring_runs(sim)
    assert len(runs) == 30 and runs["as_of"].is_monotonic_increasing
    assert not views.version_changes(runs).empty           # at least one retrain served
    assert not views.drift_long(runs).empty
    totals = views.daily_totals(views.scored_forecasts(sim))
    assert {"date", "actual", "forecast"} <= set(totals.columns) and len(totals) > 0
    sid = views.series_ids(sim)[0]
    lines = views.series_view(sim, sid)["line"].unique()
    assert "actual" in lines and any(line.startswith("forecast v") for line in lines)

    # The dashboard script runs top to bottom against this workspace without errors.
    import os

    from streamlit.testing.v1 import AppTest

    os.environ["FORECAST_CONFIG"] = str(work / "config.yaml")
    try:
        at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=120)
        at.run()
        assert not at.exception, at.exception
        assert any("Live accuracy" in h.value for h in at.subheader)
    finally:
        del os.environ["FORECAST_CONFIG"]
