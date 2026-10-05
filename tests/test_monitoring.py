import numpy as np
import pandas as pd
import pytest
import yaml

from forecast.config import MonitoringConfig
from forecast.monitoring.drift import drift_windows, run_drift
from forecast.monitoring.performance import join_forecasts_to_actuals, performance_summary
from forecast.monitoring.policy import retrain_reasons
from forecast.monitoring.store import MonitoringStore
from tests.conftest import make_cfg

D = pd.Timestamp


# ---------- unit tests ----------

def _preds(rows):
    return pd.DataFrame(rows, columns=["series_id", "date", "yhat", "cutoff", "created_at"])


def test_join_uses_freshest_forecast_made_before_the_day():
    preds = _preds([
        ("A", D("2020-01-10"), 1.0, D("2020-01-01"), "t1"),   # old forecast
        ("A", D("2020-01-10"), 5.0, D("2020-01-08"), "t2"),   # fresher -> used
        ("A", D("2020-01-10"), 9.0, D("2020-01-10"), "t3"),   # made on the day -> ignored
        ("A", D("2020-01-01"), 2.0, D("2019-12-25"), "t0"),   # outside window -> ignored
    ])
    actuals = pd.DataFrame({"id": ["A"], "date": [D("2020-01-10")], "sales": [4.0]})
    joined = join_forecasts_to_actuals(preds, actuals, D("2020-01-10"), window_days=7)
    assert joined[["yhat", "sales", "horizon"]].values.tolist() == [[5.0, 4.0, 2]]
    perf = performance_summary(joined)
    assert perf["n_points"] == 1 and perf["live_wape"] == pytest.approx(0.25)


def test_performance_summary_empty():
    perf = performance_summary(pd.DataFrame())
    assert perf["n_points"] == 0 and np.isnan(perf["live_wape"])


def test_retrain_policy_triggers():
    cfg = MonitoringConfig(max_wape_ratio=1.25, drift_share_threshold=0.5,
                           max_model_age_days=35, min_points=100)
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 500) == []
    assert "backtest" in retrain_reasons(cfg, 0.60, 0.40, 0.0, 7, 500)[0]
    assert retrain_reasons(cfg, 0.60, 0.40, 0.0, 7, 10) == []          # too few points
    assert "drifted" in retrain_reasons(cfg, 0.40, 0.40, 0.67, 7, 500)[0]
    assert "days old" in retrain_reasons(cfg, 0.40, 0.40, 0.0, 40, 500)[0]
    # Target drift alone is enough, even below the column-share threshold...
    reasons = retrain_reasons(cfg, 0.40, 0.40, 0.33, 7, 500, ["sales"], {"sales": 0.36})
    assert reasons == ["target 'sales' drifted (drift score 0.360)"]
    # ...but not a non-target column, and not when the trigger is switched off.
    assert retrain_reasons(cfg, 0.40, 0.40, 0.33, 7, 500, ["sell_price"]) == []
    off = cfg.model_copy(update={"target_drift_trigger": False})
    assert retrain_reasons(off, 0.40, 0.40, 0.33, 7, 500, ["sales"]) == []
    # A freshly retrained model gets a cooldown before drift can retrain it again.
    assert retrain_reasons(cfg, 0.40, 0.40, 1.0, 3, 500, ["sales"]) == []


def test_retrain_policy_bias():
    cfg = MonitoringConfig(max_abs_bias=0.15, min_points=100, drift_cooldown_days=7)
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 500, live_bias=-0.10) == []
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 500, live_bias=-0.22) == [
        "live bias -22.0% (under-forecasting; limit ±15%)"]
    assert "over-forecasting" in retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 500,
                                                 live_bias=0.30)[0]
    # Not on too few points, not during the post-retrain cooldown, not when switched off.
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 10, live_bias=-0.5) == []
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 6, 500, live_bias=-0.5) == []
    off = cfg.model_copy(update={"max_abs_bias": None})
    assert retrain_reasons(off, 0.40, 0.40, 0.0, 7, 500, live_bias=-0.5) == []
    assert retrain_reasons(cfg, 0.40, 0.40, 0.0, 7, 500, live_bias=float("nan")) == []


def test_drift_detects_demand_shift(long_df, tmp_path):
    end = long_df["date"].max()
    ref_end = end - pd.Timedelta(days=28)
    shocked = long_df.copy()
    shocked.loc[shocked["date"] > ref_end, "sales"] *= 3
    ref, cur = drift_windows(shocked, ref_end, end, window_days=28)
    result = run_drift(ref, cur, tmp_path / "drift.html")
    assert "sales" in result.drifted_columns
    assert (tmp_path / "drift.html").exists()

    ref, cur = drift_windows(long_df, ref_end, end, window_days=28)
    assert "sales" not in run_drift(ref, cur).drifted_columns


def test_monitoring_store_roundtrip(tmp_path):
    store = MonitoringStore(tmp_path / "m.sqlite")
    store.write({"as_of": "2020-01-01", "run_at": "x", "decision": "ok",
                 "reasons": [], "drift_scores": {"sales": 0.1}})
    df = store.read()
    assert df.loc[0, "decision"] == "ok" and df.loc[0, "drift_scores"] == '{"sales": 0.1}'


# ---------- flow tests (run real Prefect flows against a temporary server) ----------

@pytest.fixture(scope="module")
def prefect_env():
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


def _setup(tmp_path, raw_dir, df):
    cfg = make_cfg(tmp_path, raw_dir)
    df.to_parquet(cfg.data.processed_path, index=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return cfg, str(path)


@pytest.mark.slow
def test_training_then_healthy_monitoring(prefect_env, tmp_path, raw_dir, long_df):
    from forecast.flows.monitoring import monitoring_flow
    from forecast.flows.training import training_flow

    cfg, cfg_path = _setup(tmp_path, raw_dir, long_df)
    cutoff = long_df["date"].max() - pd.Timedelta(days=56)

    trained = training_flow(as_of=str(cutoff.date()), model="moving_average",
                            config_path=cfg_path)
    assert trained["promoted"] and trained["batch_forecast"]["rows"] == 20 * 28

    week_later = str((cutoff + pd.Timedelta(days=7)).date())
    result = monitoring_flow(as_of=week_later, config_path=cfg_path)
    assert result["decision"] == "ok", result["reasons"]
    assert result["n_points"] == 20 * 7
    assert result["model_age_days"] == 7

    runs = MonitoringStore(cfg.monitoring.db_path).read()
    assert len(runs) == 1 and runs.loc[0, "decision"] == "ok"


@pytest.mark.slow
def test_demand_shock_triggers_retrain(prefect_env, tmp_path, raw_dir, long_df):
    from forecast.flows.monitoring import monitoring_flow
    from forecast.flows.training import training_flow

    cutoff = long_df["date"].max() - pd.Timedelta(days=56)
    shocked = long_df.copy()
    shocked.loc[shocked["date"] > cutoff, "sales"] *= 3   # demand triples after training
    cfg, cfg_path = _setup(tmp_path, raw_dir, shocked)

    training_flow(as_of=str(cutoff.date()), model="moving_average", config_path=cfg_path)
    two_weeks = str((cutoff + pd.Timedelta(days=14)).date())
    result = monitoring_flow(as_of=two_weeks, config_path=cfg_path)

    assert result["decision"] == "retrain"
    assert any("backtest WAPE" in r for r in result["reasons"])
    assert any("drifted" in r for r in result["reasons"])
    assert "sales" in result["drifted_columns"]
    # The triggered retrain ran and refreshed the champion on post-shock data.
    assert result["training"]["promoted"] and result["training"]["as_of"] == two_weeks


@pytest.mark.slow
def test_bias_catches_a_demand_shift_within_days(prefect_env, tmp_path, raw_dir, long_df):
    from forecast.flows.monitoring import monitoring_flow
    from forecast.flows.training import training_flow

    cutoff = long_df["date"].max() - pd.Timedelta(days=56)
    shock_day = cutoff + pd.Timedelta(days=10)          # past the post-retrain cooldown
    shocked = long_df.copy()
    shocked.loc[shocked["date"] >= shock_day, "sales"] *= 1.6
    cfg, cfg_path = _setup(tmp_path, raw_dir, shocked)

    training_flow(as_of=str(cutoff.date()), model="moving_average", config_path=cfg_path)
    before = monitoring_flow(as_of=str((shock_day - pd.Timedelta(days=1)).date()),
                             config_path=cfg_path, auto_retrain=False)
    assert before["decision"] == "ok", before["reasons"]

    # Three shocked days out of the 7-day window: bias is past -15%, drift is not yet.
    after = monitoring_flow(as_of=str((shock_day + pd.Timedelta(days=2)).date()),
                            config_path=cfg_path, auto_retrain=False)
    assert after["live_bias"] < -cfg.monitoring.max_abs_bias
    assert [r for r in after["reasons"] if r.startswith("live bias")], after["reasons"]
    assert "sales" not in after["drifted_columns"]
