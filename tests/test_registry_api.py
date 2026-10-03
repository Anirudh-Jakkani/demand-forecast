import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecast.api.app import create_app
from forecast.data.future import build_future_frame
from forecast.registry import list_versions, promote_version, train_and_register
from tests.conftest import make_cfg


def test_future_frame_covers_horizon_for_live_series(long_df, raw_dir):
    fut = build_future_frame(long_df, raw_dir, horizon=28, state="CA")
    cutoff = long_df["date"].max()
    assert fut["date"].min() == cutoff + pd.Timedelta(days=1)
    assert fut["date"].nunique() == 28
    assert fut.groupby("id", observed=True).size().eq(28).all()
    assert fut["sell_price"].notna().all()
    assert "sales" not in fut.columns


@pytest.mark.slow
def test_promotion_rules(tmp_path, raw_dir, long_df):
    cfg = make_cfg(tmp_path, raw_dir)

    first = train_and_register(cfg, long_df, "seasonal_naive")
    assert first.promoted and first.reason == "no champion yet"

    # Moving average beats seasonal naive on the synthetic data -> new champion.
    better = train_and_register(cfg, long_df, "moving_average")
    assert better.promoted and better.champion_wape == pytest.approx(first.cv_wape)

    # Seasonal naive is worse than the champion -> registered as challenger only.
    worse = train_and_register(cfg, long_df, "seasonal_naive")
    assert not worse.promoted and "does not beat" in worse.reason

    # Same recipe as the champion, retrained -> promoted as a refresh.
    refresh = train_and_register(cfg, long_df, "moving_average")
    assert refresh.promoted and "same recipe" in refresh.reason

    versions = list_versions(cfg).set_index("version")
    assert versions.loc[int(refresh.version), "aliases"] == "champion"
    assert versions.loc[int(worse.version), "aliases"] == "challenger"

    promote_version(cfg, first.version)  # manual rollback
    assert list_versions(cfg).set_index("version").loc[int(first.version), "aliases"] == "champion"


@pytest.fixture(scope="module")
def api(tmp_path_factory, raw_dir, long_df):
    tmp_path = tmp_path_factory.mktemp("api")
    cfg = make_cfg(tmp_path, raw_dir)
    train_and_register(cfg, long_df, "lightgbm")
    with TestClient(create_app(cfg)) as client:
        yield client, cfg


@pytest.mark.slow
def test_health_and_model_info(api):
    client, _ = api
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["model_type"] == "lightgbm"
    assert info["version"] == "1"
    assert info["max_horizon"] == 28 and info["n_series"] == 20


@pytest.mark.slow
def test_predict_returns_forecasts_and_logs_them(api):
    client, _ = api
    body = {"items": [{"store_id": "CA_1", "item_id": "FOODS_1_001"},
                      {"store_id": "CA_2", "item_id": "FOODS_2_003"}], "horizon": 7}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["forecasts"]) == 2
    assert all(len(s["forecast"]) == 7 for s in data["forecasts"])
    assert all(d["yhat"] >= 0 for s in data["forecasts"] for d in s["forecast"])

    logged = client.app.state.prediction_log.read()
    mine = logged[logged["request_id"] == data["request_id"]]
    assert len(mine) == 14 and set(mine["model_version"]) == {"1"}


@pytest.mark.slow
def test_predict_errors(api):
    client, _ = api
    unknown = {"items": [{"store_id": "CA_1", "item_id": "NOPE"}]}
    assert client.post("/predict", json=unknown).status_code == 404
    too_far = {"items": [{"store_id": "CA_1", "item_id": "FOODS_1_001"}], "horizon": 60}
    assert client.post("/predict", json=too_far).status_code == 422
    assert client.post("/predict", json={"items": []}).status_code == 422


@pytest.mark.slow
def test_hot_reload_picks_up_new_champion(api, long_df):
    client, cfg = api
    train_and_register(cfg, long_df, "moving_average", promote=False)
    promote_version(cfg, "2")
    assert client.post("/admin/reload").json() == {"reloaded": True, "version": "2"}
    assert client.get("/model-info").json()["model_type"] == "moving_average"
