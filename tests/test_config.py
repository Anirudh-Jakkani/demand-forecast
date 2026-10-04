from forecast.config import DEFAULT_CONFIG, TRACKING_URI_ENV, load_config


def test_mlflow_setting_its_own_env_var_does_not_redirect_configs(monkeypatch, tmp_path):
    """mlflow.set_tracking_uri() writes MLFLOW_TRACKING_URI into os.environ. A config loaded
    afterwards in the same process (e.g. the replay's sim workspace) must keep its own URI."""
    import mlflow

    monkeypatch.delenv(TRACKING_URI_ENV, raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'other.db').as_posix()}")

    assert load_config(DEFAULT_CONFIG).mlflow.tracking_uri == "sqlite:///mlflow.db"


def test_forecast_tracking_uri_overrides_config(monkeypatch):
    monkeypatch.setenv(TRACKING_URI_ENV, "http://mlflow:5000")
    assert load_config(DEFAULT_CONFIG).mlflow.tracking_uri == "http://mlflow:5000"
