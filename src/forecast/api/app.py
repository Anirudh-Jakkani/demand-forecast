"""Forecast API. Serves whichever model version holds the `champion` alias and picks up
a newly promoted champion automatically, with no restart or redeploy.

    uv run forecast serve            # http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import threading
import uuid

from fastapi import FastAPI, HTTPException

from forecast.api.prediction_log import PredictionLog
from forecast.api.schemas import (
    DailyForecast,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    SeriesForecast,
)
from forecast.config import Config, load_config
from forecast.registry import CHAMPION, get_champion, load_champion
from forecast.serving_model import ForecastBundle

log = logging.getLogger(__name__)


class ModelHolder:
    """Thread-safe holder for the live champion; swaps atomically on reload."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self.bundle: ForecastBundle | None = None
        self.version = None
        self.loaded_at: dt.datetime | None = None

    def refresh(self) -> bool:
        """Load the champion if it changed since last time. Returns True if swapped."""
        from mlflow import MlflowClient

        from forecast.tracking import setup_mlflow

        setup_mlflow(self.cfg)
        champ = get_champion(MlflowClient(), self.cfg.registry.model_name)
        if champ is None or (self.version and str(champ.version) == str(self.version.version)):
            return False
        bundle, version = load_champion(self.cfg)
        with self._lock:
            self.bundle, self.version = bundle, version
            self.loaded_at = dt.datetime.now(dt.UTC)
        log.info("Serving %s v%s (%s)", self.cfg.registry.model_name, version.version,
                 bundle.model_name)
        return True

    def current(self) -> tuple[ForecastBundle, object]:
        with self._lock:
            if self.bundle is None:
                raise HTTPException(503, f"No model with alias '{CHAMPION}' is available yet")
            return self.bundle, self.version


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    holder = ModelHolder(cfg)
    pred_log = PredictionLog(cfg.resolve(cfg.serving.prediction_log))

    async def watch_champion():
        while True:
            await asyncio.sleep(cfg.serving.reload_interval_s)
            try:
                await asyncio.to_thread(holder.refresh)
            except Exception:  # keep serving the current model if the registry hiccups
                log.exception("Champion refresh failed")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            holder.refresh()
        except Exception:
            log.exception("Could not load a champion at startup; /predict will return 503")
        task = asyncio.create_task(watch_champion())
        yield
        task.cancel()

    app = FastAPI(title="M5 Demand Forecast API", version="0.1.0", lifespan=lifespan)
    app.state.holder = holder
    app.state.prediction_log = pred_log

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": holder.bundle is not None}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info():
        bundle, version = holder.current()
        cv_wape = version.tags.get("cv_wape")
        return ModelInfo(
            registered_name=cfg.registry.model_name,
            version=str(version.version),
            model_type=bundle.model_name,
            data_end=bundle.cutoff.date(),
            max_horizon=bundle.max_horizon,
            n_series=bundle.n_series,
            cv_wape=float(cv_wape) if cv_wape else None,
            promotion_reason=version.tags.get("promotion_reason"),
            loaded_at=holder.loaded_at,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        bundle, version = holder.current()
        if req.horizon > bundle.max_horizon:
            raise HTTPException(422, f"horizon must be <= {bundle.max_horizon}")

        ids, unknown = [], []
        for key in req.items:
            sid = bundle.series_id(key.store_id, key.item_id)
            (ids if sid else unknown).append(sid or f"{key.store_id}/{key.item_id}")
        if unknown:
            raise HTTPException(404, f"Unknown series: {unknown[:10]}")

        fc = bundle.forecast(ids, req.horizon)
        request_id = uuid.uuid4().hex
        pred_log.write(request_id, dt.datetime.now(dt.UTC).isoformat(), str(version.version),
                       fc, bundle.cutoff)

        forecasts = [
            SeriesForecast(
                store_id=store_id, item_id=item_id,
                forecast=[DailyForecast(date=d.date(), yhat=round(float(y), 4))
                          for d, y in zip(g["date"], g["yhat"], strict=True)],
            )
            for (store_id, item_id), g in fc.groupby(["store_id", "item_id"], sort=False)
        ]
        return PredictResponse(
            request_id=request_id,
            model_version=str(version.version),
            model_type=bundle.model_name,
            forecast_start=(bundle.cutoff + dt.timedelta(days=1)).date(),
            forecasts=forecasts,
        )

    @app.post("/admin/reload")
    def reload():
        """Force an immediate champion check instead of waiting for the poll interval."""
        swapped = holder.refresh()
        _, version = holder.current()
        return {"reloaded": swapped, "version": str(version.version)}

    return app
