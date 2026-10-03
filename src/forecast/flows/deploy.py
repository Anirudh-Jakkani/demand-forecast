"""Schedule both flows with Prefect. Needs a Prefect server:

    uv run prefect server start          # UI at http://127.0.0.1:4200
    uv run forecast pipeline serve       # in a second terminal; keeps running
"""

from __future__ import annotations

from prefect import serve
from prefect.schedules import Cron

from forecast.config import Config
from forecast.flows.monitoring import monitoring_flow
from forecast.flows.training import training_flow


def serve_flows(cfg: Config, config_path: str | None, model: str) -> None:
    s = cfg.schedule
    params = {"config_path": config_path}
    serve(
        training_flow.to_deployment(
            name="weekly-training", schedule=Cron(s.training_cron, timezone=s.timezone),
            parameters={**params, "model": model},
            description="Retrain on all data, register, promote if better, batch forecast.",
        ),
        monitoring_flow.to_deployment(
            name="daily-monitoring", schedule=Cron(s.monitoring_cron, timezone=s.timezone),
            parameters=params,
            description="Score live forecasts, check drift, retrain on triggers.",
        ),
    )
