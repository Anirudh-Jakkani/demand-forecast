"""When to retrain. Any one trigger is enough; all reasons are recorded."""

from __future__ import annotations

import math

from forecast.config import MonitoringConfig


def retrain_reasons(
    cfg: MonitoringConfig,
    live_wape: float,
    cv_wape: float | None,
    drift_share: float | None,
    model_age_days: int,
    n_points: int,
    drifted_columns: list[str] | None = None,
    drift_scores: dict[str, float] | None = None,
) -> list[str]:
    reasons = []
    if n_points >= cfg.min_points and cv_wape and not math.isnan(live_wape):
        ratio = live_wape / cv_wape
        if ratio > cfg.max_wape_ratio:
            reasons.append(f"live WAPE {live_wape:.3f} is {ratio:.2f}x the backtest "
                           f"WAPE {cv_wape:.3f} (limit {cfg.max_wape_ratio:.2f}x)")
    # A fresh model gets a cooldown: its reference window still holds pre-change data,
    # so drift would otherwise fire again the next day.
    if model_age_days >= cfg.drift_cooldown_days:
        if drift_share is not None and drift_share >= cfg.drift_share_threshold:
            reasons.append(f"{drift_share:.0%} of monitored columns drifted "
                           f"(limit {cfg.drift_share_threshold:.0%})")
        elif cfg.target_drift_trigger and cfg.target_column in (drifted_columns or []):
            # What we forecast has changed distribution: the clearest retrain signal.
            score = (drift_scores or {}).get(cfg.target_column)
            detail = f" (drift score {score:.3f})" if score is not None else ""
            reasons.append(f"target '{cfg.target_column}' drifted{detail}")
    if model_age_days > cfg.max_model_age_days:
        reasons.append(f"model is {model_age_days} days old (limit {cfg.max_model_age_days})")
    return reasons
