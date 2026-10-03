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
) -> list[str]:
    reasons = []
    if n_points >= cfg.min_points and cv_wape and not math.isnan(live_wape):
        ratio = live_wape / cv_wape
        if ratio > cfg.max_wape_ratio:
            reasons.append(f"live WAPE {live_wape:.3f} is {ratio:.2f}x the backtest "
                           f"WAPE {cv_wape:.3f} (limit {cfg.max_wape_ratio:.2f}x)")
    if drift_share is not None and drift_share >= cfg.drift_share_threshold:
        reasons.append(f"{drift_share:.0%} of monitored columns drifted "
                       f"(limit {cfg.drift_share_threshold:.0%})")
    if model_age_days > cfg.max_model_age_days:
        reasons.append(f"model is {model_age_days} days old (limit {cfg.max_model_age_days})")
    return reasons
