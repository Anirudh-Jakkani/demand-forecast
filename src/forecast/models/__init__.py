from forecast.models.base import ForecastModel
from forecast.models.naive import MovingAverage, SeasonalNaive

MODELS: dict[str, type[ForecastModel]] = {
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
}

__all__ = ["MODELS", "ForecastModel", "MovingAverage", "SeasonalNaive"]
