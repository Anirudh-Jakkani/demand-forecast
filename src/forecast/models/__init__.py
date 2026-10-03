from forecast.models.base import ForecastModel
from forecast.models.lgbm import LGBMForecaster
from forecast.models.naive import MovingAverage, SeasonalNaive

MODELS: dict[str, type[ForecastModel]] = {
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
    "lightgbm": LGBMForecaster,
}

__all__ = ["MODELS", "ForecastModel", "LGBMForecaster", "MovingAverage", "SeasonalNaive"]
