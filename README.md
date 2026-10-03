# Demand Forecasting with MLOps (M5)

Daily unit-sales forecasts per **store × item** for Walmart's M5 dataset (California, FOODS:
~5.8k series), built as a production system rather than a notebook: time-respecting
backtests, experiment tracking, a model registry, scheduled retraining, drift monitoring,
and a served API.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Ingest + validation (M5 → long Parquet) | done |
| 2 | Rolling-origin backtest harness, metrics, baselines, MLflow tracking | done |
| 3 | LightGBM (lags / rolling / price / calendar), Prophet or N-BEATS | next |
| 4 | MLflow registry + champion promotion, FastAPI serving | |
| 5 | Prefect training + monitoring flows | |
| 6 | Evidently drift, replay simulation, Streamlit dashboard | |
| 7 | Docker Compose, GitHub Actions CI | |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                              # create .venv and install
uv run pytest                        # tests run on a synthetic M5-shaped sample
```

### Run on synthetic data (no download needed)

```bash
uv run forecast make-sample
uv run forecast ingest
uv run forecast backtest -m seasonal_naive
uv run forecast backtest -m moving_average
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

### Run on real M5 data

Download from the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
competition and put these three files in `data/raw/` (overwriting any synthetic sample):

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Then run `forecast ingest` and `forecast backtest` as above. Scope (state, category,
start date) and backtest settings live in [`configs/config.yaml`](configs/config.yaml).

## Design notes

- **One interface, one harness.** Every model implements `fit(train)` / `predict(future)`
  ([`models/base.py`](src/forecast/models/base.py)). The backtest drops the `sales` column
  before calling `predict`, so a model cannot see the answers; a test enforces this.
- **Rolling-origin backtest.** 4 folds × 28-day horizon, the last fold ending on the final
  day of data. Each fold gets a fresh model.
- **Metrics.** WAPE (headline: "% of units wrong"), MAE, RMSE, bias (+ = over-forecast),
  and RMSSE (M5's metric, unweighted).
- **Pre-launch rows dropped.** Days before an item has a price are not real zero demand.

## Layout

```
configs/config.yaml        scope, backtest and MLflow settings
src/forecast/
  data/                    ingest, validation, synthetic sample generator
  evaluation/              metrics, backtest
  models/                  ForecastModel interface + baselines
  tracking.py              MLflow logging
  cli.py                   `forecast` command
tests/
```
