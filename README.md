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
| 3a | Global LightGBM (lags / rolling / price / calendar), Tweedie loss | done |
| 3b | Prophet or N-BEATS | later |
| 4 | MLflow registry + champion promotion, FastAPI serving | done |
| 5 | Prefect training + monitoring flows, Evidently drift, retrain triggers | done |
| 6 | Replay simulation with injected drift, Streamlit dashboard | next |
| 7 | Docker Compose, GitHub Actions CI | |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                              # create .venv and install
uv run pytest                        # full suite on a synthetic M5-shaped sample (~4 min)
uv run pytest -m "not slow"          # skip registry/API/Prefect tests (~20 s)
```

### Run on synthetic data (no download needed)

```bash
uv run forecast make-sample
uv run forecast ingest
uv run forecast backtest -m seasonal_naive -m moving_average -m lightgbm
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

### Train, register, serve

```bash
uv run forecast train -m lightgbm     # backtest -> fit on all data -> register -> maybe promote
uv run forecast versions              # versions, aliases (champion / challenger), scores, reasons
uv run forecast promote 2             # manual promotion or rollback
uv run forecast serve                 # API docs at http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json"   -d '{"items": [{"store_id": "CA_1", "item_id": "FOODS_1_002"}], "horizon": 7}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + whether a model is loaded |
| `GET /model-info` | serving version, model type, data end date, backtest WAPE, why it was promoted |
| `POST /predict` | forecasts for up to 1000 store/item pairs, 1..28 days; every forecast is logged |
| `POST /admin/reload` | check for a new champion now (it is also polled every 60s) |

### Pipelines (Prefect)

```bash
uv run forecast pipeline train --as-of 2016-03-31    # run the training flow once
uv run forecast pipeline monitor --as-of 2016-04-07  # run the monitoring flow once
```

`--as-of` pretends today is that date (only data up to it is visible), which is how the
replay simulation works. To run both on a schedule:

```bash
uv run prefect server start                                         # terminal 1, UI on :4200
PREFECT_API_URL=http://127.0.0.1:4200/api uv run forecast pipeline serve   # terminal 2
```

| Flow | Schedule | Steps |
|---|---|---|
| `training` | Mondays 03:00 | load data up to as-of -> validate -> backtest, fit, register, promote -> batch-forecast all series into the prediction log |
| `monitoring` | daily 06:00 | join logged forecasts to arrived actuals (live WAPE, bias, by horizon) -> Evidently drift on sales / price / price change vs the champion's training window -> record the run -> trigger `training` if any rule fires |

Retrain triggers (in `configs/config.yaml`): live WAPE > 1.25x the champion's backtest WAPE
(given >= 100 scored points), >= 50% of monitored columns drifted, or the champion's data is
more than 35 days old. Every monitoring run is stored in `data/monitoring.sqlite` and each drift
check writes an Evidently HTML report to `reports/`.

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
- **LightGBM, one global model** over all series ([`models/lgbm.py`](src/forecast/models/lgbm.py)),
  Tweedie loss for zero-heavy demand. Every sales feature is lagged >= 28 days
  ([`features/build.py`](src/forecast/features/build.py)), so the full 28-day horizon is
  predicted directly with no recursion and no leakage; a test scrambles post-cutoff sales and
  asserts no in-horizon feature changes. Hyperparameters live in `configs/config.yaml`.
- **Registry and promotion** ([`registry.py`](src/forecast/registry.py)). A *recipe* is a model
  type + hyperparameters. First model becomes `champion`; a retrain of the champion's recipe on
  newer data is promoted as a refresh; a different recipe must beat the champion's recipe by
  >= 1% WAPE when both are backtested on the same data and folds, otherwise it is registered
  as `challenger`. `forecast promote <v>` is the manual override / rollback.
- **Deployable bundle** ([`serving_model.py`](src/forecast/serving_model.py)): the fitted model
  plus a *future frame* (calendar, events, SNAP and planned prices for the next 28 days), logged
  as an MLflow pyfunc. The API loads it by alias and swaps it in-place when the alias moves.
- **Prediction log** (`data/predictions.sqlite`): every served forecast with its model version,
  ready to be joined against actual sales for monitoring.
- **Pre-launch rows dropped.** Days before an item has a price are not real zero demand.

## Layout

```
configs/config.yaml        scope, backtest and MLflow settings
src/forecast/
  data/                    ingest, validation, future frame, synthetic sample generator
  evaluation/              metrics, backtest
  features/                leak-safe feature engineering
  models/                  ForecastModel interface, baselines, LightGBM
  api/                     FastAPI app, request/response schemas, prediction log
  flows/                   Prefect training + monitoring flows, schedules
  monitoring/              live accuracy, Evidently drift, retrain policy, run store
  registry.py              train -> register -> promote, champion loading
  serving_model.py         deployable model bundle (MLflow pyfunc)
  tracking.py              MLflow logging
  cli.py                   `forecast` command
tests/
```
