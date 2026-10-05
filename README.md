# Demand Forecasting with MLOps (M5)

[![CI](https://github.com/Anirudh-Jakkani/demand-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/Anirudh-Jakkani/demand-forecast/actions/workflows/ci.yml)

Daily unit-sales forecasts per **store × item** for Walmart's M5 dataset (California, FOODS:
~5.8k series), built as a production system rather than a notebook: time-respecting
backtests, experiment tracking, a model registry, scheduled retraining, drift monitoring,
and a served API.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Ingest + validation (M5 → long Parquet) | done |
| 2 | Rolling-origin backtest harness, metrics, baselines, MLflow tracking | done |
| 3a | Global LightGBM (lags / rolling / price / calendar), Tweedie loss, horizon buckets | done |
| 3b | Prophet or N-BEATS | later |
| 4 | MLflow registry + champion promotion, FastAPI serving | done |
| 5 | Prefect training + monitoring flows, Evidently drift, retrain triggers | done |
| 6 | Replay simulation with injected drift, Streamlit dashboard | done |
| 7 | Docker Compose, GitHub Actions CI | done |

## Results on real M5 (California, FOODS)

5,748 store x item series, daily units 2013-01-01 to 2016-05-22 (6.29M rows, 49.7% of
days sell zero). Rolling-origin backtest: 4 folds x 28-day horizon, the last ending
2016-05-22; every model sees only data up to its fold's cutoff.

| Model | WAPE | RMSSE | Bias | Train time (4 folds) |
|---|---|---|---|---|
| **LightGBM, horizon buckets 7/14/28** | **0.670** | **0.762** | -3.4% | 64 min |
| LightGBM, single model (lags >= 28) | 0.680 | 0.769 | -4.9% | 16 min |
| Moving average (28 days) | 0.691 | 0.777 | -2.5% | seconds |
| Seasonal naive (same weekday last week) | 0.819 | 1.000 | -3.0% | seconds |

Where the gain comes from (latest fold, cutoff 2016-04-24, WAPE by forecast day):

| Model | Days 1-7 | Days 8-14 | Days 15-28 |
|---|---|---|---|
| LightGBM, buckets | **0.638** | **0.657** | 0.664 |
| LightGBM, single | 0.654 | 0.665 | 0.664 |
| Moving average | 0.681 | 0.675 | 0.686 |

The bucketed model trains separate boosters for days 1-7, 8-14 and 15-28 whose sales
features may be as recent as 7, 14 and 28 days. Each is leak-free for its own days (tests
scramble post-cutoff sales and assert no in-horizon feature moves). The gain is entirely
on the near days, and days 15-28 are identical by construction. Both LightGBM variants
under-forecast far days (bias -7.4% on days 15-28) because 4-week-old signal trails rising
demand; trend / year-over-year features are the next thing to try.

Item x store x day is the noisiest level of M5, so absolute WAPE is high for every model;
the comparison between models, on identical folds, is what matters.

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

### Replay demo + dashboard

```bash
uv run forecast simulate --days 120 --shock-factor 1.6   # demand x1.6 from halfway through
uv run forecast dashboard                                 # http://localhost:8501
```

Example run on the synthetic sample (start 2016-01-15, demand x1.6 from 2016-03-15):

- Live WAPE never reaches its retrain limit: it peaks at 1.14x the backtest score
  (limit 1.25x). Error on 20 noisy daily series hides a 60% jump in demand.
- Live bias does show it. It was already -14% the day before the shock, crossed the limit
  on the shock day (-20.1%), and the monitoring flow retrained that same day.
- Each new model has seen only a few shocked days, so it still under-forecasts. The bias
  rule fires again as each cooldown ends (03-22, 03-29), drift fires once more on 04-07,
  and bias is back inside -10% by mid-April.
- With the bias rule off, the same replay first retrains on 03-21 (target drift). Over the
  month after the shock it under-forecasts by 19.9% instead of 18.4%.
- With only 20 series the weekly bias is noisy (-12% to +3% before the shock), so the rule
  also fired once with no shock (02-01, -15.2%). Bias over the 5.7k real series should be
  much steadier.

`simulate` stands on day 0, trains, then walks forward one day at a time running the real
monitoring flow each day (which retrains whenever a trigger fires). A shock can be injected
from any date: `--shock-factor` scales demand, `--price-factor` scales prices, `--dept` limits
it to one department, `--weekly-retrain` adds the Monday schedule. Everything is written to
`sim/` (its own registry, logs and reports), so the real workspace is untouched.

The dashboard shows, for either workspace:

- **KPIs:** serving version, live WAPE vs backtest, live bias, share of drifting columns, model age, last decision
- **Live accuracy:** trailing-7-day WAPE against the backtest WAPE and the retrain limit, with
  markers where a new champion started serving and the shock window shaded
- **Live bias:** trailing-7-day bias (+ over, − under) against the ±15% retrain band
- **Drift:** Evidently score per monitored column against the drift threshold
- **Actual vs forecast:** total units per day, and a per-series drill-down showing each version's forecast
- **Tables:** monitoring log with every decision and reason, registry versions, and the
  embedded Evidently HTML report for any day

Retrain triggers (in `configs/config.yaml`), any one is enough:

- live WAPE > 1.25x the champion's backtest WAPE (given >= 100 scored points)
- live bias is past ±15% (given >= 100 scored points): the model is systematically
  under- or over-forecasting
- the target (`sales`) drifts, or >= 50% of monitored columns drift
- drift and bias are ignored for 7 days after a retrain, so a fresh model isn't retrained
  again on the same change while its windows still hold the old model's forecasts and
  pre-change data
- the champion's data is more than 35 days old (capped at the 28-day horizon in replays,
  so there are never days without forecasts to score)

Every monitoring run is stored in `data/monitoring.sqlite` and each drift
check writes an Evidently HTML report to `reports/`.

### Run on real M5 data

Download from the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
competition and put these three files in `data/raw/` (overwriting any synthetic sample):

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Then run `forecast ingest` and `forecast backtest` as above. Scope (state, category,
start date) and backtest settings live in [`configs/config.yaml`](configs/config.yaml).

Sized for an 8 GB laptop. Measured on an M5-shaped dataset of 4,000 series x 1,941 days
(real CA FOODS is ~5,800 series): ingest takes ~11 s with a 0.8 GB peak, and one LightGBM
fold (600 rounds, 730 training days, ~2.9M rows) takes ~8 min with a 1.9 GB peak. To get
there, ingest builds the long table from int16 arrays instead of `melt`, features are
vectorized across all series (no per-series Python), and training builds features in
chunks of series straight into one float32 matrix that is freed once LightGBM has binned
it. Tests pin every one of these to the original simple implementations.

## Docker

```bash
docker compose up -d --build
```

| Service | What | URL |
|---|---|---|
| `postgres` | metadata store for MLflow and Prefect | |
| `mlflow` | tracking server + model registry, serves artifacts from a volume | http://localhost:5000 |
| `prefect-server` | orchestration API + UI | http://localhost:4200 |
| `bootstrap` | one-shot: sample data if none, ingest, train the first champion | |
| `api` | FastAPI forecasts, hot-swaps the champion | http://localhost:8000/docs |
| `worker` | runs the weekly training and daily monitoring schedules | |
| `dashboard` | Streamlit model health | http://localhost:8501 |

All app services share one image and the same `configs/config.yaml`; the
`FORECAST_TRACKING_URI` env var points them at the MLflow server. To use real M5 data, put the
CSVs in `./data/raw` and uncomment the bind mount in `docker-compose.yml`. Credentials default
to `forecast/forecast`; override with `POSTGRES_USER` / `POSTGRES_PASSWORD` in a `.env` file.

## CI (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request:

1. **Lint + fast tests + CLI smoke**: ruff, `pytest -m "not slow"`, then synthetic data ->
   ingest -> backtest of seasonal naive and LightGBM.
2. **Slow tests**: registry promotion rules, API, Prefect flows, demand-shock retrain, replay.
3. **Docker**: build the image, `docker compose up`, wait for bootstrap to train a champion,
   then hit `/predict`, MLflow, Prefect and the dashboard; on `v*` tags the image is pushed to
   GHCR.

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
  monitoring/              live accuracy, Evidently drift, retrain policy, run store, dashboard views
  registry.py              train -> register -> promote, champion loading
  simulation/              day-by-day replay with injected shocks
  serving_model.py         deployable model bundle (MLflow pyfunc)
  tracking.py              MLflow logging
  cli.py                   `forecast` command
dashboard/app.py           Streamlit model-health dashboard
tests/
```
