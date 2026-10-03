#!/usr/bin/env bash
# One-shot setup run by the `bootstrap` service before the API/worker/dashboard start.
# Idempotent: safe to run on every `docker compose up`.
set -euo pipefail

if [ ! -f data/raw/sales_train_evaluation.csv ]; then
    echo "No M5 data in data/raw: generating the synthetic sample"
    forecast make-sample --items 25
fi

if [ ! -f data/processed/m5_ca_foods.parquet ]; then
    forecast ingest
fi

if forecast versions | grep -q champion; then
    echo "A champion model is already registered"
else
    echo "No champion yet: running the training flow"
    forecast pipeline train
fi
