"""Command-line entry point.

    forecast make-sample          # synthetic M5-shaped data -> data/raw
    forecast ingest               # raw CSVs -> processed parquet
    forecast backtest -m seasonal_naive -m lightgbm
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from forecast.config import load_config
from forecast.data.ingest import run_ingest
from forecast.data.synthetic import make_synthetic_m5
from forecast.data.validate import validate_long
from forecast.evaluation.backtest import run_backtest
from forecast.models import MODELS

log = logging.getLogger("forecast")


def cmd_make_sample(cfg, args) -> None:
    out = make_synthetic_m5(cfg.resolve(cfg.data.raw_dir), n_days=args.days)
    log.info("Synthetic M5 sample written to %s", out)


def cmd_ingest(cfg, args) -> None:
    df = run_ingest(
        cfg.resolve(cfg.data.raw_dir),
        cfg.resolve(cfg.data.processed_path),
        state=cfg.data.state,
        category=cfg.data.category,
        start_date=cfg.data.start_date,
    )
    validate_long(df)
    log.info("Validation passed")


def cmd_backtest(cfg, args) -> None:
    df = pd.read_parquet(cfg.resolve(cfg.data.processed_path))
    validate_long(df)
    bt = cfg.backtest
    data_info = {
        "n_series": int(df["id"].nunique()),
        "data_start": str(df["date"].min().date()),
        "data_end": str(df["date"].max().date()),
        "state": cfg.data.state,
        "category": cfg.data.category,
    }
    if not args.no_mlflow:
        from forecast.tracking import setup_mlflow  # mlflow import is slow

        setup_mlflow(cfg)

    summaries = {}
    for name in args.model or ["seasonal_naive"]:
        params = cfg.model_params(name)
        log.info("Backtesting %s %s", name, params)
        start = time.perf_counter()
        result = run_backtest(lambda n=name, p=params: MODELS[n](**p), df,
                              bt.horizon, bt.n_folds, bt.step)
        elapsed = time.perf_counter() - start

        print(f"\n== {name} ({elapsed:.0f}s) ==")
        print(result.fold_metrics.to_string(index=False))
        summaries[name] = {**result.summary, "seconds": elapsed}

        if not args.no_mlflow:
            from forecast.tracking import log_backtest

            run_id = log_backtest(name, result.last_model.get_params(), result, cfg, data_info)
            log.info("Logged MLflow run %s", run_id)

    table = pd.DataFrame(summaries).T.sort_values("wape")
    print("\n== mean over folds (sorted by WAPE) ==")
    print(table.round(4).to_string())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="forecast")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sample", help="write synthetic M5-shaped CSVs to data/raw")
    p.add_argument("--days", type=int, default=500)
    p.set_defaults(func=cmd_make_sample)

    p = sub.add_parser("ingest", help="raw M5 CSVs -> processed parquet")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("backtest", help="rolling-origin backtest of one or more models")
    p.add_argument("-m", "--model", choices=sorted(MODELS), action="append",
                   help="repeatable: -m seasonal_naive -m lightgbm (default: seasonal_naive)")
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    args.func(cfg, args)


if __name__ == "__main__":
    main()
