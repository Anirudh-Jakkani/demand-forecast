"""Command-line entry point.

    forecast make-sample          # synthetic M5-shaped data -> data/raw
    forecast ingest               # raw CSVs -> processed parquet
    forecast backtest -m seasonal_naive -m lightgbm
    forecast train -m lightgbm    # backtest, fit on all data, register, maybe promote
    forecast versions             # registry: versions, aliases, scores
    forecast promote 3            # manual promote / rollback
    forecast serve                # FastAPI on http://127.0.0.1:8000/docs
    forecast pipeline train [--as-of 2016-04-24]     # Prefect training flow, run once
    forecast pipeline monitor [--as-of 2016-05-01]   # Prefect monitoring flow, run once
    forecast pipeline serve       # schedule both flows (needs `prefect server start`)
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


def cmd_train(cfg, args) -> None:
    from forecast.registry import train_and_register

    df = pd.read_parquet(cfg.resolve(cfg.data.processed_path))
    validate_long(df)
    out = train_and_register(cfg, df, args.model, promote=not args.no_promote)
    status = "PROMOTED to champion" if out.promoted else "not promoted"
    print(f"\n{cfg.registry.model_name} v{out.version} ({args.model}): "
          f"cv WAPE {out.cv_wape:.4f} -> {status}\n  reason: {out.reason}")


def cmd_versions(cfg, args) -> None:
    from forecast.registry import list_versions

    table = list_versions(cfg)
    print(table.to_string(index=False) if len(table) else "No registered versions yet.")


def cmd_promote(cfg, args) -> None:
    from forecast.registry import promote_version

    promote_version(cfg, args.version)
    print(f"{cfg.registry.model_name} v{args.version} is now the champion")


def cmd_serve(cfg, args) -> None:
    import uvicorn

    from forecast.api.app import create_app

    uvicorn.run(create_app(cfg), host=args.host, port=args.port)


def cmd_pipeline(cfg, args) -> None:
    import json

    if args.action == "serve":
        from forecast.flows.deploy import serve_flows

        serve_flows(cfg, args.config, args.model)
        return
    if args.action == "train":
        from forecast.flows.training import training_flow

        result = training_flow(as_of=args.as_of, model=args.model, config_path=args.config)
    else:
        from forecast.flows.monitoring import monitoring_flow

        result = monitoring_flow(as_of=args.as_of, config_path=args.config,
                                 auto_retrain=not args.no_retrain)
    print(json.dumps(result, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "alembic", "mlflow.utils", "mlflow.store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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

    p = sub.add_parser("train", help="backtest, fit on all data, register, and maybe promote")
    p.add_argument("-m", "--model", choices=sorted(MODELS), default="lightgbm")
    p.add_argument("--no-promote", action="store_true", help="register only")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("versions", help="list registered model versions")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("promote", help="point the champion alias at a version")
    p.add_argument("version")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("serve", help="run the forecast API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("pipeline", help="run or schedule the Prefect flows")
    p.add_argument("action", choices=["train", "monitor", "serve"])
    p.add_argument("--as-of", default=None, help="pretend today is this date (YYYY-MM-DD)")
    p.add_argument("-m", "--model", choices=sorted(MODELS), default="lightgbm",
                   help="model for the training flow")
    p.add_argument("--no-retrain", action="store_true",
                   help="monitor only; don't trigger retraining")
    p.set_defaults(func=cmd_pipeline)

    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    args.func(cfg, args)


if __name__ == "__main__":
    main()
