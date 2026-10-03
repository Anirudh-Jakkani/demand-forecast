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
    forecast simulate             # replay history with an injected shock -> sim/
    forecast dashboard            # Streamlit model-health dashboard
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from forecast.config import load_config
from forecast.data.ingest import read_processed, run_ingest
from forecast.data.synthetic import make_synthetic_m5
from forecast.data.validate import validate_long
from forecast.evaluation.backtest import run_backtest
from forecast.models import MODELS

log = logging.getLogger("forecast")


def cmd_make_sample(cfg, args) -> None:
    out = make_synthetic_m5(cfg.resolve(cfg.data.raw_dir), n_days=args.days,
                            items_per_dept=args.items)
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
    df = read_processed(cfg.resolve(cfg.data.processed_path))
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

    df = read_processed(cfg.resolve(cfg.data.processed_path))
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


def cmd_simulate(cfg, args) -> None:
    from forecast.simulation.replay import Shock, run_replay

    df_dates = read_processed(cfg.resolve(cfg.data.processed_path), ["date"])["date"]
    end = df_dates.max()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=args.days)
    shock = None
    if args.shock_factor != 1.0 or args.price_factor != 1.0:
        shock_date = (pd.Timestamp(args.shock_date) if args.shock_date
                      else start + pd.Timedelta(days=args.days // 2))
        shock = Shock(str(shock_date.date()), args.shock_factor, args.price_factor, args.dept)
    print(f"Replaying {args.days} days from {start.date()} with {args.model}; shock: {shock}")
    results = run_replay(cfg, str(start.date()), args.days, shock, args.model,
                         args.weekly_retrain)
    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    print("\nOpen the dashboard: uv run forecast dashboard")


def cmd_dashboard(cfg, args) -> None:
    import subprocess

    from forecast.config import PROJECT_ROOT

    app = PROJECT_ROOT / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app),
                    "--server.port", str(args.port), "--browser.gatherUsageStats", "false"],
                   check=False)


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to a legacy codepage; MLflow prints emoji when talking to a
    # tracking server, which would crash the command mid-run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "alembic", "mlflow.utils", "mlflow.store"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="forecast")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sample", help="write synthetic M5-shaped CSVs to data/raw")
    p.add_argument("--days", type=int, default=500)
    p.add_argument("--items", type=int, default=5, help="items per department")
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

    p = sub.add_parser("simulate", help="replay history day by day with an injected shock")
    p.add_argument("--start", default=None, help="day 0 (default: last day minus --days)")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("-m", "--model", choices=sorted(MODELS), default="lightgbm")
    p.add_argument("--shock-date", default=None, help="default: halfway through the replay")
    p.add_argument("--shock-factor", type=float, default=1.6, help="demand multiplier")
    p.add_argument("--price-factor", type=float, default=1.0, help="price multiplier")
    p.add_argument("--dept", default=None, help="limit the shock to one dept, e.g. FOODS_3")
    p.add_argument("--weekly-retrain", action="store_true",
                   help="also retrain every Monday, like the production schedule")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("dashboard", help="run the Streamlit monitoring dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    args.func(cfg, args)


if __name__ == "__main__":
    main()
