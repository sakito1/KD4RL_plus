#!/usr/bin/env python
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import utils.config as runtime_config
import utils.config_Nas as nas_config
import utils.config_SH as sh_config
from create_DeepAries_data import save_deeparies_data


ROOT = Path(__file__).resolve().parent
MARKET_CONFIGS = {
    "sh": ("A-share", "sh", sh_config),
    "nas": ("NAS100", "nas", nas_config),
}


def apply_market_config(config_module):
    for name, value in vars(config_module).items():
        if not name.startswith("__"):
            setattr(runtime_config, name, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepAries baseline on KD4RL feature data for A-share and NAS100."
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=sorted(MARKET_CONFIGS),
        default=["sh", "nas"],
        help="Markets to run in order. Default: sh nas.",
    )
    parser.add_argument(
        "--output_root",
        default="checkpoints/deeparies_baseline",
        help="Root directory for exported inputs, logs, checkpoints, and results.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run folder name. Default: current timestamp.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch DeepAries/main.py.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed shorthand. Ignored if --seeds is set.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run each requested market once per seed, e.g. --seeds 42 43 44.",
    )
    parser.add_argument(
        "--num_stocks",
        default="alphastock",
        help="'alphastock' uses config.alphastcok.model_param.trade_num; 'all' uses the full pool; 'default' keeps DeepAries default; or pass an integer.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=None,
        help="Default: config.alphastcok['num_epoch'] for each market.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Optional per-epoch DeepAries training step cap. Default: no cap.",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Default: config.alphastcok['look_back'] for each market.",
    )
    parser.add_argument("--label_len", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--model", default="iTransformer", choices=["iTransformer", "Transformer"])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rollout_len", type=int, default=30)
    parser.add_argument(
        "--fee_rate",
        type=float,
        default=None,
        help="Default: the market TRANSACTION_COST_RATE (one basis point).",
    )
    parser.add_argument(
        "--initial_amount",
        type=float,
        default=None,
        help="Default: config.initial_amount for each market.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU by hiding CUDA devices.")
    parser.add_argument(
        "--heartbeat_seconds",
        type=int,
        default=60,
        help="Print a monitor heartbeat when DeepAries is quiet.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick connectivity run: 3 stocks, tiny model, one short epoch.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with the next market if one market fails.",
    )
    return parser.parse_args()


def stream_process(command, cwd, env, log_path, prefix, heartbeat_seconds):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(map(str, command)) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = queue.Queue()

        def reader():
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        start = time.time()
        last_output = start
        while process.poll() is None or not lines.empty():
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if now - last_output >= heartbeat_seconds:
                    elapsed = int(now - start)
                    msg = f"[{prefix}] still running, elapsed={elapsed}s, log={log_path}"
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    last_output = now
                continue

            text = f"[{prefix}] {line}"
            print(text, end="", flush=True)
            log.write(text)
            log.flush()
            last_output = time.time()

        thread.join(timeout=2.0)
        return process.returncode


def _alphastock_param(name, default=None):
    return runtime_config.alphastcok.get(name, default)


def _alphastock_trade_num(default=10):
    return runtime_config.alphastcok.get("model_param", {}).get("trade_num", default)


def effective_alphastock_aligned_settings(args, summary):
    if args.num_stocks == "alphastock":
        num_stocks = min(int(_alphastock_trade_num()), int(summary["stocks"]))
    elif args.num_stocks == "all":
        num_stocks = int(summary["stocks"])
    elif args.num_stocks == "default":
        num_stocks = "deeparies_default"
    else:
        num_stocks = int(args.num_stocks)
    return {
        "train_epochs": args.train_epochs if args.train_epochs is not None else _alphastock_param("num_epoch", 1),
        "seq_len": args.seq_len if args.seq_len is not None else _alphastock_param("look_back", 20),
        "num_stocks": num_stocks,
        "trade_num_source": _alphastock_trade_num(),
        "initial_amount": args.initial_amount if args.initial_amount is not None else getattr(runtime_config, "initial_amount", 1.0),
        "fee_rate": args.fee_rate if args.fee_rate is not None else runtime_config.TRANSACTION_COST_RATE,
        "features": list(runtime_config.dataset.get("features_name", [])),
    }


def build_deeparies_command(args, market, deep_market, run_dir, summary, seed):
    input_dir = run_dir / "input" / deep_market
    results_dir = run_dir / "results"
    checkpoints_dir = run_dir / "checkpoints"

    train_epochs = args.train_epochs if args.train_epochs is not None else _alphastock_param("num_epoch", 1)
    seq_len = args.seq_len if args.seq_len is not None else _alphastock_param("look_back", 20)
    fee_rate = args.fee_rate if args.fee_rate is not None else runtime_config.TRANSACTION_COST_RATE
    initial_amount = args.initial_amount if args.initial_amount is not None else getattr(runtime_config, "initial_amount", 1.0)

    command = [
        args.python,
        "-u",
        "main.py",
        "--market",
        deep_market,
        "--model",
        args.model,
        "--root_path",
        str(input_dir),
        "--data_path",
        f"{deep_market}_data.csv",
        "--results_root",
        str(results_dir),
        "--checkpoints",
        str(checkpoints_dir),
        "--seed",
        str(seed),
        "--valid_year",
        str(runtime_config.valid_start_date),
        "--test_year",
        str(runtime_config.test_start_date),
        "--train_start_date",
        str(runtime_config.train_start_date),
        "--train_end_date",
        str(runtime_config.train_end_date),
        "--valid_end_date",
        str(runtime_config.valid_end_date),
        "--test_end_date",
        str(runtime_config.test_end_date),
        "--seq_len",
        str(seq_len),
        "--label_len",
        str(args.label_len),
        "--horizons",
        *[str(h) for h in args.horizons],
        "--train_epochs",
        str(train_epochs),
        "--patience",
        str(args.patience),
        "--d_model",
        str(args.d_model),
        "--n_heads",
        str(args.n_heads),
        "--e_layers",
        str(args.e_layers),
        "--d_layers",
        str(args.d_layers),
        "--d_ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--learning_rate",
        str(args.learning_rate),
        "--rollout_len",
        str(args.rollout_len),
        "--fee_rate",
        str(fee_rate),
        "--initial_amount",
        str(initial_amount),
        "--temperature",
        str(args.temperature),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.gpu),
    ]
    if args.num_stocks == "alphastock":
        command.extend(["--num_stocks", str(min(int(_alphastock_trade_num()), int(summary["stocks"])))])
    elif args.num_stocks == "all":
        command.extend(["--num_stocks", str(summary["stocks"])])
    elif args.num_stocks != "default":
        command.extend(["--num_stocks", str(int(args.num_stocks))])
    if args.max_train_steps is not None:
        command.extend(["--max_train_steps", str(args.max_train_steps)])
    return command


def apply_smoke_overrides(args):
    args.num_stocks = "3"
    args.train_epochs = 1
    args.max_train_steps = 8
    args.seq_len = 10
    args.label_len = 2
    args.horizons = [1, 5]
    args.d_model = 32
    args.n_heads = 2
    args.e_layers = 1
    args.d_layers = 1
    args.d_ff = 64
    args.num_workers = 0
    args.rollout_len = 4


def resolve_seeds(args):
    if args.seeds:
        return args.seeds
    if args.seed is not None:
        return [args.seed]
    return [getattr(runtime_config, "seed", 42)]


def run_market_seed(args, market, run_root, seed):
    label, deep_market, config_module = MARKET_CONFIGS[market]
    apply_market_config(config_module)

    run_dir = run_root / market / f"seed_{seed}"
    input_dir = run_dir / "input" / deep_market
    stocks_limit = 3 if args.smoke else None

    prefix = f"{market}-s{seed}"
    print(f"\n===== DeepAries baseline: {label} ({market}), seed={seed} =====", flush=True)
    print(f"[{prefix}] source feature: {runtime_config.dataset['feature_path']}", flush=True)
    print(
        f"[{prefix}] split: train [{runtime_config.train_start_date}, {runtime_config.train_end_date}], "
        f"val [{runtime_config.valid_start_date}, {runtime_config.valid_end_date}], "
        f"test [{runtime_config.test_start_date}, {runtime_config.test_end_date}]",
        flush=True,
    )

    summary = save_deeparies_data(
        market=deep_market,
        output_root=input_dir,
        start_date=runtime_config.train_start_date,
        end_date=runtime_config.test_end_date,
        stocks_limit=stocks_limit,
    )
    aligned_settings = effective_alphastock_aligned_settings(args, summary)

    metadata = {
        "market": market,
        "seed": seed,
        "label": label,
        "summary": summary,
        "alphastock_aligned_settings": aligned_settings,
        "split": {
            "train_start": runtime_config.train_start_date,
            "train_end": runtime_config.train_end_date,
            "valid_start": runtime_config.valid_start_date,
            "valid_end": runtime_config.valid_end_date,
            "test_start": runtime_config.test_start_date,
            "test_end": runtime_config.test_end_date,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "input_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(
        f"[{prefix}] exported {summary['stocks']} stocks, {summary['dates']} dates, "
        f"features={summary['feature_cols']}",
        flush=True,
    )
    print(f"[{prefix}] AlphaStock-aligned settings: {aligned_settings}", flush=True)

    command = build_deeparies_command(args, market, deep_market, run_dir, summary, seed)
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
    env["PYTHONUNBUFFERED"] = "1"
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""

    command_path = run_dir / "command.json"
    with command_path.open("w", encoding="utf-8") as fh:
        json.dump({"command": command, "cwd": str(ROOT / "DeepAries")}, fh, ensure_ascii=False, indent=2)

    returncode = stream_process(
        command,
        cwd=ROOT / "DeepAries",
        env=env,
        log_path=run_dir / "deeparies.log",
        prefix=prefix,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    if returncode != 0:
        raise RuntimeError(f"DeepAries failed for {market}; see {run_dir / 'deeparies.log'}")
    print(f"[{prefix}] completed. Outputs: {run_dir}", flush=True)


def write_seed_summary(run_root):
    rows = []
    for metrics_path in sorted(run_root.glob("*/seed_*/results/*/backtest_metrics.csv")):
        rel = metrics_path.relative_to(run_root)
        market = rel.parts[0]
        seed = rel.parts[1].replace("seed_", "")
        setting = rel.parts[3]
        with metrics_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row = {
                    "market": market,
                    "seed": seed,
                    "setting": setting,
                    **row,
                    "metrics_path": str(metrics_path),
                }
                rows.append(row)
    if not rows:
        return None

    summary_path = run_root / "deeparies_seed_summary.csv"
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main():
    args = parse_args()
    if args.smoke:
        apply_smoke_overrides(args)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT / args.output_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"DeepAries baseline run root: {run_root}", flush=True)

    failures = []
    for market in args.markets:
        _, _, config_module = MARKET_CONFIGS[market]
        apply_market_config(config_module)
        seeds = resolve_seeds(args)
        for seed in seeds:
            try:
                run_market_seed(args, market, run_root, seed)
            except Exception as exc:
                run_id = f"{market}/seed_{seed}"
                failures.append((run_id, str(exc)))
                print(f"[{run_id}] FAILED: {exc}", flush=True)
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break

    if failures:
        summary_path = write_seed_summary(run_root)
        if summary_path:
            print(f"\nPartial seed summary: {summary_path}", flush=True)
        print("\nFailures:", flush=True)
        for market, error in failures:
            print(f"  {market}: {error}", flush=True)
        sys.exit(1)

    summary_path = write_seed_summary(run_root)
    if summary_path:
        print(f"\nSeed summary: {summary_path}", flush=True)
    print("\nAll requested DeepAries baseline runs completed.", flush=True)


if __name__ == "__main__":
    main()
