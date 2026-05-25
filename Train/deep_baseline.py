import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

import utils.config as config
from AlphaStock.Train import Alpha_stock


ROOT = Path(__file__).resolve().parents[1]


def _market_name():
    data_path = str(config.dataset.get("ssm_data_path", "")).lower()
    return "sh" if ("沪深" in data_path or "sh" in data_path) else "nas"


def _run_command(command, cwd, logger):
    logger.info("Running deep baseline command: %s", " ".join(map(str, command)))
    child_env = os.environ.copy()
    child_env.setdefault("MKL_THREADING_LAYER", "GNU")
    child_env.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        logger.info(completed.stdout[-6000:])
    if completed.returncode != 0:
        raise RuntimeError(f"Deep baseline failed with return code {completed.returncode}: {command[1]}")


def _run_alphastock(cun_path, logger, smoke):
    old_dataset = copy.deepcopy(config.dataset)
    old_alpha = copy.deepcopy(config.alphastcok)
    date_names = [
        "train_start_date", "train_end_date", "valid_start_date",
        "valid_end_date", "test_start_date", "test_end_date",
    ]
    old_dates = {name: getattr(config, name) for name in date_names}
    try:
        config.dataset = copy.deepcopy(config.dataset)
        # The SSM feature files contain the seven adjusted-price fields used
        # by current baseline models in both markets.
        config.dataset["feature_path"] = config.dataset["ssm_data_path"]
        if smoke:
            config.alphastcok = copy.deepcopy(config.alphastcok)
            config.alphastcok.update(
                look_back=20, step_size=5, num_epoch=1, batch_size=32, num_steps=2
            )
            config.train_start_date, config.train_end_date = "2023-01-03", "2023-12-29"
            config.valid_start_date, config.valid_end_date = "2024-01-02", "2024-03-28"
            config.test_start_date, config.test_end_date = "2024-04-01", "2024-06-28"
        Alpha_stock(str(cun_path), logger)
    finally:
        config.dataset = old_dataset
        config.alphastcok = old_alpha
        for name, value in old_dates.items():
            setattr(config, name, value)


def _run_deeptrader(cun_path, logger, market, smoke):
    source_config = ROOT / "DeepTrader" / "src" / ("hyper_SH.json" if market == "sh" else "hyper_NAS.json")
    with source_config.open() as fh:
        settings = json.load(fh)
    if smoke:
        settings.update(epochs=1, max_batches=1, use_gpu=False, batch_size=8, max_steps=2)
    output_dir = Path(cun_path) / "DeepTrader" / market
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "runner_config.json"
    with config_path.open("w") as fh:
        json.dump(settings, fh, indent=2)
    _run_command(
        [
            sys.executable, "run.py", "--config", str(config_path),
            "--seed", str(getattr(config, "seed", 42)),
            "--output_dir", str(output_dir),
        ],
        ROOT / "DeepTrader" / "src",
        logger,
    )


def _prepare_deeparies_input(cun_path, market, smoke):
    source = ROOT / "DeepAries" / "data" / market / f"{market}_data.csv"
    target_dir = Path(cun_path) / "DeepAries" / market / "input" / market
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{market}_data.csv"
    raw = pd.read_csv(source)
    if "open" not in raw.columns:
        raw["open"] = raw["adjopen"]
        raw["close"] = raw["adjclose"]
        raw["high"] = raw["adjhigh"]
        raw["low"] = raw["adjlow"]
        raw["volume"] = raw["amount"]
    raw["date"] = pd.to_datetime(raw["date"])
    if smoke:
        tickers = raw["tic"].drop_duplicates().iloc[:3]
        raw = raw[
            raw["tic"].isin(tickers) &
            raw["date"].between("2023-01-03", "2024-02-16")
        ]
    raw.to_csv(target, index=False)
    return target_dir


def _run_deeparies(cun_path, logger, market, smoke):
    root_path = _prepare_deeparies_input(cun_path, market, smoke)
    output_dir = Path(cun_path) / "DeepAries" / market
    cmd = [
        sys.executable, "main.py", "--market", market, "--root_path", str(root_path),
        "--data_path", f"{market}_data.csv", "--results_root", str(output_dir / "results"),
        "--checkpoints", str(output_dir / "checkpoints"),
        "--seed", str(getattr(config, "seed", 42)),
    ]
    if smoke:
        cmd.extend([
            "--valid_year", "2023-11-01", "--test_year", "2024-01-02",
            "--seq_len", "10", "--label_len", "2", "--horizons", "1", "5",
            "--num_stocks", "3", "--d_model", "32", "--n_heads", "2",
            "--e_layers", "1", "--d_ff", "64", "--train_epochs", "1",
            "--num_workers", "0", "--max_train_steps", "8", "--rollout_len", "4",
        ])
    _run_command(cmd, ROOT / "DeepAries", logger)


def deep_baseline(cun_path, logger, smoke=False, models=None):
    """Run deep learning baselines through one market-aware entry point."""
    models = models or ("AlphaStock", "DeepTrader", "DeepAries")
    market = _market_name()
    output = (Path(cun_path) / "deep_baseline").resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger.info("------Begin Deep Baseline (%s, smoke=%s)------", market, smoke)
    for model in models:
        logger.info("Begin deep baseline model: %s", model)
        if model == "AlphaStock":
            _run_alphastock(output, logger, smoke)
        elif model == "DeepTrader":
            _run_deeptrader(output, logger, market, smoke)
        elif model == "DeepAries":
            _run_deeparies(output, logger, market, smoke)
        else:
            raise ValueError(f"Unsupported deep baseline: {model}")
    logger.info("------End Deep Baseline------")
