"""Collect table-matched baseline curves for paper figures.

This script is eval-only. It reads existing baseline outputs/checkpoints and
creates a clean, auditable folder under:

    paper_experiments_outputs/baseline_matched/

It deliberately does not modify source checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "paper_experiments_outputs" / "baseline_matched"


@dataclass(frozen=True)
class ExpectedMetric:
    total_return_pct: float
    ar_pct: float
    vol_pct: float
    sharpe: float
    maxdd_pct: float
    cr: float


EXPECTED: Dict[str, Dict[str, ExpectedMetric]] = {
    "nas": {
        "anticor": ExpectedMetric(259.97, 29.85, 35.45, 0.84, 44.59, 0.669),
        "buy_hold": ExpectedMetric(160.64, 19.80, 20.88, 0.95, 29.79, 0.665),
        "markowitz": ExpectedMetric(57.76, 9.88, 17.25, 0.57, 24.73, 0.400),
        "olmar": ExpectedMetric(157.77, 25.48, 40.14, 0.63, 45.48, 0.560),
        "ucrp": ExpectedMetric(170.39, 20.27, 19.77, 1.02, 26.63, 0.761),
        "wmamr": ExpectedMetric(264.39, 29.95, 35.32, 0.85, 33.88, 0.884),
        "alphastock": ExpectedMetric(185.35, 21.55, 21.07, 1.02, 21.90, 0.984),
        "deeparies": ExpectedMetric(162.96, 19.33, 12.88, 1.50, 10.83, 1.785),
        "deeptrader": ExpectedMetric(196.52, 23.07, 25.60, 0.90, 24.54, 0.940),
    },
    "sh": {
        "anticor": ExpectedMetric(123.82, 23.73, 38.82, 0.61, 43.33, 0.548),
        "buy_hold": ExpectedMetric(54.37, 10.87, 20.51, 0.53, 29.86, 0.364),
        "markowitz": ExpectedMetric(50.53, 11.39, 25.03, 0.45, 36.11, 0.315),
        "olmar": ExpectedMetric(180.46, 30.22, 43.60, 0.69, 45.34, 0.667),
        "ucrp": ExpectedMetric(72.58, 12.94, 19.54, 0.66, 23.56, 0.549),
        "wmamr": ExpectedMetric(117.12, 24.68, 42.75, 0.58, 50.52, 0.489),
        "alphastock": ExpectedMetric(128.39, 19.11, 21.87, 0.87, 36.37, 0.525),
        "deeparies": ExpectedMetric(148.76, 21.01, 23.36, 0.90, 15.23, 1.380),
        "deeptrader": ExpectedMetric(212.81, 26.35, 25.58, 1.03, 31.86, 0.827),
    },
}

TRADITIONAL_SOURCES = {
    "nas": {
        "anticor": REPO_ROOT / "checkpoints/nas100/anticor/portfolio_performance.csv",
        "buy_hold": REPO_ROOT / "checkpoints/nas100/BH/portfolio_performance.csv",
        "markowitz": REPO_ROOT / "checkpoints/nas100/Markowitz/portfolio_performance.csv",
        "olmar": REPO_ROOT / "checkpoints/nas100/olmar/portfolio_performance.csv",
        "wmamr": REPO_ROOT / "checkpoints/nas100/wmamr/portfolio_performance.csv",
    },
    "sh": {
        "anticor": REPO_ROOT / "checkpoints/CN_wind_复权/anticor/portfolio_performance.csv",
        "buy_hold": REPO_ROOT / "checkpoints/CN_wind_复权/BH/portfolio_performance.csv",
        "markowitz": REPO_ROOT / "checkpoints/CN_wind_复权/Markowitz/portfolio_performance.csv",
        "olmar": REPO_ROOT / "checkpoints/CN_wind_复权/olmar/portfolio_performance.csv",
        "wmamr": REPO_ROOT / "checkpoints/CN_wind_复权/wmamr/portfolio_performance.csv",
    },
}

MARKET_CONFIG = {
    "nas": {
        "feature_dir": REPO_ROOT / "Dataset/Nas100数据/feature",
        "stocks_path": REPO_ROOT / "utils/NAS100_pool.txt",
        "train_start": "2000-04-07",
        "test_start": "2020-04-23",
        "test_end": "2025-10-03",
        "checkpoint_name": "nas100",
    },
    "sh": {
        "feature_dir": REPO_ROOT / "Dataset/沪深数据/feature",
        "stocks_path": REPO_ROOT / "utils/SH_pool.txt",
        "train_start": "2000-04-07",
        "test_start": "2020-01-02",
        "test_end": "2025-02-28",
        "checkpoint_name": "CN_wind_复权",
    },
}

ALPHASTOCK = {
    "nas": {
        "actions": REPO_ROOT / "checkpoints/nas100/AlphaStock/actions/test.csv",
        "log": REPO_ROOT / "checkpoints/nas100/01-22.log",
        "seed": "46",
        "snippet_start": 1850,
        "snippet_end": 2311,
        "curve_status": "available",
    },
    "sh": {
        "actions": REPO_ROOT / "checkpoints/CN_wind_复权/AlphaStock/actions/test.csv",
        "log": REPO_ROOT / "checkpoints/CN_wind_复权/01-21.log",
        "seed": "72",
        "snippet_start": 11455,
        "snippet_end": 11761,
        "curve_status": "metric_only_missing_curve",
    },
}

DEEPARIES = {
    "nas": REPO_ROOT
    / "checkpoints/deeparies_baseline/deeparies_alphastock_aligned_5seeds/nas/seed_42/results/iTransformer_DeepAries_nas_general_num_stocks(10)_sl(20)_pl(20)_8de74e1a",
    "sh": REPO_ROOT
    / "checkpoints/deeparies_baseline/deeparies_alphastock_aligned_5seeds/sh/seed_42/results/iTransformer_DeepAries_sh_general_num_stocks(10)_sl(20)_pl(20)_c250004d",
}

DEEPTRADER = {
    "nas": {
        "output_dir": REPO_ROOT / "DeepTrader/DeepTrader/src/outputs/0121/12_12_22",
        "epoch": 6,
        "val_idx": 4465,
        "test_idx": 5045,
        "test_end_idx": 6415,
        "note": "NAS replay uses split_idx.txt test indices; the original run.py hardcoded SH indices.",
    },
    "sh": {
        "output_dir": REPO_ROOT / "DeepTrader/DeepTrader/src/outputs/0121/12_30_43",
        "epoch": 13,
        "val_idx": 4365,
        "test_idx": 4852,
        "test_end_idx": 6100,
        "note": "SH replay follows original run.py indices and matches the logged table row.",
    },
}


def ensure_dirs() -> None:
    for path in [
        OUTPUT_ROOT / "nas" / "curves",
        OUTPUT_ROOT / "sh" / "curves",
        OUTPUT_ROOT / "log_snippets",
        OUTPUT_ROOT / "manifest",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_stocks(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_adjclose_matrix(feature_dir: Path, stocks: List[str]) -> pd.DataFrame:
    frames = []
    for stock in stocks:
        stock_path = feature_dir / f"{stock}.csv"
        if not stock_path.exists():
            raise FileNotFoundError(stock_path)
        df = pd.read_csv(stock_path, usecols=["Date", "adjclose"], parse_dates=["Date"])
        df = df.rename(columns={"adjclose": stock}).set_index("Date")
        frames.append(df)
    return pd.concat(frames, axis=1, join="inner").sort_index()


def write_curve(df: pd.DataFrame, path: Path) -> Path:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return path


def read_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed") or first.lower() == "date":
        df = df.rename(columns={first: "date"})
    if "date" not in df.columns:
        raise ValueError(f"Cannot find date column in {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def standardize_curve(path: Path, out_path: Path, scale_initial: bool = False) -> Tuple[Path, pd.DataFrame]:
    src = read_curve(path)
    if "portfolio_value" not in src.columns:
        raise ValueError(f"Missing portfolio_value in {path}")
    value = pd.to_numeric(src["portfolio_value"], errors="coerce")
    if scale_initial and value.dropna().iloc[0] != 0:
        value = value / value.dropna().iloc[0]
    if "daily_return" in src.columns:
        returns = pd.to_numeric(src["daily_return"], errors="coerce")
    elif "period_return" in src.columns:
        returns = pd.to_numeric(src["period_return"], errors="coerce")
    else:
        returns = value.pct_change()
    out = pd.DataFrame({"portfolio_value": value, "daily_return": returns}, index=src.index)
    return write_curve(out, out_path), out


def max_drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    peak = clean.cummax()
    drawdown = clean / peak - 1.0
    return abs(float(drawdown.min()))


def metrics_from_curve(df: pd.DataFrame, returns_source: str = "daily_return") -> Dict[str, float]:
    values = pd.to_numeric(df["portfolio_value"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns_source in df.columns:
        returns = pd.to_numeric(df[returns_source], errors="coerce")
    else:
        returns = values.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}
    total_return = float(values.iloc[-1] - 1.0)
    ar = float(returns.mean() * 252.0) if len(returns) else float("nan")
    vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else float("nan")
    sharpe = float(ar / vol) if np.isfinite(vol) and vol > 1e-12 else float("nan")
    mdd = max_drawdown(values)
    cr = float(ar / mdd) if np.isfinite(mdd) and mdd > 1e-12 else float("nan")
    return {
        "recomputed_return_pct": total_return * 100.0,
        "recomputed_ar_pct": ar * 100.0,
        "recomputed_vol_pct": vol * 100.0,
        "recomputed_sharpe": sharpe,
        "recomputed_maxdd_pct": mdd * 100.0,
        "recomputed_cr": cr,
    }


def metrics_from_log_values(metric: ExpectedMetric) -> Dict[str, float]:
    return {
        "recomputed_return_pct": metric.total_return_pct,
        "recomputed_ar_pct": metric.ar_pct,
        "recomputed_vol_pct": metric.vol_pct,
        "recomputed_sharpe": metric.sharpe,
        "recomputed_maxdd_pct": metric.maxdd_pct,
        "recomputed_cr": metric.cr,
    }


def replay_ucrp(market: str, out_path: Path, transaction_cost: float = 0.0005) -> Tuple[Path, pd.DataFrame]:
    cfg = MARKET_CONFIG[market]
    stocks = read_stocks(cfg["stocks_path"])
    prices = load_adjclose_matrix(cfg["feature_dir"], stocks)
    prices = prices.loc[cfg["train_start"] : cfg["test_end"]]
    returns_c2c = prices.pct_change().shift(-1)
    start_pos = prices.index.get_loc(pd.Timestamp(cfg["test_start"]))
    end_pos = len(prices.index) - 1
    target = np.ones(len(stocks), dtype=np.float64) / len(stocks)
    prev_weights = None
    prev_prices = prices.iloc[start_pos - 1] if start_pos > 0 else prices.iloc[start_pos]
    rows = []
    for i in range(start_pos, end_pos + 1):
        current_prices = prices.iloc[i]
        if prev_weights is None:
            trade = target
        else:
            drift = prev_weights * (current_prices / prev_prices).to_numpy(dtype=np.float64)
            drift = drift / drift.sum()
            trade = np.abs(target - drift)
        cost = float(np.sum(trade) * transaction_cost)
        ret = float(np.dot(returns_c2c.iloc[i].to_numpy(dtype=np.float64), target) - cost)
        if np.isfinite(ret):
            rows.append((prices.index[i], ret))
        prev_weights = target
        prev_prices = current_prices
    daily = pd.Series({date: ret for date, ret in rows}, dtype="float64")
    wealth = (1.0 + daily).cumprod()
    out = pd.DataFrame({"portfolio_value": wealth, "daily_return": daily})
    return write_curve(out, out_path), out


def replay_alphastock_nas(out_path: Path) -> Tuple[Path, pd.DataFrame]:
    action_path = ALPHASTOCK["nas"]["actions"]
    weights = pd.read_csv(action_path, index_col=0, parse_dates=True)
    cfg = MARKET_CONFIG["nas"]
    stocks = list(weights.columns)
    prices = load_adjclose_matrix(cfg["feature_dir"], stocks).loc[cfg["train_start"] : cfg["test_end"]]
    returns_c2c = prices.pct_change().shift(-1).reindex(weights.index)
    returns_c2c = returns_c2c[weights.columns]
    daily = (weights * returns_c2c).sum(axis=1)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    wealth = (1.0 + daily).cumprod()
    out = pd.DataFrame({"portfolio_value": wealth, "daily_return": daily})
    return write_curve(out, out_path), out


def extract_snippet(src: Path, start: int, end: int, out: Path) -> Optional[Path]:
    if not src.exists():
        return None
    lines = src.read_text(errors="ignore").splitlines()
    snippet = lines[max(0, start - 1) : min(len(lines), end)]
    out.write_text("\n".join(snippet) + "\n")
    return out


def extract_regex_snippet(src: Path, pattern: str, out: Path, context: int = 0) -> Optional[Path]:
    if not src.exists():
        return None
    regex = re.compile(pattern)
    lines = src.read_text(errors="ignore").splitlines()
    selected: List[str] = []
    for i, line in enumerate(lines):
        if regex.search(line):
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            selected.extend(lines[start:end])
            selected.append("")
    if not selected:
        return None
    out.write_text("\n".join(selected).rstrip() + "\n")
    return out


def read_deeparies_metrics(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        "recomputed_return_pct": float(row["cumulative_return"]) * 100.0,
        "recomputed_ar_pct": float(row["annualized_return"]) * 100.0,
        "recomputed_vol_pct": float(row["annualized_volatility"]) * 100.0,
        "recomputed_sharpe": float(row["sharpe"]),
        "recomputed_maxdd_pct": abs(float(row["max_drawdown"])) * 100.0,
        "recomputed_cr": float(row["annualized_return"]) / abs(float(row["max_drawdown"])),
    }


def replay_deeptrader(market: str, out_path: Path) -> Tuple[Optional[Path], Optional[pd.DataFrame], Dict[str, float], str]:
    cfg = DEEPTRADER[market]
    src_root = REPO_ROOT / "DeepTrader/DeepTrader/src"
    if not src_root.exists():
        return None, None, {}, "DeepTrader nested src directory not found."
    old_cwd = Path.cwd()
    inserted = False
    try:
        sys.path.insert(0, str(src_root))
        inserted = True
        os.chdir(src_root)
        import torch  # type: ignore
        from agent import RLAgent  # type: ignore
        from environment.portfolio_env import PortfolioEnv  # type: ignore
        from utils.functions import calculate_daliy_metrics  # type: ignore
        from utils.parse_config import ConfigParser  # type: ignore

        output_dir: Path = cfg["output_dir"]
        with (output_dir / "log_file/hyper.json").open() as f:
            options = json.load(f)
        options["use_gpu"] = False
        args = ConfigParser(options)
        args.device = torch.device("cpu")
        data_prefix = src_root / "data" / args.market
        stocks_data = np.load(data_prefix / "features.npy")
        rate_of_return = np.load(data_prefix / "rets.npy")
        env = PortfolioEnv(
            assets_data=stocks_data,
            market_data=None,
            rtns_data=rate_of_return,
            in_features=args.in_features,
            val_idx=cfg["val_idx"],
            test_idx=cfg["test_idx"],
            test_end_idx=cfg["test_end_idx"],
            batch_size=args.batch_size,
            window_len=args.window_len,
            trade_len=args.trade_len,
            max_steps=args.max_steps,
            mode=args.mode,
            norm_type=args.norm_type,
            allow_short=False,
        )
        ckpt = output_dir / f"model_file/best_val_cr-{cfg['epoch']}.pkl"
        try:
            actor = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            actor = torch.load(ckpt, map_location="cpu")
        actor.args.device = torch.device("cpu")
        actor = actor.to(args.device)
        agent = RLAgent(env, actor, args)
        _, _, test_wealth_list, _ = agent.evaluation()
        metrics_raw = calculate_daliy_metrics(test_wealth_list, args.trade_mode)
        wealth = np.concatenate(test_wealth_list, axis=-1).reshape(-1)
        dates = pd.to_datetime((data_prefix / "dates.txt").read_text().splitlines())
        start = int(cfg["test_idx"]) + 1
        date_index = dates[start : start + len(wealth)]
        series = pd.Series(wealth, index=date_index, dtype="float64")
        out = pd.DataFrame({"portfolio_value": series, "daily_return": series.pct_change()})
        metrics = {
            "recomputed_return_pct": (float(series.iloc[-1]) - 1.0) * 100.0,
            "recomputed_ar_pct": float(np.ravel(metrics_raw["APR"])[0]) * 100.0,
            "recomputed_vol_pct": float(np.ravel(metrics_raw["AVOL"])[0]) * 100.0,
            "recomputed_sharpe": float(np.ravel(metrics_raw["ASR"])[0]),
            "recomputed_maxdd_pct": float(np.ravel(metrics_raw["MDD"])[0]) * 100.0,
            "recomputed_cr": float(np.ravel(metrics_raw["CR"])[0]),
        }
        return write_curve(out, out_path), out, metrics, cfg["note"]
    except Exception as exc:  # noqa: BLE001 - manifest records the exact failure.
        return None, None, {}, f"DeepTrader replay failed: {exc}"
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(str(src_root))
            except ValueError:
                pass


def build_row(
    market: str,
    method: str,
    curve_status: str,
    curve_path: Optional[Path],
    source_path: Optional[Path],
    metrics_source: str,
    recomputed: Dict[str, float],
    seed: str = "",
    log_snippet: Optional[Path] = None,
    notes: str = "",
) -> Dict[str, object]:
    expected = EXPECTED[market][method]
    ret = recomputed.get("recomputed_return_pct", float("nan"))
    return {
        "market": market,
        "method": method,
        "table_return_pct": expected.total_return_pct,
        "table_ar_pct": expected.ar_pct,
        "table_vol_pct": expected.vol_pct,
        "table_sharpe": expected.sharpe,
        "table_maxdd_pct": expected.maxdd_pct,
        "table_cr": expected.cr,
        "recomputed_return_pct": recomputed.get("recomputed_return_pct", ""),
        "recomputed_ar_pct": recomputed.get("recomputed_ar_pct", ""),
        "recomputed_vol_pct": recomputed.get("recomputed_vol_pct", ""),
        "recomputed_sharpe": recomputed.get("recomputed_sharpe", ""),
        "recomputed_maxdd_pct": recomputed.get("recomputed_maxdd_pct", ""),
        "recomputed_cr": recomputed.get("recomputed_cr", ""),
        "return_abs_diff_pct": abs(float(ret) - expected.total_return_pct) if np.isfinite(ret) else "",
        "curve_status": curve_status,
        "curve_path": relpath(curve_path),
        "source_path": relpath(source_path),
        "metrics_source": metrics_source,
        "log_snippet_path": relpath(log_snippet),
        "seed": seed,
        "notes": notes,
    }


def write_readme(rows: List[Dict[str, object]]) -> None:
    available = sum(1 for row in rows if row["curve_status"] == "available")
    metric_only = sum(1 for row in rows if row["curve_status"] != "available")
    text = f"""# Baseline Matched Results

这个目录只管理和论文表格数值能对应上的 baseline 结果，不混入消融实验。

## 目录

- `nas/curves/`: NAS 市场可用曲线。
- `sh/curves/`: SH 市场可用曲线。
- `log_snippets/`: 用于定位表格数值的关键日志片段。
- `manifest/baseline_sources.csv`: 每个 baseline 的来源、表格指标、重算指标和备注。

## 当前状态

- 可用曲线数量: {available}
- 仅指标/暂缺曲线数量: {metric_only}
- AlphaStock NAS 对应 seed 46，可用当前 action 曲线。
- AlphaStock SH 对应 seed 72，但当前 `actions/test.csv` 已被后续 seed 覆盖，所以先标记为 `metric_only_missing_curve`。
- DeepTrader 不复制大 checkpoint，只记录 checkpoint 路径并通过 eval-only replay 导出曲线。

## 使用建议

主实验画总收益曲线时优先读取 `curve_status=available` 的记录；若某个 baseline 是
`metric_only_missing_curve`，只用于柱形指标或表格，不要画成收益曲线。
"""
    (OUTPUT_ROOT / "README.md").write_text(text)
    for market, label in [("nas", "NAS"), ("sh", "SH")]:
        subset = [row for row in rows if row["market"] == market]
        curve_names = [row["method"] for row in subset if row["curve_status"] == "available"]
        missing = [row["method"] for row in subset if row["curve_status"] != "available"]
        market_text = f"""# {label} Curves

本目录存放 {label} 市场中已经和论文表格对齐的 baseline 收益曲线。

- 可画曲线: {", ".join(curve_names) if curve_names else "none"}
- 暂缺曲线/仅指标: {", ".join(missing) if missing else "none"}

读取 `../manifest/baseline_sources.csv` 可以看到每条曲线的原始来源、重算收益和备注。
"""
        (OUTPUT_ROOT / market / "README.md").write_text(market_text)

    (OUTPUT_ROOT / "manifest" / "README.md").write_text(
        """# Manifest

`baseline_sources.csv` 是本目录的索引表。核心字段：

- `curve_status`: `available` 表示可以用于收益曲线，其他状态只用于指标或溯源。
- `table_*`: 论文表格中的指标。
- `recomputed_*`: 从曲线、日志或指标文件重算/读取的对应值。
- `source_path` / `log_snippet_path`: 原始结果和关键日志位置。
"""
    )
    (OUTPUT_ROOT / "log_snippets" / "README.md").write_text(
        """# Log Snippets

这里保存能够定位表格数值的关键日志片段，主要用于 AlphaStock 和 DeepTrader 的结果溯源。
"""
    )


def main() -> None:
    ensure_dirs()
    rows: List[Dict[str, object]] = []

    # Traditional baselines with already saved daily portfolio traces.
    for market, methods in TRADITIONAL_SOURCES.items():
        for method, src in methods.items():
            dst = OUTPUT_ROOT / market / "curves" / f"{method}.csv"
            curve_path, df = standardize_curve(src, dst)
            rows.append(
                build_row(
                    market,
                    method,
                    "available",
                    curve_path,
                    src,
                    "portfolio_performance.csv",
                    metrics_from_curve(df),
                    notes="Copied from matched traditional baseline output.",
                )
            )

    # UCRP deterministic replay with the historical cost used by the table.
    for market in ["nas", "sh"]:
        dst = OUTPUT_ROOT / market / "curves" / "ucrp.csv"
        curve_path, df = replay_ucrp(market, dst, transaction_cost=0.0005)
        rows.append(
            build_row(
                market,
                "ucrp",
                "available",
                curve_path,
                None,
                "deterministic replay, transaction_cost=0.0005",
                metrics_from_curve(df),
                notes="Replayed because the original UCRP CSV was missing.",
            )
        )

    # AlphaStock: NAS curve is currently recoverable; SH table seed is only in logs.
    nas_alpha_dst = OUTPUT_ROOT / "nas" / "curves" / "alphastock.csv"
    nas_alpha_curve, nas_alpha_df = replay_alphastock_nas(nas_alpha_dst)
    nas_snip = extract_snippet(
        ALPHASTOCK["nas"]["log"],
        ALPHASTOCK["nas"]["snippet_start"],
        ALPHASTOCK["nas"]["snippet_end"],
        OUTPUT_ROOT / "log_snippets" / "alphastock_nasdaq100.txt",
    )
    rows.append(
        build_row(
            "nas",
            "alphastock",
            "available",
            nas_alpha_curve,
            ALPHASTOCK["nas"]["actions"],
            "replayed from current AlphaStock actions/test.csv",
            metrics_from_curve(nas_alpha_df),
            seed=ALPHASTOCK["nas"]["seed"],
            log_snippet=nas_snip,
            notes="Log table row: checkpoints/nas100/01-22.log seed 46.",
        )
    )

    sh_snip = extract_snippet(
        ALPHASTOCK["sh"]["log"],
        ALPHASTOCK["sh"]["snippet_start"],
        ALPHASTOCK["sh"]["snippet_end"],
        OUTPUT_ROOT / "log_snippets" / "alphastock_csi300.txt",
    )
    rows.append(
        build_row(
            "sh",
            "alphastock",
            "metric_only_missing_curve",
            None,
            ALPHASTOCK["sh"]["log"],
            "log snippet only",
            metrics_from_log_values(EXPECTED["sh"]["alphastock"]),
            seed=ALPHASTOCK["sh"]["seed"],
            log_snippet=sh_snip,
            notes="Current SH actions/test.csv is overwritten by a later seed and replays to 51.34%, not the table's 128.39%.",
        )
    )

    # DeepAries aligned seed 42.
    for market, src_dir in DEEPARIES.items():
        src_curve = src_dir / "portfolio_performance.csv"
        src_metrics = src_dir / "backtest_metrics.csv"
        dst = OUTPUT_ROOT / market / "curves" / "deeparies.csv"
        curve_path, _ = standardize_curve(src_curve, dst, scale_initial=True)
        rows.append(
            build_row(
                market,
                "deeparies",
                "available",
                curve_path,
                src_curve,
                relpath(src_metrics),
                read_deeparies_metrics(src_metrics),
                seed="42",
                notes="Aligned 10-stock DeepAries seed 42 result.",
            )
        )

    # DeepTrader eval-only replay from matched checkpoints.
    for market in ["nas", "sh"]:
        dst = OUTPUT_ROOT / market / "curves" / "deeptrader.csv"
        curve_path, _df, metrics, note = replay_deeptrader(market, dst)
        cfg = DEEPTRADER[market]
        output_dir = cfg["output_dir"]
        ckpt = output_dir / f"model_file/best_val_cr-{cfg['epoch']}.pkl"
        log = output_dir / "log_file/logger.log"
        snip = extract_regex_snippet(
            log,
            rf"after training {cfg['epoch']} round",
            OUTPUT_ROOT / "log_snippets" / f"deeptrader_{market}_epoch{cfg['epoch']}.txt",
        )
        rows.append(
            build_row(
                market,
                "deeptrader",
                "available" if curve_path else "checkpoint_only_replay_failed",
                curve_path,
                ckpt,
                relpath(log),
                metrics if metrics else metrics_from_log_values(EXPECTED[market]["deeptrader"]),
                seed="-1",
                log_snippet=snip,
                notes=note,
            )
        )

    manifest = pd.DataFrame(rows)
    manifest_path = OUTPUT_ROOT / "manifest" / "baseline_sources.csv"
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_readme(rows)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {OUTPUT_ROOT / 'README.md'}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
