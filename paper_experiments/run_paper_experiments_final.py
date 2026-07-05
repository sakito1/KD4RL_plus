#!/usr/bin/env python3
"""Generate final paper figures for matched baselines and explanations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from paper_experiments.metrics import compute_financial_metrics, summarize_all
from paper_experiments.trace_utils import discover_runs, parse_seed_specs


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D9DEE7",
        "axes.labelcolor": "#253142",
        "xtick.color": "#3B4657",
        "ytick.color": "#3B4657",
        "grid.color": "#E8ECF3",
        "grid.linewidth": 0.8,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)


MARKET_LABELS = {"nas": "Nasdaq-100", "sh": "CSI-300"}
DEFAULT_SEEDS = {"nas": 49, "sh": 90}
SCENARIO_LABELS = {
    "fixed_hrl_no_inner": "Outer-only",
    "fixed_hrl": "Outer + Inner",
    "controller_outer": "Outer + Controller",
    "full_controller": "Ours",
}
CORE_SCENARIOS = ["fixed_hrl_no_inner", "fixed_hrl", "controller_outer", "full_controller"]
METHOD_LABELS = {
    "anticor": "Anticor",
    "buy_hold": "Buy&Hold",
    "markowitz": "Markowitz",
    "olmar": "OLMAR",
    "ucrp": "UCRP",
    "wmamr": "WMAMR",
    "alphastock": "AlphaStock",
    "deeparies": "DeepAries",
    "deeptrader": "DeepTrader",
}
BASELINE_COLORS = {
    "anticor": "#4C78A8",
    "buy_hold": "#7F8796",
    "markowitz": "#72B7B2",
    "olmar": "#59A14F",
    "ucrp": "#B279A2",
    "wmamr": "#E0A33A",
    "alphastock": "#9C755F",
    "deeparies": "#2F9C95",
    "deeptrader": "#F28E2B",
}
OURS_COLOR = "#B63A4A"
CONTROLLER_COLOR = "#2A9D8F"
HOLD_COLOR = "#8B93A1"
INNER_COLOR = "#3B6FB6"
FIXED_COLORS = ["#5E6C84", "#8E7CC3", "#B07AA1", "#76B7B2", "#EDC948"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval-only final paper experiments.")
    parser.add_argument("--results_root", default=str(ROOT / "results" / "end"))
    parser.add_argument("--baseline_dir", default=str(ROOT / "paper_experiments_outputs" / "baseline_matched"))
    parser.add_argument("--end2end_dir", default=str(ROOT / "paper_experiments_outputs" / "end_to_end_explain"))
    parser.add_argument("--output_dir", default=str(ROOT / "paper_experiments_outputs" / "paper_experiments_final"))
    parser.add_argument("--markets", nargs="+", default=["nas", "sh"], choices=["nas", "sh"])
    parser.add_argument("--seeds", nargs="*", default=["nas:49", "sh:90"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixed_windows", nargs="+", type=int, default=[5, 10, 20, 30, 60])
    parser.add_argument("--test_max_days", type=int, default=None)
    parser.add_argument("--force_fixed_eval", action="store_true")
    parser.add_argument("--skip_fixed_eval", action="store_true")
    parser.add_argument("--counterfactual_horizon", type=int, default=30)
    parser.add_argument("--force_counterfactual_eval", action="store_true")
    parser.add_argument("--case_count", type=int, default=2)
    return parser.parse_args()


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "main": root / "01_main_experiment",
        "ablation": root / "02_ablation",
        "controller": root / "03_controller_interpretability",
        "inner": root / "04_inner_actor_interpretability",
        "tables": root / "tables",
        "cache": root / "_cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D9DEE7")
    ax.spines["bottom"].set_color("#D9DEE7")


def ensure_calmar(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure CR/Calmar is annualized return divided by maximum drawdown."""
    out = df.copy()
    ar = pd.to_numeric(out.get("annualized_return"), errors="coerce")
    mdd = pd.to_numeric(out.get("max_drawdown"), errors="coerce")
    computed = ar / mdd.replace(0, np.nan)
    if "calmar" in out:
        current = pd.to_numeric(out["calmar"], errors="coerce")
        out["calmar"] = current.where(current.notna(), computed)
    else:
        out["calmar"] = computed
    return out


def format_scaled(value: float, *, scale: float = 1.0, suffix: str = "", decimals: int = 2, signed: bool = False) -> str:
    if pd.isna(value):
        return ""
    number = float(value) * float(scale)
    sign = "+" if signed else ""
    return f"{number:{sign}.{int(decimals)}f}{suffix}"


def write_display_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    percent_cols: Sequence[str] = (),
    decimal_cols: Sequence[str] = (),
    int_cols: Sequence[str] = (),
    four_decimal_cols: Sequence[str] = (),
) -> None:
    out = df.copy()
    for col in percent_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: format_scaled(x, scale=100.0, suffix="%"))
    for col in decimal_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: format_scaled(x))
    for col in four_decimal_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: format_scaled(x, decimals=4))
    for col in int_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(round(float(x)))))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def read_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df:
        df["date"] = parse_dates(df["date"])
    values = pd.to_numeric(df["portfolio_value"], errors="coerce")
    if values.dropna().empty:
        raise ValueError(f"empty portfolio_value in {path}")
    scale = 1000.0 if float(values.dropna().iloc[0]) > 50.0 else 1.0
    out = pd.DataFrame({"date": df.get("date", pd.RangeIndex(len(df))), "wealth": values / scale})
    out["wealth"] = out["wealth"].astype(float)
    if "daily_simple_return" in df:
        out["daily_return"] = pd.to_numeric(df["daily_simple_return"], errors="coerce")
    elif "daily_return" in df:
        out["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")
    else:
        out["daily_return"] = out["wealth"].pct_change()
    return out.dropna(subset=["wealth"]).reset_index(drop=True)


def read_portfolio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df:
        df["date"] = parse_dates(df["date"])
    return df


FINANCIAL_COLS = [
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "final_value",
    "daily_win_rate",
]


def recompute_financial_from_trace(path: Path) -> Dict[str, float]:
    return compute_financial_metrics(read_portfolio(path))


def update_financial_fields(row: Dict[str, object], metrics: Dict[str, float]) -> Dict[str, object]:
    out = dict(row)
    for col in FINANCIAL_COLS:
        if col in metrics:
            out[col] = metrics[col]
    return out


def baseline_manifest(baseline_dir: Path) -> pd.DataFrame:
    path = baseline_dir / "manifest" / "baseline_sources.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing matched baseline manifest: {path}")
    return pd.read_csv(path)


def seed_for_market(seed_map: Dict[str, List[int]], market: str) -> int:
    values = seed_map.get(market) or [DEFAULT_SEEDS[market]]
    return int(values[0])


def load_ours_metric(end2end_dir: Path, market: str, seed: int) -> Dict[str, float]:
    curve_path = end2end_dir / "traces" / f"{market}_seed{seed}_full_controller_portfolio.csv"
    metrics_path = end2end_dir / "metrics" / "inference_ablation.csv"
    row_data: Dict[str, object] = {}
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        row = metrics[
            (metrics["market"] == market)
            & (metrics["seed"].astype(int) == int(seed))
            & (metrics["scenario"] == "full_controller")
        ]
        if not row.empty:
            row_data = row.iloc[0].to_dict()
    if curve_path.exists():
        return update_financial_fields(row_data, recompute_financial_from_trace(curve_path))
    return row_data


def load_main_metrics(manifest: pd.DataFrame, end2end_dir: Path, market: str, seed: int) -> pd.DataFrame:
    rows = []
    matched = manifest[manifest["market"] == market].copy()
    matched = matched[pd.to_numeric(matched["recomputed_return_pct"], errors="coerce").notna()]
    for _, row in matched.iterrows():
        method = str(row["method"])
        rows.append(
            {
                "market": market,
                "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "total_return": float(row["recomputed_return_pct"]) / 100.0,
                "annualized_return": float(row["recomputed_ar_pct"]) / 100.0,
                "annualized_volatility": float(row["recomputed_vol_pct"]) / 100.0,
                "sharpe": float(row["recomputed_sharpe"]),
                "max_drawdown": float(row["recomputed_maxdd_pct"]) / 100.0,
                "calmar": float(row["recomputed_cr"]) if "recomputed_cr" in row and pd.notna(row["recomputed_cr"]) else np.nan,
                "source": "matched_baseline",
            }
        )
    ours = load_ours_metric(end2end_dir, market, seed)
    rows.append(
        {
            "market": market,
            "method_key": "ours",
            "method": "Ours",
            "total_return": float(ours.get("total_return", np.nan)),
            "annualized_return": float(ours.get("annualized_return", np.nan)),
            "annualized_volatility": float(ours.get("annualized_volatility", np.nan)),
            "sharpe": float(ours.get("sharpe", np.nan)),
            "max_drawdown": float(ours.get("max_drawdown", np.nan)),
            "calmar": float(ours.get("calmar", np.nan)),
            "source": "ours_full_controller",
        }
    )
    return ensure_calmar(pd.DataFrame(rows))


def plot_main_equity(manifest: pd.DataFrame, end2end_dir: Path, market: str, seed: int, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    matched = manifest[(manifest["market"] == market) & (manifest["curve_status"] == "available")].copy()
    for _, row in matched.iterrows():
        method = str(row["method"])
        curve = read_curve(ROOT / str(row["curve_path"]))
        ax.plot(
            curve["date"],
            curve["wealth"],
            lw=1.65,
            alpha=0.82,
            color=BASELINE_COLORS.get(method, "#8B93A1"),
            label=METHOD_LABELS.get(method, method),
        )
    ours_path = end2end_dir / "traces" / f"{market}_seed{seed}_full_controller_portfolio.csv"
    ours = read_curve(ours_path)
    ax.plot(ours["date"], ours["wealth"], lw=3.0, color=OURS_COLOR, label="Ours", zorder=5)
    ax.set_title(f"{MARKET_LABELS[market]} Portfolio Wealth")
    ax.set_ylabel("Wealth multiple")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    clean_axis(ax)
    ax.grid(True, axis="both", alpha=0.65)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    save_figure(fig, out_dir / f"main_equity_{market}")


def plot_metric_panels(df: pd.DataFrame, path_base: Path, title: str) -> None:
    df = ensure_calmar(df)
    order = df[df["method"] != "Ours"].sort_values("total_return", ascending=True)["method"].tolist()
    if (df["method"] == "Ours").any():
        order.append("Ours")
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(16.2, max(5.2, 0.38 * len(order) + 2.0)), sharey=True)
    metric_specs = [
        ("total_return", "Total return ↑", 100.0, "%", "higher is better", False),
        ("sharpe", "Sharpe ratio ↑", 1.0, "", "higher is better", False),
        ("max_drawdown", "Max drawdown ↓", 100.0, "%", "lower is better", True),
        ("calmar", "CR ↑", 1.0, "", "return / drawdown", False),
    ]
    indexed = df.set_index("method")
    for ax, (metric, label, scale, suffix, direction, lower_is_better) in zip(axes, metric_specs):
        values = indexed.loc[order, metric].astype(float) * scale
        best_method = values.idxmin() if lower_is_better else values.idxmax()
        colors = []
        edges = []
        for method in order:
            if method == "Ours":
                colors.append(OURS_COLOR)
                edges.append("#7f1d1d")
            elif method == best_method:
                colors.append(CONTROLLER_COLOR)
                edges.append("#0f766e")
            else:
                colors.append("#8B93A1")
                edges.append("#8B93A1")
        bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=0.8, height=0.68, alpha=0.95)
        ax.set_title(label)
        ax.text(0.98, 1.035, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.1, color="#667085")
        ax.set_xlabel("%" if suffix == "%" else "ratio")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        clean_axis(ax)
        ax.grid(True, axis="x", alpha=0.55)
        ax.grid(False, axis="y")
        limit = max(float(np.nanmax(values)) if len(values) else 1.0, 1.0)
        ax.set_xlim(0, limit * 1.18)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_width() + limit * 0.025,
                bar.get_y() + bar.get_height() / 2,
                format_scaled(float(value), scale=1.0, suffix=suffix),
                va="center",
                ha="left",
                fontsize=8.5,
                color="#253142",
            )
    fig.suptitle(title, y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, path_base)


def main_experiment(manifest: pd.DataFrame, end2end_dir: Path, markets: Sequence[str], seeds: Dict[str, int], dirs: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for market in markets:
        seed = seeds[market]
        plot_main_equity(manifest, end2end_dir, market, seed, dirs["main"])
        metrics = load_main_metrics(manifest, end2end_dir, market, seed)
        rows.append(metrics)
        plot_metric_panels(metrics, dirs["main"] / f"main_metrics_{market}", f"{MARKET_LABELS[market]} Main Metrics")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(dirs["tables"] / "main_experiment_metrics_full.csv", index=False)
    out.to_csv(dirs["main"] / "main_experiment_metrics_full.csv", index=False)
    main_report = out[
        [
            col
            for col in ["market", "method_key", "method", "total_return", "sharpe", "max_drawdown", "calmar", "source"]
            if col in out
        ]
    ].copy()
    main_report.to_csv(dirs["tables"] / "main_experiment_metrics.csv", index=False)
    main_report.to_csv(dirs["main"] / "main_experiment_metrics.csv", index=False)
    main_display = main_report.copy()
    write_display_csv(
        main_display,
        dirs["tables"] / "main_experiment_metrics_display.csv",
        percent_cols=["total_return", "max_drawdown"],
        decimal_cols=["sharpe", "calmar"],
    )
    write_display_csv(
        main_display,
        dirs["main"] / "main_experiment_metrics_display.csv",
        percent_cols=["total_return", "max_drawdown"],
        decimal_cols=["sharpe", "calmar"],
    )
    return out


def fixed_cache_paths(cache_dir: Path, market: str, seed: int, window: int) -> Dict[str, Path]:
    stem = f"{market}_seed{seed}_fixed_window_{window}"
    return {
        "portfolio": cache_dir / f"{stem}_portfolio.csv",
        "actions": cache_dir / f"{stem}_actions.csv",
        "switch_events": cache_dir / f"{stem}_switch_events.csv",
    }


def ensure_fixed_window_eval(args: argparse.Namespace, markets: Sequence[str], seeds: Dict[str, int], dirs: Dict[str, Path]) -> pd.DataFrame:
    cache_dir = dirs["cache"] / "fixed_windows"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cache_dir / "fixed_window_metrics.csv"
    if args.skip_fixed_eval:
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
        return pd.DataFrame()
    if metrics_path.exists() and not args.force_fixed_eval:
        cached = pd.read_csv(metrics_path)
        expected = {(m, int(seeds[m]), int(w)) for m in markets for w in args.fixed_windows}
        have = {
            (str(r["market"]), int(r["seed"]), int(r["fixed_window_days"]))
            for _, r in cached.iterrows()
            if str(r.get("status", "ok")) == "ok"
        }
        if expected.issubset(have):
            return cached

    from paper_experiments.eval_end_to_end_explain import (
        build_loaded_trainer,
        collect_eval_trace,
        load_checkpoint_into_trainer,
    )

    seed_map = {market: [int(seeds[market])] for market in markets}
    runs = discover_runs(Path(args.results_root), markets=markets, seed_map=seed_map)
    rows = []
    for run in runs:
        trainer, _, torch_module = build_loaded_trainer(
            run,
            output_dir=dirs["cache"] / "_fixed_eval_runtime",
            device=args.device,
            results_root=Path(args.results_root),
        )
        ckpt = run.checkpoints["best_model"]
        if not load_checkpoint_into_trainer(trainer, torch_module, ckpt.path):
            for window in args.fixed_windows:
                rows.append(
                    {
                        "market": run.market,
                        "seed": run.seed,
                        "scenario": f"fixed_window_{window}",
                        "fixed_window_days": int(window),
                        "status": "missing_checkpoint",
                    }
                )
            continue
        for window in args.fixed_windows:
            paths = fixed_cache_paths(cache_dir, run.market, run.seed, int(window))
            if paths["portfolio"].exists() and paths["actions"].exists() and not args.force_fixed_eval:
                bundle = {
                    "portfolio": pd.read_csv(paths["portfolio"]),
                    "actions": pd.read_csv(paths["actions"]),
                    "switch_events": pd.read_csv(paths["switch_events"]) if paths["switch_events"].exists() else pd.DataFrame(),
                }
            else:
                bundle = collect_eval_trace(
                    trainer,
                    scenario=f"fixed_window_{window}",
                    fixed_cycle=int(window),
                    disable_inner=False,
                    test_max_days=args.test_max_days,
                )
                bundle["portfolio"].to_csv(paths["portfolio"], index=False)
                bundle["actions"].to_csv(paths["actions"], index=False)
                bundle["switch_events"].to_csv(paths["switch_events"], index=False)
            row = {
                "market": run.market,
                "seed": int(run.seed),
                "scenario": f"fixed_window_{window}",
                "fixed_window_days": int(window),
                "status": "ok",
            }
            row.update(summarize_all(bundle["portfolio"]))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(metrics_path, index=False)
    return out


def ensure_counterfactual_horizon_eval(
    args: argparse.Namespace,
    markets: Sequence[str],
    seeds: Dict[str, int],
    dirs: Dict[str, Path],
) -> Dict[str, Dict[str, Path]]:
    horizon = max(20, int(args.counterfactual_horizon))
    cache_dir = dirs["cache"] / f"counterfactual_horizon{horizon}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundles: Dict[str, Dict[str, Path]] = {}

    def paths_for(market: str, seed: int) -> Dict[str, Path]:
        stem = f"{market}_seed{seed}_full_controller_horizon{horizon}"
        return {
            "portfolio": cache_dir / f"{stem}_portfolio.csv",
            "actions": cache_dir / f"{stem}_actions.csv",
            "switch_events": cache_dir / f"{stem}_switch_events.csv",
        }

    expected = {market: paths_for(market, seeds[market]) for market in markets}
    if not args.force_counterfactual_eval:
        cached_ok = True
        for market, paths in expected.items():
            if not paths["portfolio"].exists() or not paths["actions"].exists():
                cached_ok = False
                break
            try:
                cols = pd.read_csv(paths["actions"], nrows=1).columns
                if f"hold_curve_{horizon}" not in cols:
                    cached_ok = False
                    break
            except Exception:
                cached_ok = False
                break
        if cached_ok:
            return expected

    from paper_experiments.eval_end_to_end_explain import (
        build_loaded_trainer,
        collect_eval_trace,
        load_checkpoint_into_trainer,
    )

    seed_map = {market: [int(seeds[market])] for market in markets}
    runs = discover_runs(Path(args.results_root), markets=markets, seed_map=seed_map)
    for run in runs:
        paths = paths_for(run.market, run.seed)
        trainer, _, torch_module = build_loaded_trainer(
            run,
            output_dir=dirs["cache"] / "_counterfactual_eval_runtime",
            device=args.device,
            results_root=Path(args.results_root),
        )
        ckpt = run.checkpoints["best_model"]
        if not load_checkpoint_into_trainer(trainer, torch_module, ckpt.path):
            continue
        bundle = collect_eval_trace(
            trainer,
            scenario=f"full_controller_horizon{horizon}",
            fixed_cycle=None,
            disable_inner=False,
            test_max_days=args.test_max_days,
            counterfactual_horizon=horizon,
        )
        bundle["portfolio"].to_csv(paths["portfolio"], index=False)
        bundle["actions"].to_csv(paths["actions"], index=False)
        bundle["switch_events"].to_csv(paths["switch_events"], index=False)
        bundles[run.market] = paths
    for market, paths in expected.items():
        if paths["portfolio"].exists() and paths["actions"].exists():
            bundles[market] = paths
    return bundles


def load_ablation_metrics(end2end_dir: Path, fixed_metrics: pd.DataFrame, markets: Sequence[str], seeds: Dict[str, int]) -> pd.DataFrame:
    core = pd.read_csv(end2end_dir / "metrics" / "inference_ablation.csv")
    frames = []
    for market in markets:
        seed = seeds[market]
        sub = core[(core["market"] == market) & (core["seed"].astype(int) == int(seed)) & core["scenario"].isin(CORE_SCENARIOS)].copy()
        for idx, row in sub.iterrows():
            path = end2end_dir / "traces" / f"{market}_seed{seed}_{row['scenario']}_portfolio.csv"
            if path.exists():
                for col, value in recompute_financial_from_trace(path).items():
                    sub.loc[idx, col] = value
        sub["method"] = sub["scenario"].map(SCENARIO_LABELS)
        sub["group"] = "component"
        frames.append(sub)
    if fixed_metrics is not None and not fixed_metrics.empty:
        fixed = fixed_metrics[fixed_metrics["status"].fillna("ok").eq("ok")].copy()
        fixed["method"] = fixed["fixed_window_days"].map(lambda x: f"Fixed {int(x)}d")
        fixed["group"] = "fixed_window"
        frames.append(fixed)
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep = [
        "market",
        "seed",
        "scenario",
        "method",
        "group",
        "fixed_window_days",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "switch_count",
        "free_switch_count",
    ]
    for col in keep:
        if col not in out:
            out[col] = np.nan
    return ensure_calmar(out[keep])


def refresh_ablation_financial_metrics(
    df: pd.DataFrame,
    end2end_dir: Path,
    dirs: Dict[str, Path],
    seeds: Dict[str, int],
) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        market = str(row["market"])
        seed = int(row.get("seed", seeds.get(market, DEFAULT_SEEDS.get(market, 0))))
        path = scenario_curve_path(
            end2end_dir,
            dirs["cache"],
            market,
            seed,
            str(row["scenario"]),
            row.get("fixed_window_days", np.nan),
        )
        if not path.exists():
            continue
        for col, value in recompute_financial_from_trace(path).items():
            if col in out.columns:
                out.loc[idx, col] = value
    return out


def scenario_curve_path(end2end_dir: Path, cache_dir: Path, market: str, seed: int, scenario: str, fixed_window: float = np.nan) -> Path:
    if scenario.startswith("fixed_window_"):
        return fixed_cache_paths(cache_dir / "fixed_windows", market, seed, int(fixed_window))["portfolio"]
    return end2end_dir / "traces" / f"{market}_seed{seed}_{scenario}_portfolio.csv"


def plot_ablation_equity(df: pd.DataFrame, end2end_dir: Path, dirs: Dict[str, Path], market: str, seed: int) -> None:
    sub = df[df["market"] == market].copy()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    core_colors = {
        "Outer-only": "#7F8796",
        "Outer + Inner": INNER_COLOR,
        "Outer + Controller": CONTROLLER_COLOR,
        "Ours": OURS_COLOR,
    }
    fixed_color_map = {name: color for name, color in zip(sorted(sub[sub["group"] == "fixed_window"]["method"].unique()), FIXED_COLORS)}
    for _, row in sub.iterrows():
        path = scenario_curve_path(end2end_dir, dirs["cache"], market, seed, str(row["scenario"]), row.get("fixed_window_days", np.nan))
        if not path.exists():
            continue
        curve = read_curve(path)
        method = str(row["method"])
        is_fixed = str(row["group"]) == "fixed_window"
        ax.plot(
            curve["date"],
            curve["wealth"],
            lw=1.65 if is_fixed else (3.0 if method == "Ours" else 2.3),
            ls="--" if is_fixed else "-",
            alpha=0.75 if is_fixed else 0.95,
            color=fixed_color_map.get(method, core_colors.get(method, "#8B93A1")),
            label=method,
        )
    ax.set_title(f"{MARKET_LABELS[market]} Ablation Wealth")
    ax.set_ylabel("Wealth multiple")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    clean_axis(ax)
    ax.grid(True, axis="both", alpha=0.65)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    save_figure(fig, dirs["ablation"] / f"ablation_equity_{market}")


def plot_ablation_metrics(df: pd.DataFrame, dirs: Dict[str, Path], market: str) -> None:
    sub = ensure_calmar(df[df["market"] == market]).copy()
    method_order = ["Outer-only", "Outer + Inner", "Outer + Controller"]
    method_order += sorted(sub[sub["group"] == "fixed_window"]["method"].unique(), key=lambda x: int(re.findall(r"\d+", x)[0]))
    method_order += ["Ours"]
    sub["method"] = pd.Categorical(sub["method"], categories=method_order, ordered=True)
    sub = sub.sort_values("method")
    y = np.arange(len(sub))
    colors = []
    for _, row in sub.iterrows():
        if row["method"] == "Ours":
            colors.append(OURS_COLOR)
        elif row["group"] == "fixed_window":
            colors.append("#8E7CC3")
        elif row["method"] == "Outer + Controller":
            colors.append(CONTROLLER_COLOR)
        elif row["method"] == "Outer + Inner":
            colors.append(INNER_COLOR)
        else:
            colors.append("#7F8796")
    fig, axes = plt.subplots(1, 4, figsize=(16.2, max(5.2, 0.38 * len(sub) + 2.0)), sharey=True)
    specs = [
        ("total_return", "Total return ↑", 100.0, "%", "higher is better", False),
        ("sharpe", "Sharpe ratio ↑", 1.0, "", "higher is better", False),
        ("max_drawdown", "Max drawdown ↓", 100.0, "%", "lower is better", True),
        ("calmar", "CR ↑", 1.0, "", "return / drawdown", False),
    ]
    for ax, (metric, title, scale, suffix, direction, lower_is_better) in zip(axes, specs):
        values = pd.to_numeric(sub[metric], errors="coerce") * scale
        finite_values = values.to_numpy(dtype="float64")
        best_pos = int(np.nanargmin(finite_values) if lower_is_better else np.nanargmax(finite_values)) if values.notna().any() else -1
        edgecolors = ["#111827" if i == best_pos else color for i, color in enumerate(colors)]
        bars = ax.barh(y, values, color=colors, edgecolor=edgecolors, linewidth=0.8, height=0.68)
        ax.set_title(title)
        ax.text(0.98, 1.035, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.1, color="#667085")
        ax.set_xlabel("%" if suffix == "%" else "ratio")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["method"].astype(str))
        clean_axis(ax)
        ax.grid(True, axis="x", alpha=0.55)
        ax.grid(False, axis="y")
        limit = max(float(np.nanmax(values)) if values.notna().any() else 1.0, 1.0)
        ax.set_xlim(0, limit * 1.18)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_width() + limit * 0.025,
                bar.get_y() + bar.get_height() / 2,
                format_scaled(float(value), suffix=suffix),
                va="center",
                fontsize=8.5,
            )
    fig.suptitle(f"{MARKET_LABELS[market]} Ablation Metrics", y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, dirs["ablation"] / f"ablation_metrics_{market}")


def ablation_experiment(end2end_dir: Path, fixed_metrics: pd.DataFrame, markets: Sequence[str], seeds: Dict[str, int], dirs: Dict[str, Path]) -> pd.DataFrame:
    metrics = load_ablation_metrics(end2end_dir, fixed_metrics, markets, seeds)
    metrics = refresh_ablation_financial_metrics(metrics, end2end_dir, dirs, seeds)
    metrics = ensure_calmar(metrics)
    metrics.to_csv(dirs["tables"] / "ablation_metrics_full.csv", index=False)
    metrics.to_csv(dirs["ablation"] / "ablation_metrics_full.csv", index=False)
    ablation_report = metrics[
        [
            col
            for col in [
                "market",
                "scenario",
                "method",
                "group",
                "fixed_window_days",
                "total_return",
                "sharpe",
                "max_drawdown",
                "calmar",
                "switch_count",
                "free_switch_count",
            ]
            if col in metrics
        ]
    ].copy()
    ablation_report.to_csv(dirs["tables"] / "ablation_metrics.csv", index=False)
    ablation_report.to_csv(dirs["ablation"] / "ablation_metrics.csv", index=False)
    ablation_display = ablation_report.copy()
    write_display_csv(
        ablation_display,
        dirs["tables"] / "ablation_metrics_display.csv",
        percent_cols=["total_return", "max_drawdown"],
        decimal_cols=["sharpe", "calmar"],
        int_cols=["seed", "fixed_window_days", "switch_count", "free_switch_count"],
    )
    write_display_csv(
        ablation_display,
        dirs["ablation"] / "ablation_metrics_display.csv",
        percent_cols=["total_return", "max_drawdown"],
        decimal_cols=["sharpe", "calmar"],
        int_cols=["seed", "fixed_window_days", "switch_count", "free_switch_count"],
    )
    for market in markets:
        seed = seeds[market]
        plot_ablation_equity(metrics, end2end_dir, dirs, market, seed)
        plot_ablation_metrics(metrics, dirs, market)
    return metrics


def numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_curve_json(value) -> np.ndarray:
    if isinstance(value, str) and value.strip():
        try:
            arr = np.asarray(json.loads(value), dtype="float64")
            return arr[np.isfinite(arr)]
        except Exception:
            return np.asarray([], dtype="float64")
    return np.asarray([], dtype="float64")


def max_drawdown_from_values(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max((peak - arr) / np.maximum(peak, 1e-12)))


def drawdown_curve(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.asarray([], dtype="float64")
    peak = np.maximum.accumulate(arr)
    return (peak - arr) / np.maximum(peak, 1e-12)


def select_controller_cases(actions: pd.DataFrame, portfolio: pd.DataFrame, count: int) -> pd.DataFrame:
    df = actions.copy()
    horizon = 30 if "hold_future_return_30" in df.columns and "hold_curve_30" in df.columns else 20
    hold_ret_col = f"hold_future_return_{horizon}"
    switch_ret_col = f"switch_future_return_{horizon}"
    hold_mdd_col = f"hold_future_mdd_{horizon}"
    switch_mdd_col = f"switch_future_mdd_{horizon}"
    df = numeric_cols(
        df,
        [
            "step",
            "duration_before_decision",
            "is_switch",
            "is_free_switch",
            "exit_prob",
            "switch_future_return_20",
            "hold_future_return_20",
            "switch_future_mdd_20",
            "hold_future_mdd_20",
            "switch_future_return_30",
            "hold_future_return_30",
            "switch_future_mdd_30",
            "hold_future_mdd_30",
        ],
    )
    switch_flag = "is_free_switch" if "is_free_switch" in df.columns else "is_switch"
    switched = df[(df["decision_type"] == "free_decision") & (df[switch_flag] > 0)].copy()
    if switched.empty:
        switched = df[df["is_switch"] > 0].copy()
    switched["ret_gain"] = switched[switch_ret_col] - switched[hold_ret_col]
    switched["mdd_gain"] = switched[hold_mdd_col] - switched[switch_mdd_col]
    switched["hold_loss"] = (-switched[hold_ret_col]).clip(lower=0.0)
    switched["duration_bonus"] = np.minimum(switched["duration_before_decision"].fillna(0.0), 15.0)
    switched["score"] = (
        5.0 * switched["ret_gain"].fillna(0.0)
        + 2.0 * switched["mdd_gain"].fillna(0.0)
        + 5.0 * switched["hold_loss"].fillna(0.0)
        + 0.10 * switched["exit_prob"].fillna(0.0)
        + 0.004 * switched["duration_bonus"].fillna(0.0)
    )
    max_step = int(pd.to_numeric(portfolio["step"], errors="coerce").max())
    downside = (
        ((switched[hold_ret_col] < 0.0) | (switched[hold_mdd_col] > 0.07))
        & (switched["ret_gain"] > 0.0)
    )
    candidate_pool = switched[downside].copy()
    if len(candidate_pool) < int(count):
        candidate_pool = switched.copy()
    candidates = candidate_pool.sort_values("score", ascending=False)
    selected = []
    used_steps: List[int] = []
    for _, row in candidates.iterrows():
        step = int(row["step"])
        if step > max_step - (horizon + 2):
            continue
        if any(abs(step - used) < 55 for used in used_steps):
            continue
        selected.append(row)
        used_steps.append(step)
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        for _, row in candidates.iterrows():
            step = int(row["step"])
            if any(int(row["step"]) == int(sel["step"]) for sel in selected):
                continue
            if step <= max_step - (horizon + 2):
                selected.append(row)
            if len(selected) >= int(count):
                break
    return pd.DataFrame(selected)


def plot_controller_case(market: str, case_id: int, case: pd.Series, portfolio: pd.DataFrame, actions: pd.DataFrame, out_dir: Path) -> Dict[str, float]:
    key_step = int(case["step"])
    hold_curve = parse_curve_json(case.get("hold_curve_30", ""))
    switch_curve = parse_curve_json(case.get("switch_curve_30", ""))
    if len(hold_curve) == 0 or len(switch_curve) == 0:
        hold_curve = parse_curve_json(case.get("hold_curve_20", ""))
        switch_curve = parse_curve_json(case.get("switch_curve_20", ""))
    curve_len = int(min(len(hold_curve), len(switch_curve)))
    if curve_len < 2:
        raise ValueError(f"empty counterfactual curve for {market} controller case at step {key_step}")
    hold_curve = hold_curve[:curve_len]
    switch_curve = switch_curve[:curve_len]
    days = np.arange(curve_len)
    realized_horizon = curve_len - 1

    keep_color = "#C65D4B"
    switch_color = CONTROLLER_COLOR
    probability_color = "#315F9E"
    key_date = parse_dates(pd.Series([case["date"]])).iloc[0]
    key_date_label = key_date.strftime("%Y/%m/%d") if pd.notna(key_date) else str(case.get("date", ""))
    exit_prob = float(case.get("exit_prob", np.nan))

    hold_return = float(hold_curve[-1] - 1.0)
    switch_return = float(switch_curve[-1] - 1.0)
    ret_gain = switch_return - hold_return
    hold_mdd = max_drawdown_from_values(hold_curve)
    switch_mdd = max_drawdown_from_values(switch_curve)
    mdd_gain = hold_mdd - switch_mdd
    hold_ret_path = (hold_curve - 1.0) * 100.0
    switch_ret_path = (switch_curve - 1.0) * 100.0
    hold_dd_path = drawdown_curve(hold_curve) * 100.0
    switch_dd_path = drawdown_curve(switch_curve) * 100.0

    step_series = pd.to_numeric(actions["step"], errors="coerce")
    action_window = actions[(step_series >= key_step - 10) & (step_series <= key_step + 10)].copy()
    action_window = numeric_cols(
        action_window,
        [
            "step",
            "exit_prob",
            "switch_advantage_20",
            "switch_advantage_30",
            "is_switch",
            "is_free_switch",
        ],
    )
    action_window["relative_day"] = action_window["step"] - key_step
    switches = action_window[pd.to_numeric(action_window.get("is_switch"), errors="coerce").fillna(0) > 0]

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.25, 1.65, 1.55], hspace=0.50)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    fig.suptitle(
        f"{MARKET_LABELS[market]} Controller Switch Case",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=16.5,
        fontweight="semibold",
        color="#1F2937",
    )
    fig.text(
        0.055,
        0.952,
        f"Key decision {key_date_label} | p(switch)={exit_prob:.2f} | frozen {realized_horizon}-trading-day counterfactual",
        ha="left",
        va="top",
        fontsize=9.4,
        color="#526071",
    )

    ax0.plot(days, hold_ret_path, color=keep_color, lw=2.5, label="No-controller keep")
    ax0.plot(days, switch_ret_path, color=switch_color, lw=2.8, label="Controller switch")
    ax0.fill_between(
        days,
        hold_ret_path,
        switch_ret_path,
        where=switch_ret_path >= hold_ret_path,
        color="#BFE6DD",
        alpha=0.58,
        interpolate=True,
        label="Switch advantage area",
    )
    ax0.fill_between(days, hold_ret_path, switch_ret_path, where=switch_ret_path < hold_ret_path, color="#F4C7BE", alpha=0.32, interpolate=True)
    ax0.axhline(0, color="#CBD2DD", lw=1.0)
    ax0.scatter([realized_horizon], [hold_ret_path[-1]], color=keep_color, s=48, zorder=4, edgecolor="white", linewidth=0.8)
    ax0.scatter([realized_horizon], [switch_ret_path[-1]], color=switch_color, s=48, zorder=4, edgecolor="white", linewidth=0.8)
    ax0.annotate(
        f"keep {hold_return * 100:+.2f}%",
        xy=(realized_horizon, hold_ret_path[-1]),
        xytext=(-66, -14 if hold_ret_path[-1] <= switch_ret_path[-1] else 12),
        textcoords="offset points",
        fontsize=9.0,
        color=keep_color,
    )
    ax0.annotate(
        f"switch {switch_return * 100:+.2f}%",
        xy=(realized_horizon, switch_ret_path[-1]),
        xytext=(-76, 10 if hold_ret_path[-1] <= switch_ret_path[-1] else -16),
        textcoords="offset points",
        fontsize=9.0,
        color=switch_color,
        fontweight="semibold",
    )
    ax0.text(0.02, 0.06, f"Return gap: {ret_gain * 100:+.2f} pp", transform=ax0.transAxes, ha="left", va="bottom", fontsize=10.0, color="#1F2937", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D9DEE7", "alpha": 0.96})
    ax0.set_title("A. Future return after the switch decision", loc="left", fontsize=11.0, pad=6)
    ax0.set_ylabel("Return (%)")
    ax0.set_xlim(0, realized_horizon)
    clean_axis(ax0)
    ax0.grid(True, axis="both", alpha=0.60)
    ax0.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.4)

    ax1.plot(days, hold_dd_path, color=keep_color, lw=2.3, label="No-controller keep")
    ax1.plot(days, switch_dd_path, color=switch_color, lw=2.6, label="Controller switch")
    ax1.fill_between(days, switch_dd_path, hold_dd_path, where=hold_dd_path >= switch_dd_path, color="#F1B7AB", alpha=0.40, interpolate=True, label="Avoided drawdown")
    hold_mdd_day = int(np.nanargmax(hold_dd_path)) if len(hold_dd_path) else realized_horizon
    switch_mdd_day = int(np.nanargmax(switch_dd_path)) if len(switch_dd_path) else realized_horizon
    ax1.scatter([hold_mdd_day], [hold_dd_path[hold_mdd_day]], color=keep_color, s=44, zorder=4, edgecolor="white", linewidth=0.8)
    ax1.scatter([switch_mdd_day], [switch_dd_path[switch_mdd_day]], color=switch_color, s=44, zorder=4, edgecolor="white", linewidth=0.8)
    ax1.text(0.02, 0.84, f"Max DD dots: {hold_mdd * 100:.2f}% -> {switch_mdd * 100:.2f}%\nReduction: {mdd_gain * 100:+.2f} pp", transform=ax1.transAxes, ha="left", va="top", fontsize=9.5, color="#1F2937", bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D9DEE7", "alpha": 0.96})
    ax1.set_title("B. Future drawdown under the same frozen window", loc="left", fontsize=11.0, pad=6)
    ax1.set_ylabel("Drawdown (%)")
    ax1.set_xlim(0, realized_horizon)
    ax1.set_ylim(bottom=0)
    clean_axis(ax1)
    ax1.grid(True, axis="both", alpha=0.60)

    ax2.set_title("C. Day-0 decision evidence for the key switch", loc="left", fontsize=11.0, pad=6)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    card_specs = [
        {
            "x": 0.02,
            "title": "Switch probability",
            "value": f"{exit_prob:.2f}",
            "sub": "threshold = 0.50",
            "color": probability_color,
            "kind": "prob",
        },
        {
            "x": 0.355,
            "title": "30d return gap",
            "value": f"{ret_gain * 100:+.2f} pp",
            "sub": "switch - keep",
            "color": switch_color if ret_gain >= 0 else keep_color,
            "kind": "metric",
        },
        {
            "x": 0.69,
            "title": "30d MDD reduction",
            "value": f"{mdd_gain * 100:+.2f} pp",
            "sub": "keep MDD - switch MDD",
            "color": switch_color if mdd_gain >= 0 else keep_color,
            "kind": "metric",
        },
    ]
    for spec in card_specs:
        x0 = spec["x"]
        box = FancyBboxPatch(
            (x0, 0.22),
            0.29,
            0.58,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#FFFFFF",
            edgecolor="#D9DEE7",
            linewidth=1.0,
        )
        ax2.add_patch(box)
        ax2.text(x0 + 0.025, 0.70, spec["title"], ha="left", va="center", fontsize=8.8, color="#526071")
        ax2.text(x0 + 0.025, 0.53, spec["value"], ha="left", va="center", fontsize=16.0, fontweight="semibold", color=spec["color"])
        ax2.text(x0 + 0.025, 0.34, spec["sub"], ha="left", va="center", fontsize=8.2, color="#6B7280")
        if spec["kind"] == "prob":
            bar_x = x0 + 0.025
            bar_y = 0.26
            bar_w = 0.24
            ax2.plot([bar_x, bar_x + bar_w], [bar_y, bar_y], color="#CBD2DD", lw=6, solid_capstyle="round")
            fill_w = bar_w * min(max(exit_prob if np.isfinite(exit_prob) else 0.0, 0.0), 1.0)
            ax2.plot([bar_x, bar_x + fill_w], [bar_y, bar_y], color=probability_color, lw=6, solid_capstyle="round")
            threshold_x = bar_x + bar_w * 0.5
            ax2.plot([threshold_x, threshold_x], [bar_y - 0.045, bar_y + 0.045], color="#8A95A6", lw=1.0, linestyle=(0, (3, 2)))
    ax2.text(
        0.02,
        0.05,
        "Day 0 is the selected actual switch. Panels A and B freeze the two alternatives from this same day for 30 trading days.",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#526071",
    )
    fig.subplots_adjust(left=0.11, right=0.88, top=0.90, bottom=0.08)
    save_figure(fig, out_dir / f"controller_case_{market}_{case_id:02d}")

    return {
        "market": market,
        "case_id": case_id,
        "key_date": str(case["date"]),
        "key_step": key_step,
        "case_horizon": int(realized_horizon),
        "signal_window_start": int(action_window["step"].min()) if not action_window.empty else key_step,
        "signal_window_end": int(action_window["step"].max()) if not action_window.empty else key_step,
        "switches_in_window": int(pd.to_numeric(switches["is_switch"], errors="coerce").fillna(0).sum()) if not switches.empty else 0,
        "free_switches_in_window": int(pd.to_numeric(switches.get("is_free_switch", 0), errors="coerce").fillna(0).sum()) if not switches.empty else 0,
        "hold_future_return_20": float(case.get("hold_future_return_20", np.nan)),
        "switch_future_return_20": float(case.get("switch_future_return_20", np.nan)),
        "hold_future_mdd_20": float(case.get("hold_future_mdd_20", np.nan)),
        "switch_future_mdd_20": float(case.get("switch_future_mdd_20", np.nan)),
        "ret_gain_20": float(case.get("switch_future_return_20", np.nan) - case.get("hold_future_return_20", np.nan)),
        "mdd_gain_20": float(case.get("hold_future_mdd_20", np.nan) - case.get("switch_future_mdd_20", np.nan)),
        "hold_future_return_30": hold_return,
        "switch_future_return_30": switch_return,
        "hold_future_mdd_30": hold_mdd,
        "switch_future_mdd_30": switch_mdd,
        "ret_gain_30": ret_gain,
        "mdd_gain_30": mdd_gain,
        "exit_prob": float(case.get("exit_prob", np.nan)),
        "score": float(case.get("score", np.nan)),
    }


def plot_switch_distribution(market: str, actions: pd.DataFrame, out_dir: Path) -> Dict[str, float]:
    df = actions.copy()
    df = numeric_cols(df, ["is_switch", "is_free_switch", "hold_future_return_20", "switch_future_return_20", "hold_future_mdd_20", "switch_future_mdd_20"])
    switched = df[(df["decision_type"] == "free_decision") & (df["is_switch"] > 0)].copy()
    if switched.empty:
        switched = df[df["is_switch"] > 0].copy()
    switched["ret_gain"] = switched["switch_future_return_20"] - switched["hold_future_return_20"]
    switched["mdd_gain"] = switched["hold_future_mdd_20"] - switched["switch_future_mdd_20"]
    data = [
        switched["hold_future_return_20"].dropna().values * 100.0,
        switched["switch_future_return_20"].dropna().values * 100.0,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    viol = axes[0].violinplot(data, showmeans=True, showextrema=False)
    for idx, body in enumerate(viol["bodies"]):
        body.set_facecolor([HOLD_COLOR, CONTROLLER_COLOR][idx])
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    viol["cmeans"].set_color("#253142")
    axes[0].boxplot(data, widths=0.16, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": "#253142"}, medianprops={"color": "#253142"})
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Keep", "Switch"])
    axes[0].set_ylabel("20-day future return (%)")
    axes[0].set_title("Counterfactual returns")
    clean_axis(axes[0])
    axes[0].grid(True, axis="y", alpha=0.65)

    gain = switched["ret_gain"].dropna() * 100.0
    axes[1].hist(gain, bins=28, color=CONTROLLER_COLOR, alpha=0.78, edgecolor="white")
    axes[1].axvline(0, color="#253142", lw=1.1)
    axes[1].axvline(float(gain.mean()) if len(gain) else 0.0, color=OURS_COLOR, lw=2.2, label="Mean")
    axes[1].set_title("Switch advantage distribution")
    axes[1].set_xlabel("Switch - keep return (pp)")
    axes[1].set_ylabel("Switch decisions")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])
    fig.suptitle(f"{MARKET_LABELS[market]} Switch Counterfactuals", y=1.02, fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_dir / f"switch_counterfactual_distribution_{market}")
    return {
        "market": market,
        "n_switch_decisions": int(len(switched)),
        "mean_hold_future_return_20": float(switched["hold_future_return_20"].mean()),
        "mean_switch_future_return_20": float(switched["switch_future_return_20"].mean()),
        "mean_ret_gain_20": float(switched["ret_gain"].mean()),
        "positive_ret_gain_ratio": float((switched["ret_gain"] > 0).mean()),
        "mean_mdd_gain_20": float(switched["mdd_gain"].mean()),
        "positive_mdd_gain_ratio": float((switched["mdd_gain"] > 0).mean()),
    }


def compute_switch_remaining_horizon_distribution(actions: pd.DataFrame, max_hold: int = 30, horizon: int = 30) -> pd.DataFrame:
    horizon = int(horizon)
    max_hold = int(max_hold)
    hold_curve_col = f"hold_curve_{horizon}" if f"hold_curve_{horizon}" in actions.columns else "hold_curve_20"
    switch_curve_col = f"switch_curve_{horizon}" if f"switch_curve_{horizon}" in actions.columns else "switch_curve_20"
    df = actions.copy()
    df = numeric_cols(df, ["step", "duration_before_decision", "is_switch", "is_free_switch", "exit_prob"])
    df["step"] = df["step"].astype(int)
    free_switches = df[(df["decision_type"] == "free_decision") & (df["is_free_switch"] > 0)].copy()
    rows = []
    for _, event in free_switches.iterrows():
        duration = float(event.get("duration_before_decision", np.nan))
        if not np.isfinite(duration):
            continue
        remaining = int(max(1, min(horizon, max_hold - int(duration))))
        hold_curve = parse_curve_json(event.get(hold_curve_col, ""))
        switch_curve = parse_curve_json(event.get(switch_curve_col, ""))
        if len(hold_curve) <= remaining or len(switch_curve) <= remaining:
            continue
        hold_return = float(hold_curve[remaining] - 1.0)
        switch_return = float(switch_curve[remaining] - 1.0)
        hold_mdd = max_drawdown_from_values(hold_curve[: remaining + 1])
        switch_mdd = max_drawdown_from_values(switch_curve[: remaining + 1])
        rows.append(
            {
                "date": event.get("date"),
                "step": int(event["step"]),
                "duration_before_decision": int(duration),
                "remaining_holding_days": remaining,
                "exit_prob": float(event.get("exit_prob", np.nan)),
                "counterfactual_hold_return_to_original_end": hold_return,
                "switch_return_to_original_end": switch_return,
                "switch_minus_counterfactual_hold": switch_return - hold_return,
                "counterfactual_hold_mdd_to_original_end": hold_mdd,
                "switch_mdd_to_original_end": switch_mdd,
                "counterfactual_mdd_minus_switch_mdd": hold_mdd - switch_mdd,
                "curve_horizon": horizon if hold_curve_col.endswith(str(horizon)) else 20,
            }
        )
    return pd.DataFrame(rows)


def plot_switch_remaining_horizon_distribution(
    market: str,
    actions: pd.DataFrame,
    out_dir: Path,
    *,
    max_hold: int = 30,
    horizon: int = 30,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    dist = compute_switch_remaining_horizon_distribution(actions, max_hold=max_hold, horizon=horizon)
    dist.to_csv(out_dir / f"switch_remaining_horizon_counterfactual_distribution_{market}.csv", index=False)
    if dist.empty:
        return {"market": market, "n_remaining_horizon_switches": 0}, dist

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.0, 1.2, 1.0]})
    return_data = [
        dist["counterfactual_hold_return_to_original_end"].dropna().values * 100.0,
        dist["switch_return_to_original_end"].dropna().values * 100.0,
    ]
    viol = axes[0].violinplot(return_data, showmeans=True, showextrema=False)
    for idx, body in enumerate(viol["bodies"]):
        body.set_facecolor([HOLD_COLOR, CONTROLLER_COLOR][idx])
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    viol["cmeans"].set_color("#253142")
    axes[0].boxplot(
        return_data,
        widths=0.16,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#253142"},
        medianprops={"color": "#253142"},
    )
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Keep", "Switch"])
    axes[0].set_ylabel("Return to original hold end (%)")
    axes[0].set_title("Frozen decision returns")
    clean_axis(axes[0])

    gain = dist["switch_minus_counterfactual_hold"].dropna() * 100.0
    axes[1].hist(gain, bins=28, color=CONTROLLER_COLOR, alpha=0.78, edgecolor="white")
    axes[1].axvline(0, color="#253142", lw=1.1)
    axes[1].axvline(float(gain.mean()) if len(gain) else 0.0, color=OURS_COLOR, lw=2.2, label="Mean")
    axes[1].set_title("Switch advantage")
    axes[1].set_xlabel("Switch - keep return (pp)")
    axes[1].set_ylabel("Switch decisions")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])

    mdd_data = [
        dist["counterfactual_hold_mdd_to_original_end"].dropna().values * 100.0,
        dist["switch_mdd_to_original_end"].dropna().values * 100.0,
    ]
    viol2 = axes[2].violinplot(mdd_data, showmeans=True, showextrema=False)
    for idx, body in enumerate(viol2["bodies"]):
        body.set_facecolor([HOLD_COLOR, CONTROLLER_COLOR][idx])
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    viol2["cmeans"].set_color("#253142")
    axes[2].boxplot(
        mdd_data,
        widths=0.16,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#253142"},
        medianprops={"color": "#253142"},
    )
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(["Keep", "Switch"])
    axes[2].set_ylabel("Drawdown to original hold end (%)")
    axes[2].set_title("Frozen decision drawdown")
    clean_axis(axes[2])
    fig.suptitle(f"{MARKET_LABELS[market]} Switch Decision Counterfactuals", y=1.02, fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_dir / f"switch_remaining_horizon_counterfactual_distribution_{market}")

    total_free_switches = int(
        (
            (pd.to_numeric(actions.get("is_free_switch"), errors="coerce").fillna(0) > 0)
            & (actions.get("decision_type") == "free_decision")
        ).sum()
    )
    return (
        {
            "market": market,
            "n_remaining_horizon_switches": int(len(dist)),
            "total_free_switches": total_free_switches,
            "remaining_horizon_coverage_ratio": float(len(dist) / max(total_free_switches, 1)),
            "mean_remaining_holding_days": float(dist["remaining_holding_days"].mean()),
            "mean_switch_return_to_original_end": float(dist["switch_return_to_original_end"].mean()),
            "mean_counterfactual_hold_return_to_original_end": float(dist["counterfactual_hold_return_to_original_end"].mean()),
            "mean_switch_gain_to_original_end": float(dist["switch_minus_counterfactual_hold"].mean()),
            "positive_switch_gain_ratio": float((dist["switch_minus_counterfactual_hold"] > 0).mean()),
            "mean_switch_mdd_to_original_end": float(dist["switch_mdd_to_original_end"].mean()),
            "mean_counterfactual_hold_mdd_to_original_end": float(dist["counterfactual_hold_mdd_to_original_end"].mean()),
            "positive_mdd_improvement_ratio": float((dist["counterfactual_mdd_minus_switch_mdd"] > 0).mean()),
        },
        dist,
    )


def plot_probability_resonance(market: str, actions: pd.DataFrame, out_dir: Path) -> Dict[str, float]:
    df = actions.copy()
    df["date"] = parse_dates(df["date"])
    df = numeric_cols(df, ["exit_prob", "switch_advantage_20", "is_switch", "switch_future_return_20", "hold_future_return_20"])
    free = df[df["decision_type"] == "free_decision"].copy()
    free["ret_gain"] = free["switch_future_return_20"] - free["hold_future_return_20"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.4), height_ratios=[1.55, 1.25])
    roll = max(10, min(30, len(free) // 20))
    axes[0].plot(free["date"], free["exit_prob"].rolling(roll, min_periods=3).mean(), color="#4C78A8", lw=2.1, label="Exit probability")
    axes0b = axes[0].twinx()
    axes0b.plot(free["date"], (free["ret_gain"] * 100.0).rolling(roll, min_periods=3).mean(), color=CONTROLLER_COLOR, lw=2.0, label="Future advantage")
    switch_dates = free.loc[free["is_switch"] > 0, "date"]
    for dt in switch_dates:
        axes[0].axvline(dt, color=OURS_COLOR, lw=0.45, alpha=0.13)
    axes[0].set_title("Probability moves with future switching advantage")
    axes[0].set_ylabel("Exit probability")
    axes0b.set_ylabel("Advantage (pp)")
    axes[0].set_ylim(0, 1.0)
    clean_axis(axes[0])
    axes[0].grid(True, axis="both", alpha=0.55)
    axes0b.spines["top"].set_visible(False)
    axes0b.spines["right"].set_color("#D9DEE7")
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[0].xaxis.get_major_locator()))

    q = min(5, max(2, free["exit_prob"].dropna().nunique()))
    binned = free.dropna(subset=["exit_prob", "ret_gain"]).copy()
    binned["prob_bin"] = pd.qcut(binned["exit_prob"], q=q, duplicates="drop")
    grouped = binned.groupby("prob_bin", observed=False).agg(
        exit_prob=("exit_prob", "mean"),
        ret_gain=("ret_gain", "mean"),
        switch_rate=("is_switch", "mean"),
        n=("ret_gain", "size"),
    )
    axes[1].bar(grouped["exit_prob"], grouped["ret_gain"] * 100.0, width=0.055, color=CONTROLLER_COLOR, alpha=0.75, label="Mean advantage")
    axes[1].plot(grouped["exit_prob"], grouped["switch_rate"] * 100.0, color=OURS_COLOR, marker="o", lw=2.2, label="Switch rate")
    axes[1].axhline(0, color="#253142", lw=1.0)
    axes[1].set_xlabel("Exit probability bin mean")
    axes[1].set_ylabel("Advantage / switch rate")
    axes[1].set_title("Binned controller response")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])
    axes[1].grid(True, axis="both", alpha=0.55)
    fig.suptitle(f"{MARKET_LABELS[market]} Controller Probability Resonance", y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out_dir / f"controller_probability_resonance_{market}")
    corr = free[["exit_prob", "ret_gain"]].dropna().corr().iloc[0, 1] if len(free[["exit_prob", "ret_gain"]].dropna()) > 2 else np.nan
    return {
        "market": market,
        "free_decisions": int(len(free)),
        "free_switches": int((free["is_switch"] > 0).sum()),
        "exit_prob_mean": float(free["exit_prob"].mean()),
        "ret_gain_mean": float(free["ret_gain"].mean()),
        "corr_exit_prob_ret_gain": float(corr) if np.isfinite(corr) else np.nan,
    }


def plot_random_switch(market: str, random_comparison: pd.DataFrame, out_dir: Path) -> None:
    row = random_comparison[random_comparison["market"] == market]
    if row.empty:
        return
    row = row.iloc[0]
    specs = [
        ("Return ↑", "full_total_return", "random_mean_total_return", "random_std_total_return", 100.0, "%", "higher is better"),
        ("Sharpe ↑", "full_sharpe", "random_mean_sharpe", "random_std_sharpe", 1.0, "", "higher is better"),
        ("MaxDD ↓", "full_mdd", "random_mean_mdd", "random_std_mdd", 100.0, "%", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.5))
    for ax, (title, ours_col, rand_col, err_col, scale, suffix, direction) in zip(axes, specs):
        ours = float(row[ours_col]) * scale
        rand = float(row[rand_col]) * scale
        err = float(row[err_col]) * scale
        bars = ax.bar([0, 1], [ours, rand], color=[OURS_COLOR, "#8B93A1"], width=0.58, yerr=[0.0, err], capsize=4, ecolor="#4B5563")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Ours", "Random"])
        ax.set_title(title)
        ax.text(0.98, 1.035, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.1, color="#667085")
        ax.set_ylabel("%" if suffix == "%" else "ratio")
        limit = max(ours, rand + err, 1.0)
        ax.set_ylim(0, limit * 1.22)
        clean_axis(ax)
        ax.grid(True, axis="y", alpha=0.65)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + limit * 0.035,
                format_scaled(float(bar.get_height()), suffix=suffix),
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#253142",
            )
        if err > 0:
            ax.text(1, rand + err + limit * 0.035, "±1 std", ha="center", va="bottom", fontsize=8.0, color="#4B5563")
    fig.suptitle(f"{MARKET_LABELS[market]} Controller vs Random Switching", y=1.02, fontsize=14, fontweight="semibold")
    fig.text(
        0.5,
        0.010,
        "Random policies use the same switch budget; error bars show one standard deviation across sampled random schedules.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#667085",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.98))
    save_figure(fig, out_dir / f"random_switch_comparison_{market}")


def plot_fixed_window_comparison(market: str, ablation_metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = ensure_calmar(ablation_metrics[ablation_metrics["market"] == market]).copy()
    fixed = sub[sub["group"] == "fixed_window"].copy()
    ours = sub[sub["method"] == "Ours"].copy()
    if fixed.empty or ours.empty:
        return
    fixed["window_order"] = pd.to_numeric(fixed["fixed_window_days"], errors="coerce")
    fixed = fixed.sort_values("window_order")
    rows = pd.concat([ours.head(1), fixed], ignore_index=True)
    labels = ["Ours" if method == "Ours" else f"{int(window)}d" for method, window in zip(rows["method"], rows["fixed_window_days"])]
    colors = [OURS_COLOR if label == "Ours" else "#8E7CC3" for label in labels]
    specs = [
        ("TR ↑", "total_return", 100.0, "%", "higher is better", False),
        ("Sharpe ↑", "sharpe", 1.0, "", "higher is better", False),
        ("MDD ↓", "max_drawdown", 100.0, "%", "lower is better", True),
        ("CR ↑", "calmar", 1.0, "", "higher is better", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7))
    axes = axes.ravel()
    x = np.arange(len(rows))
    for ax, (title, metric, scale, suffix, direction, lower_is_better) in zip(axes, specs):
        values = pd.to_numeric(rows[metric], errors="coerce") * scale
        arr = values.to_numpy(dtype="float64")
        best_idx = int(np.nanargmin(arr) if lower_is_better else np.nanargmax(arr))
        edgecolors = ["#111827" if i == best_idx else color for i, color in enumerate(colors)]
        bars = ax.bar(x, values, color=colors, edgecolor=edgecolors, linewidth=0.85, width=0.64)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(title)
        ax.text(0.98, 1.035, direction, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color="#667085")
        ax.set_ylabel("%" if suffix == "%" else "ratio")
        limit = max(float(np.nanmax(values)) if values.notna().any() else 1.0, 1.0)
        ax.set_ylim(0, limit * 1.36)
        clean_axis(ax)
        ax.grid(True, axis="y", alpha=0.65)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + limit * 0.035,
                format_scaled(float(value), suffix=suffix),
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#253142",
                rotation=90,
            )
    fig.suptitle(f"{MARKET_LABELS[market]} Learned Controller vs Fixed Holding Windows", y=0.995, fontsize=14, fontweight="semibold")
    fig.text(
        0.5,
        0.010,
        "Fixed windows force rebalancing every 5/10/20/30/60 trading days; the learned controller evaluates the holding state daily.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#667085",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.955), h_pad=1.8, w_pad=1.4)
    save_figure(fig, out_dir / f"fixed_window_comparison_{market}")


def recompute_random_switch_comparison(
    end2end_dir: Path,
    markets: Sequence[str],
    seeds: Dict[str, int],
    existing: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    trace_dir = end2end_dir / "traces"
    existing = existing.copy() if existing is not None else pd.DataFrame()
    for market in markets:
        seed = int(seeds[market])
        row = {}
        old = existing[(existing.get("market", pd.Series(dtype=str)) == market)] if not existing.empty else pd.DataFrame()
        if not old.empty:
            row.update(old.iloc[0].to_dict())
        row.update({"market": market, "seed": seed})
        full_path = trace_dir / f"{market}_seed{seed}_full_controller_portfolio.csv"
        if full_path.exists():
            full = recompute_financial_from_trace(full_path)
            row.update(
                {
                    "full_total_return": full.get("total_return", np.nan),
                    "full_sharpe": full.get("sharpe", np.nan),
                    "full_mdd": full.get("max_drawdown", np.nan),
                }
            )
        random_metrics = []
        for path in sorted(trace_dir.glob(f"{market}_seed{seed}_random_switch_matched_count_*_portfolio.csv")):
            metrics = recompute_financial_from_trace(path)
            random_metrics.append(metrics)
        if random_metrics:
            rand = pd.DataFrame(random_metrics)
            row.update(
                {
                    "random_mean_total_return": float(rand["total_return"].mean()),
                    "random_std_total_return": float(rand["total_return"].std(ddof=1)),
                    "random_mean_sharpe": float(rand["sharpe"].mean()),
                    "random_std_sharpe": float(rand["sharpe"].std(ddof=1)),
                    "random_mean_mdd": float(rand["max_drawdown"].mean()),
                    "random_std_mdd": float(rand["max_drawdown"].std(ddof=1)),
                    "random_run_count": int(len(rand)),
                }
            )
            if "full_total_return" in row:
                row["full_percentile_by_return"] = float((rand["total_return"] <= row["full_total_return"]).mean())
            if "full_sharpe" in row:
                row["full_percentile_by_sharpe"] = float((rand["sharpe"] <= row["full_sharpe"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def controller_experiment(
    end2end_dir: Path,
    markets: Sequence[str],
    seeds: Dict[str, int],
    dirs: Dict[str, Path],
    case_count: int,
    *,
    ablation_metrics: pd.DataFrame = None,
    counterfactual_bundles: Dict[str, Dict[str, Path]] = None,
    counterfactual_horizon: int = 30,
) -> pd.DataFrame:
    case_rows = []
    summary_rows = []
    remaining_rows = []
    remaining_frames = []
    counterfactual_bundles = counterfactual_bundles or {}
    for market in markets:
        seed = seeds[market]
        portfolio = read_portfolio(end2end_dir / "traces" / f"{market}_seed{seed}_full_controller_portfolio.csv")
        actions = read_portfolio(end2end_dir / "traces" / f"{market}_seed{seed}_full_controller_actions.csv")
        actions = numeric_cols(actions, ["step", "is_switch", "is_free_switch"])
        counterfactual_paths = counterfactual_bundles.get(market)
        if counterfactual_paths and counterfactual_paths.get("actions") and counterfactual_paths["actions"].exists():
            case_actions = read_portfolio(counterfactual_paths["actions"])
            case_actions = numeric_cols(case_actions, ["step", "is_switch", "is_free_switch"])
        else:
            case_actions = actions
        cases = select_controller_cases(case_actions, portfolio, case_count)
        cases.to_csv(dirs["controller"] / f"selected_controller_cases_{market}.csv", index=False)
        for idx, (_, case) in enumerate(cases.iterrows(), start=1):
            case_rows.append(plot_controller_case(market, idx, case, portfolio, case_actions, dirs["controller"]))
        switch_summary = plot_switch_distribution(market, actions, dirs["controller"])
        prob_summary = plot_probability_resonance(market, actions, dirs["controller"])
        summary_rows.append({**switch_summary, **prob_summary})
        if counterfactual_paths and counterfactual_paths.get("actions") and counterfactual_paths["actions"].exists():
            counterfactual_actions = read_portfolio(counterfactual_paths["actions"])
        else:
            counterfactual_actions = actions
        remaining_summary, remaining_dist = plot_switch_remaining_horizon_distribution(
            market,
            counterfactual_actions,
            dirs["controller"],
            max_hold=counterfactual_horizon,
            horizon=counterfactual_horizon,
        )
        remaining_rows.append(remaining_summary)
        if not remaining_dist.empty:
            remaining_dist.insert(0, "market", market)
            remaining_frames.append(remaining_dist)
        if ablation_metrics is not None and not ablation_metrics.empty:
            plot_fixed_window_comparison(market, ablation_metrics, dirs["controller"])
    case_df = pd.DataFrame(case_rows)
    summary_df = pd.DataFrame(summary_rows)
    remaining_summary_df = pd.DataFrame(remaining_rows)
    remaining_dist_df = pd.concat(remaining_frames, ignore_index=True) if remaining_frames else pd.DataFrame()
    case_df.to_csv(dirs["controller"] / "controller_case_summary.csv", index=False)
    summary_df.to_csv(dirs["controller"] / "controller_statistical_summary.csv", index=False)
    remaining_summary_df.to_csv(dirs["controller"] / "switch_remaining_horizon_summary.csv", index=False)
    remaining_dist_df.to_csv(dirs["controller"] / "switch_remaining_horizon_distribution_all.csv", index=False)
    case_df.to_csv(dirs["tables"] / "controller_case_summary.csv", index=False)
    summary_df.to_csv(dirs["tables"] / "controller_statistical_summary.csv", index=False)
    remaining_summary_df.to_csv(dirs["tables"] / "switch_remaining_horizon_summary.csv", index=False)
    remaining_dist_df.to_csv(dirs["tables"] / "switch_remaining_horizon_distribution_all.csv", index=False)
    case_percent_cols = [
        "hold_future_return_20",
        "switch_future_return_20",
        "hold_future_mdd_20",
        "switch_future_mdd_20",
        "ret_gain_20",
        "mdd_gain_20",
        "hold_future_return_30",
        "switch_future_return_30",
        "hold_future_mdd_30",
        "switch_future_mdd_30",
        "ret_gain_30",
        "mdd_gain_30",
    ]
    summary_percent_cols = [
        "mean_hold_future_return_20",
        "mean_switch_future_return_20",
        "mean_ret_gain_20",
        "positive_ret_gain_ratio",
        "mean_mdd_gain_20",
        "positive_mdd_gain_ratio",
        "ret_gain_mean",
    ]
    for base in [dirs["controller"], dirs["tables"]]:
        write_display_csv(
            case_df,
            base / "controller_case_summary_display.csv",
            percent_cols=case_percent_cols,
            decimal_cols=["exit_prob"],
            four_decimal_cols=["score"],
            int_cols=["case_id", "key_step", "case_horizon", "signal_window_start", "signal_window_end", "switches_in_window", "free_switches_in_window"],
        )
        write_display_csv(
            summary_df,
            base / "controller_statistical_summary_display.csv",
            percent_cols=summary_percent_cols,
            decimal_cols=["exit_prob_mean", "corr_exit_prob_ret_gain"],
            int_cols=["n_switch_decisions", "free_decisions", "free_switches"],
        )
        write_display_csv(
            remaining_summary_df,
            base / "switch_remaining_horizon_summary_display.csv",
            percent_cols=[
                "remaining_horizon_coverage_ratio",
                "mean_switch_return_to_original_end",
                "mean_counterfactual_hold_return_to_original_end",
                "mean_switch_gain_to_original_end",
                "positive_switch_gain_ratio",
                "mean_switch_mdd_to_original_end",
                "mean_counterfactual_hold_mdd_to_original_end",
                "positive_mdd_improvement_ratio",
            ],
            decimal_cols=["mean_remaining_holding_days"],
            int_cols=["n_remaining_horizon_switches", "total_free_switches"],
        )
        write_display_csv(
            remaining_dist_df,
            base / "switch_remaining_horizon_distribution_all_display.csv",
            percent_cols=[
                "counterfactual_hold_return_to_original_end",
                "switch_return_to_original_end",
                "switch_minus_counterfactual_hold",
                "counterfactual_hold_mdd_to_original_end",
                "switch_mdd_to_original_end",
                "counterfactual_mdd_minus_switch_mdd",
            ],
            decimal_cols=["exit_prob"],
            int_cols=["step", "duration_before_decision", "remaining_holding_days", "curve_horizon"],
        )
    return case_df


def parse_weight_text(text) -> Dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {}
    out = {}
    for part in text.split(";"):
        if ":" not in part:
            continue
        asset, value = part.split(":", 1)
        asset = asset.strip()
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if not asset or not match:
            continue
        out[asset] = float(match.group(0)) / 100.0
    return out


def plot_weight_stack(market: str, case_step: int, interpretability_dir: Path, out_dir: Path) -> None:
    path = interpretability_dir / f"scenario_traces_{market}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[(df["scenario"] == "Controller+HRL") & (pd.to_numeric(df["step"], errors="coerce") == int(case_step))]
    if df.empty:
        return
    row = df.iloc[0]
    hold = parse_weight_text(row.get("hold_top_weights", ""))
    switch = parse_weight_text(row.get("switch_top_weights", ""))
    assets = list(dict.fromkeys(list(hold.keys()) + list(switch.keys())))
    if not assets:
        return
    cmap = plt.get_cmap("tab20")
    colors = {asset: cmap(i % 20) for i, asset in enumerate(assets)}
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    lefts = np.zeros(2)
    labels = ["Keep candidate", "Switch candidate"]
    for asset in assets:
        vals = np.array([hold.get(asset, 0.0), switch.get(asset, 0.0)]) * 100.0
        ax.barh([1, 0], vals, left=lefts, color=colors[asset], edgecolor="white", linewidth=0.8, label=asset)
        for y, left, val in zip([1, 0], lefts, vals):
            if val >= 4.0:
                ax.text(left + val / 2, y, asset.split(".")[0], ha="center", va="center", fontsize=7.6, color="white", fontweight="semibold")
        lefts += vals
    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Top-weight share (%)")
    ax.set_title(f"{MARKET_LABELS[market]} Actor Weight Distribution at Switch")
    clean_axis(ax)
    ax.grid(True, axis="x", alpha=0.55)
    ax.grid(False, axis="y")
    ax.legend(ncol=min(5, len(assets)), loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    save_figure(fig, out_dir / f"inner_actor_weight_stack_{market}")


def plot_inner_actor(market: str, seed: int, end2end_dir: Path, case_step: int, out_dir: Path) -> Dict[str, float]:
    full = read_portfolio(end2end_dir / "traces" / f"{market}_seed{seed}_full_controller_portfolio.csv")
    outer = read_portfolio(end2end_dir / "traces" / f"{market}_seed{seed}_controller_outer_portfolio.csv")
    full["date"] = parse_dates(full["date"])
    outer["date"] = parse_dates(outer["date"])
    full = numeric_cols(full, ["portfolio_value", "inner_alpha", "base_log_return", "daily_log_return", "turnover", "step"])
    outer = numeric_cols(outer, ["portfolio_value", "step"])
    full["wealth"] = full["portfolio_value"] / 1000.0
    outer["wealth"] = outer["portfolio_value"] / 1000.0
    full["cum_inner_alpha"] = full["inner_alpha"].fillna(0.0).cumsum()
    full["rolling_inner_alpha"] = full["inner_alpha"].rolling(20, min_periods=5).mean() * 100.0
    full["rolling_base_ret"] = full["base_log_return"].rolling(20, min_periods=5).mean() * 100.0
    full["rolling_exec_ret"] = full["daily_log_return"].rolling(20, min_periods=5).mean() * 100.0

    fig = plt.figure(figsize=(12.0, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.7, 1.35, 1.25], hspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    ax0.plot(outer["date"], outer["wealth"], color=CONTROLLER_COLOR, lw=2.2, label="Outer + Controller")
    ax0.plot(full["date"], full["wealth"], color=OURS_COLOR, lw=2.8, label="Ours")
    ax0.set_title(f"{MARKET_LABELS[market]} Inner Actor Contribution")
    ax0.set_ylabel("Wealth multiple")
    ax0.legend(frameon=False)
    clean_axis(ax0)
    ax0.grid(True, axis="both", alpha=0.58)

    ax1.plot(full["date"], full["cum_inner_alpha"] * 100.0, color=INNER_COLOR, lw=2.4, label="Cumulative inner alpha")
    ax1.axhline(0, color="#C9CED8", lw=1.0)
    ax1.set_ylabel("Log alpha (pp)")
    ax1.legend(frameon=False)
    clean_axis(ax1)
    ax1.grid(True, axis="both", alpha=0.58)

    ax2.plot(full["date"], full["rolling_exec_ret"], color=OURS_COLOR, lw=2.0, label="Executed return")
    ax2.plot(full["date"], full["rolling_base_ret"], color=HOLD_COLOR, lw=1.9, label="Base return")
    ax2b = ax2.twinx()
    ax2b.fill_between(full["date"], 0, full["turnover"].rolling(20, min_periods=5).mean(), color="#D9C5F0", alpha=0.45, label="Turnover")
    if pd.notna(case_step):
        case_rows = full[full["step"] == int(case_step)]
        if not case_rows.empty:
            for ax in [ax0, ax1, ax2]:
                ax.axvline(case_rows["date"].iloc[0], color=OURS_COLOR, lw=1.2, alpha=0.35)
    ax2.set_ylabel("Rolling log return (%)")
    ax2b.set_ylabel("Turnover")
    ax2.legend(frameon=False, loc="upper left")
    clean_axis(ax2)
    ax2.grid(True, axis="both", alpha=0.58)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_color("#D9DEE7")
    for ax in [ax0, ax1, ax2]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.08, hspace=0.38)
    save_figure(fig, out_dir / f"inner_actor_alpha_{market}")

    return {
        "market": market,
        "cumulative_inner_alpha": float(full["inner_alpha"].sum()),
        "mean_inner_alpha": float(full["inner_alpha"].mean()),
        "positive_inner_alpha_ratio": float((full["inner_alpha"] > 0).mean()),
        "mean_turnover": float(full["turnover"].mean()),
        "ours_total_return": float(full["wealth"].iloc[-1] - 1.0),
        "outer_controller_total_return": float(outer["wealth"].iloc[-1] - 1.0),
    }


def inner_experiment(end2end_dir: Path, interpretability_dir: Path, markets: Sequence[str], seeds: Dict[str, int], case_df: pd.DataFrame, dirs: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for market in markets:
        market_cases = case_df[case_df["market"] == market]
        case_step = int(market_cases.iloc[0]["key_step"]) if not market_cases.empty else -1
        rows.append(plot_inner_actor(market, seeds[market], end2end_dir, case_step, dirs["inner"]))
        plot_weight_stack(market, case_step, interpretability_dir, dirs["inner"])
    out = pd.DataFrame(rows)
    out.to_csv(dirs["inner"] / "inner_actor_summary.csv", index=False)
    out.to_csv(dirs["tables"] / "inner_actor_summary.csv", index=False)
    for base in [dirs["inner"], dirs["tables"]]:
        write_display_csv(
            out,
            base / "inner_actor_summary_display.csv",
            percent_cols=[
                "cumulative_inner_alpha",
                "mean_inner_alpha",
                "positive_inner_alpha_ratio",
                "mean_turnover",
                "ours_total_return",
                "outer_controller_total_return",
            ],
        )
    return out


def write_readmes(dirs: Dict[str, Path], markets: Sequence[str]) -> None:
    market_text = "、".join(MARKET_LABELS[m] for m in markets)
    write_text(
        dirs["root"] / "README.md",
        f"""
# Paper Experiments Final

本目录汇总 eval-only 论文实验图，覆盖 {market_text} 两个市场。所有图同时保存为 PNG 和 PDF；图中文字使用 DejaVu Sans 并设置 TrueType PDF 字体嵌入，避免 PDF 字体乱码。

目录：
- `EXPERIMENT_REQUIREMENT_AUDIT.md`：逐条复核用户要求与当前输出文件的对应关系。
- `FIGURE_INTERPRETATION.md`：论文图怎么读、说明了什么、哪些结论能说以及哪些结论需要谨慎表述。
- `01_main_experiment/`：Ours vs matched baselines。
- `02_ablation/`：Outer、Inner、Controller 与固定窗口切仓消融。
- `03_controller_interpretability/`：controller switch case、固定持仓窗口对比、反事实收益分布、概率共振。
- `04_inner_actor_interpretability/`：inner actor 收益贡献、换仓/收益共振、switch 时候选权重分布。
- `tables/`：所有实验对应的 CSV 指标。
""",
    )
    write_text(
        dirs["main"] / "README.md",
        """
# Main Experiment

这个实验比较 Ours（HRL + controller）和能够与论文表格对齐的 baseline。`main_equity_*.png` 看累计财富曲线，`main_metrics_*.png` 看总收益、Sharpe、最大回撤和 CR（年化收益率/最大回撤）。

读图时重点看红色 Ours 曲线和柱子：若收益更高且最大回撤更低，说明 controller 参与的完整框架不仅提高收益，也改善了风险控制。SH 的 AlphaStock 因历史 action 被覆盖，进入指标柱状图和表格，但不进入收益曲线图。
""",
    )
    write_text(
        dirs["ablation"] / "README.md",
        """
# Ablation Experiment

这个实验拆开组件：Outer-only、Outer + Inner、Outer + Controller、不同固定窗口切仓，以及 Ours。`ablation_equity_*.png` 展示各变体累计财富，`ablation_metrics_*.png` 展示收益、Sharpe、最大回撤和 CR。

读图时比较三条主线：Outer + Inner 相对 Outer-only 体现 inner actor 的边际贡献；Outer + Controller 相对 Outer-only 体现 controller 的动态切仓贡献；Ours 同时结合 inner 和 controller。固定窗口曲线用于说明 controller 不是简单固定周期切仓。
""",
    )
    write_text(
        dirs["controller"] / "README.md",
        """
# Controller Interpretability

这个目录解释 controller 到底在什么情况下 switch。`controller_case_*.png` 是自动筛选出的关键 free switch：第一行固定切点后 30 个交易日，比较“继续旧基础组合（无 controller）”和“切到新基础组合（controller）”的反事实收益；第二行比较同一冻结窗口下的未来回撤；第三行展示切点前后的 exit probability 与反事实切仓优势。这样可以避免真实路径后续多次切仓污染单个 switch 的比较。

`switch_counterfactual_distribution_*.png` 比较所有实际 free switch 点之后 20 日的 switch/hold 反事实收益分布；`switch_remaining_horizon_counterfactual_distribution_*.png` 进一步比较每个实际 switch 在“切仓前组合原本剩余持仓期”内的 switch/hold 冻结反事实收益分布，避免真实路径后续多次切仓污染比较。`fixed_window_comparison_*.png` 比较 learned controller 与 5/10/20/30/60 日固定持仓窗口；`controller_probability_resonance_*.png` 展示 exit probability 是否和未来切仓优势同向变化。

结论文字可概括为：controller 的 switch 在多个 case 中对应即将恶化的持仓；切仓后的冻结反事实路径通常降低回撤或改善未来收益；从所有 switch 的剩余持仓期分布看，switch 候选可以和继续持有旧组合进行同 horizon 比较；相比多个固定持仓窗口，learned controller 在 TR、Sharpe 和 CR 上更稳定，说明收益改善不是来自某个手工固定周期。
""",
    )
    write_text(
        dirs["inner"] / "README.md",
        """
# Inner Actor Interpretability

这个目录解释 inner actor 的作用。`inner_actor_alpha_*.png` 比较 Ours 与 Outer + Controller，并展示累计 inner alpha、rolling executed/base return 和 turnover。它用于说明 inner actor 不是静态噪声，而是在收益波动和持仓调整之间产生可观察的贡献。

`inner_actor_weight_stack_*.png` 展示关键 switch 时“继续持有候选组合”和“切到新组合候选组合”的 top-weight 分布。读图时看两条堆叠条的资产权重变化：权重迁移对应 controller 决策后的新持仓方向。
""",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dirs = ensure_dirs(output_dir)
    markets = list(args.markets)
    seed_map = parse_seed_specs(args.seeds, markets=markets)
    seeds = {market: seed_for_market(seed_map, market) for market in markets}
    manifest = baseline_manifest(Path(args.baseline_dir))
    end2end_dir = Path(args.end2end_dir)
    fixed_metrics = ensure_fixed_window_eval(args, markets, seeds, dirs)
    counterfactual_bundles = ensure_counterfactual_horizon_eval(args, markets, seeds, dirs)
    main_metrics = main_experiment(manifest, end2end_dir, markets, seeds, dirs)
    ablation_metrics = ablation_experiment(end2end_dir, fixed_metrics, markets, seeds, dirs)
    case_df = controller_experiment(
        end2end_dir,
        markets,
        seeds,
        dirs,
        args.case_count,
        ablation_metrics=ablation_metrics,
        counterfactual_bundles=counterfactual_bundles,
        counterfactual_horizon=max(20, int(args.counterfactual_horizon)),
    )
    inner_summary = inner_experiment(end2end_dir, Path(args.results_root) / "interpretability", markets, seeds, case_df, dirs)
    write_readmes(dirs, markets)
    print(f"wrote final paper experiments to {output_dir}")
    print(f"main metrics rows: {len(main_metrics)}")
    print(f"ablation metrics rows: {len(ablation_metrics)}")
    print(f"controller cases: {len(case_df)}")
    print(f"inner summary rows: {len(inner_summary)}")


if __name__ == "__main__":
    main()
