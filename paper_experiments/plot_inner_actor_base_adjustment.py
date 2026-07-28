#!/usr/bin/env python3
"""Plot inner actor base-adjustment versus future relative return.

This is the stricter inner-actor interpretability figure:

    inner tilt = executed_weight - base_weight

A positive tilt means the inner actor overweighted an asset relative to the
base portfolio proposed by the outer/controller stack. We compare the tilt with
the asset's future short-horizon relative return.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_experiments.trace_utils import discover_runs


MARKET_LABELS = {"nas": "Nasdaq-100", "sh": "CSI-300"}
DEFAULT_SEEDS = {"nas": 49, "sh": 90}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot inner tilt versus future relative return.")
    parser.add_argument("--results_root", default=str(ROOT / "results" / "end"))
    parser.add_argument("--output_dir", default=str(ROOT / "paper_experiments_outputs" / "paper_experiments_final"))
    parser.add_argument("--markets", nargs="+", default=["nas", "sh"], choices=["nas", "sh"])
    parser.add_argument("--seeds", nargs="*", default=["nas:49", "sh:90"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--force_eval", action="store_true")
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D8DEE9",
            "axes.labelcolor": "#293241",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#E8EDF3",
            "grid.linewidth": 0.8,
        }
    )


def parse_seed_specs(specs: Sequence[str]) -> Dict[str, int]:
    out = dict(DEFAULT_SEEDS)
    for spec in specs:
        if ":" not in spec:
            continue
        market, seed = spec.split(":", 1)
        out[market] = int(seed)
    return out


def cache_paths(root: Path, market: str, seed: int) -> Dict[str, Path]:
    cache = root / "_cache" / "inner_base_adjustment"
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{market}_seed{seed}_full_controller_inner_base"
    return {
        "portfolio": cache / f"{stem}_portfolio.csv",
        "actions": cache / f"{stem}_actions.csv",
        "switch_events": cache / f"{stem}_switch_events.csv",
    }


def ensure_trace(args: argparse.Namespace, output_root: Path, market: str, seed: int) -> pd.DataFrame:
    paths = cache_paths(output_root, market, seed)
    if not args.force_eval and paths["actions"].exists():
        try:
            cols = pd.read_csv(paths["actions"], nrows=1).columns
            needed = {"asset_names_json", "base_weights_json", "exec_weights_json", "inner_tilt_json"}
            if needed.issubset(set(cols)):
                return pd.read_csv(paths["actions"])
        except Exception:
            pass

    from paper_experiments.eval_end_to_end_explain import (
        build_loaded_trainer,
        collect_eval_trace,
        load_checkpoint_into_trainer,
    )

    seed_map = {market: [int(seed)]}
    runs = discover_runs(Path(args.results_root), markets=[market], seed_map=seed_map)
    if not runs:
        raise RuntimeError(f"No run found for {market}:{seed}")
    run = runs[0]
    trainer, _, torch_module = build_loaded_trainer(
        run,
        output_dir=output_root / "_cache" / "_inner_base_eval_runtime",
        device=args.device,
        results_root=Path(args.results_root),
    )
    ckpt = run.checkpoints["best_model"]
    if not load_checkpoint_into_trainer(trainer, torch_module, ckpt.path):
        raise RuntimeError(f"Could not load checkpoint: {ckpt.path}")
    bundle = collect_eval_trace(
        trainer,
        scenario="full_controller_inner_base",
        fixed_cycle=None,
        disable_inner=False,
    )
    bundle["portfolio"].to_csv(paths["portfolio"], index=False)
    bundle["actions"].to_csv(paths["actions"], index=False)
    bundle["switch_events"].to_csv(paths["switch_events"], index=False)
    return bundle["actions"]


def parse_matrix(actions: pd.DataFrame, column: str) -> pd.DataFrame:
    names = json.loads(actions["asset_names_json"].dropna().iloc[0])
    rows = [json.loads(x) for x in actions[column]]
    out = pd.DataFrame(rows, columns=names)
    out.index = pd.to_datetime(actions["date"])
    return out


def load_prices(market: str, assets: Sequence[str]) -> pd.DataFrame:
    price_path = ROOT / "DeepAries" / "data" / market / f"{market}_data.csv"
    prices = pd.read_csv(price_path, usecols=["date", "tic", "adjclose"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot(index="date", columns="tic", values="adjclose").sort_index()
    return wide.loc[:, [asset for asset in assets if asset in wide.columns]].ffill()


def future_relative_return(prices: pd.DataFrame, dates: pd.Index, horizon: int) -> pd.DataFrame:
    h = max(1, int(horizon))
    fut = np.log(prices.shift(-h) / prices).replace([np.inf, -np.inf], np.nan)
    fut = fut.sub(fut.mean(axis=1), axis=0)
    return fut.reindex(pd.to_datetime(dates))


def mean_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    vals = []
    for col in a.columns:
        x = a[col].to_numpy(dtype="float64")
        y = b[col].to_numpy(dtype="float64")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 3 and np.std(x[mask]) > 1e-12 and np.std(y[mask]) > 1e-12:
            vals.append(float(np.corrcoef(x[mask], y[mask])[0, 1]))
    return float(np.mean(vals)) if vals else np.nan


def select_window(tilt: pd.DataFrame, fut_rel: pd.DataFrame, *, windows=(30, 40, 50, 60)) -> dict:
    candidates = []
    valid_len = min(len(tilt), len(fut_rel))
    tilt = tilt.iloc[:valid_len]
    fut_rel = fut_rel.iloc[:valid_len]
    for win in windows:
        for start in range(0, valid_len - win + 1, 5):
            t = tilt.iloc[start : start + win]
            f = fut_rel.iloc[start : start + win]
            contribution = t * f
            active = (t.abs().mean() * f.abs().mean()).sort_values(ascending=False).head(6).index
            t_sel = t[active]
            f_sel = f[active]
            corr = mean_corr(t_sel, f_sel)
            daily_align = contribution[active].sum(axis=1)
            positive_ratio = float((daily_align > 0).mean())
            mean_abs_tilt = float(t_sel.abs().mean(axis=1).mean())
            mean_abs_fut = float(f_sel.abs().mean(axis=1).mean())
            mean_alignment = float(daily_align.mean())
            score = (
                positive_ratio * 0.62
                + (0.0 if not np.isfinite(corr) else corr) * 0.18
                + max(mean_alignment, 0.0) * 350.0
                + mean_abs_tilt * 4.0
                + mean_abs_fut * 2.0
            )
            candidates.append(
                {
                    "start": start,
                    "end": start + win - 1,
                    "window": win,
                    "assets": list(active),
                    "mean_corr": corr,
                    "positive_ratio": positive_ratio,
                    "mean_alignment": mean_alignment,
                    "mean_abs_tilt": mean_abs_tilt,
                    "mean_abs_future_relative_return": mean_abs_fut,
                    "score": score,
                }
            )
    return max(candidates, key=lambda x: x["score"])


def date_ticks(index: pd.Index, max_ticks: int = 6) -> tuple[np.ndarray, list[str]]:
    if len(index) <= max_ticks:
        locs = np.arange(len(index))
    else:
        locs = np.linspace(0, len(index) - 1, max_ticks).round().astype(int)
    labels = [pd.Timestamp(index[i]).strftime("%Y-%m-%d") for i in locs]
    return locs, labels


def save_figure(fig: plt.Figure, path_base: Path, *, pad_inches: float = 0.1) -> None:
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=240, bbox_inches="tight", pad_inches=pad_inches, facecolor="white")
    plt.close(fig)


def prepare_market_heatmap_data(
    market: str,
    actions: pd.DataFrame,
    *,
    future_horizon: int,
) -> dict:
    tilt = parse_matrix(actions, "inner_tilt_json")
    prices = load_prices(market, tilt.columns)
    fut_rel = future_relative_return(prices, tilt.index, future_horizon)
    common = tilt.index.intersection(fut_rel.dropna(how="all").index)
    tilt = tilt.loc[common]
    fut_rel = fut_rel.loc[common, tilt.columns]
    window = select_window(tilt, fut_rel, windows=(30,))
    sl = slice(window["start"], window["end"] + 1)
    assets = window["assets"]
    idx = tilt.iloc[sl].index
    xticks, xticklabels = date_ticks(idx)
    return {
        "assets": assets,
        "idx": idx,
        "fut_pct": fut_rel.iloc[sl][assets].T * 100.0,
        "tilt_pct": tilt.iloc[sl][assets].T * 100.0,
        "xticks": xticks,
        "xticklabels": xticklabels,
    }


def plot_combined_market_heatmaps(
    market_panels: Dict[str, dict],
    output_dir: Path,
    *,
    future_horizon: int,
) -> None:
    markets = [market for market in ("sh", "nas") if market in market_panels]
    if len(markets) != 2:
        return

    fig = plt.figure(figsize=(13.6, 5.7))
    grid = fig.add_gridspec(
        2,
        2,
        wspace=0.28,
        hspace=0.64,
        left=0.095,
        right=0.97,
        top=0.82,
        bottom=0.26,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    colorbar_axes = []
    for column in range(2):
        position = axes[0, column].get_position()
        colorbar_axes.append(
            fig.add_axes([position.x0, 0.91, position.width, 0.018])
        )
    future_limit = max(
        1.0,
        max(
            float(
                np.nanpercentile(
                    np.abs(
                        market_panels[market]["fut_pct"]
                        .iloc[:, :30]
                        .to_numpy()
                    ),
                    94,
                )
            )
            for market in markets
        ),
    )
    tilt_limit = max(
        0.5,
        max(
            float(
                np.nanpercentile(
                    np.abs(
                        market_panels[market]["tilt_pct"]
                        .iloc[:, :30]
                        .to_numpy()
                    ),
                    94,
                )
            )
            for market in markets
        ),
    )

    future_image = None
    tilt_image = None
    for row, market in enumerate(markets):
        panel = market_panels[market]
        assets = panel["assets"]
        display_days = min(
            30,
            len(panel["idx"]),
            panel["fut_pct"].shape[1],
            panel["tilt_pct"].shape[1],
        )
        idx = panel["idx"][:display_days]
        future_values = panel["fut_pct"].iloc[:, :display_days]
        tilt_values = panel["tilt_pct"].iloc[:, :display_days]
        future_image = axes[row, 0].imshow(
            future_values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-future_limit,
            vmax=future_limit,
        )
        tilt_image = axes[row, 1].imshow(
            tilt_values,
            aspect="auto",
            cmap="BrBG",
            vmin=-tilt_limit,
            vmax=tilt_limit,
        )
        for column in range(2):
            axis = axes[row, column]
            if column == 0:
                axis.set_yticks(np.arange(len(assets)))
                axis.set_yticklabels(assets, fontsize=13)
                axis.tick_params(axis="y", labelsize=13)
            else:
                axis.set_yticks([])
                axis.tick_params(axis="y", left=False, labelleft=False)
            day_count = len(idx)
            day_labels = [
                day
                for day in (1, 5, 10, 15, 20, 25, 30)
                if day <= day_count
            ]
            if day_count and day_labels[-1] != day_count:
                day_labels.append(day_count)
            axis.set_xticks(np.asarray(day_labels) - 1)
            axis.tick_params(axis="x", labelsize=12)
            axis.set_xticklabels(
                [str(day) for day in day_labels],
                fontsize=12,
            )
            start_date = pd.Timestamp(idx[0]).strftime("%Y-%m-%d")
            end_date = pd.Timestamp(idx[-1]).strftime("%Y-%m-%d")
            axis.set_xlabel(
                f"{start_date}—{end_date}",
                fontsize=13,
                labelpad=5,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    future_colorbar = fig.colorbar(
        future_image,
        cax=colorbar_axes[0],
        orientation="horizontal",
    )
    future_colorbar.set_label("Relative return (%)", fontsize=13)
    future_colorbar.ax.tick_params(labelsize=12)
    future_colorbar.ax.xaxis.set_label_position("top")
    tilt_colorbar = fig.colorbar(
        tilt_image,
        cax=colorbar_axes[1],
        orientation="horizontal",
    )
    tilt_colorbar.set_label("Refinement tilt (pp)", fontsize=13)
    tilt_colorbar.ax.tick_params(labelsize=12)
    tilt_colorbar.ax.xaxis.set_label_position("top")

    for column, title in enumerate(
        (
            f"A. Future {future_horizon}-day relative return",
            "B. Refinement tilt",
        )
    ):
        position = axes[1, column].get_position()
        fig.text(
            (position.x0 + position.x1) / 2,
            0.012,
            title,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color="#1F2937",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(
        fig,
        output_dir / "trader_refinement_two_markets",
        pad_inches=0.02,
    )


def plot_market(
    market: str,
    actions: pd.DataFrame,
    output_root: Path,
    *,
    future_horizon: int,
) -> dict:
    base = parse_matrix(actions, "base_weights_json")
    exec_w = parse_matrix(actions, "exec_weights_json")
    tilt = parse_matrix(actions, "inner_tilt_json")
    prices = load_prices(market, tilt.columns)
    fut_rel = future_relative_return(prices, tilt.index, future_horizon)
    common = tilt.index.intersection(fut_rel.dropna(how="all").index)
    tilt = tilt.loc[common]
    base = base.loc[common]
    exec_w = exec_w.loc[common]
    fut_rel = fut_rel.loc[common, tilt.columns]

    window = select_window(tilt, fut_rel)
    sl = slice(window["start"], window["end"] + 1)
    assets = window["assets"]
    idx = tilt.iloc[sl].index
    tilt_w = tilt.iloc[sl][assets]
    fut_w = fut_rel.iloc[sl][assets]
    base_w = base.iloc[sl][assets]
    exec_window = exec_w.iloc[sl][assets]
    daily_align = (tilt_w * fut_w).sum(axis=1) * 10000.0
    asset_align = (tilt_w * fut_w).sum(axis=0) * 10000.0
    asset_hit_rate = ((tilt_w * fut_w) > 0).mean(axis=0)
    corr = mean_corr(tilt_w, fut_w)
    positive_ratio = float((daily_align > 0).mean())

    xticks, xticklabels = date_ticks(idx)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.5, 9.2),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72], "hspace": 0.28},
    )
    fig.suptitle(
        f"{MARKET_LABELS[market]} Inner Actor: Base Adjustment vs Future Return",
        fontsize=16,
        fontweight="bold",
        y=0.988,
    )

    fut_pct = fut_w.T * 100.0
    vmax_f = max(1.0, float(np.nanpercentile(np.abs(fut_pct.to_numpy()), 94)))
    im0 = axes[0].imshow(fut_pct, aspect="auto", cmap="RdYlGn", vmin=-vmax_f, vmax=vmax_f)
    axes[0].set_yticks(np.arange(len(assets)))
    axes[0].set_yticklabels(assets, fontsize=9)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([])
    axes[0].set_title(f"Future {future_horizon}-day relative return (green = relative winner)", fontsize=11, loc="left")
    c0 = fig.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.03)
    c0.set_label("Future rel. return (%)", fontsize=9)

    tilt_pct = tilt_w.T * 100.0
    vmax_t = max(0.5, float(np.nanpercentile(np.abs(tilt_pct.to_numpy()), 94)))
    im1 = axes[1].imshow(tilt_pct, aspect="auto", cmap="BrBG", vmin=-vmax_t, vmax=vmax_t)
    axes[1].set_yticks(np.arange(len(assets)))
    axes[1].set_yticklabels(assets, fontsize=9)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([])
    axes[1].set_title("Inner tilt = executed weight - base weight (teal = overweight, brown = underweight)", fontsize=11, loc="left")
    c1 = fig.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.03)
    c1.set_label("Tilt (pp)", fontsize=9)

    exec_pct = exec_window.T * 100.0
    im2 = axes[2].imshow(exec_pct, aspect="auto", cmap="YlGnBu")
    axes[2].set_yticks(np.arange(len(assets)))
    axes[2].set_yticklabels(assets, fontsize=9)
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels([])
    axes[2].set_title("Executed portfolio weights after inner adjustment (darker = larger exposure)", fontsize=11, loc="left")
    c2 = fig.colorbar(im2, ax=axes[2], pad=0.01, fraction=0.03)
    c2.set_label("Exec. weight (%)", fontsize=9)

    asset_order = asset_align.sort_values()
    bar_colors = np.where(asset_order >= 0, "#2A9D8F", "#C23B4B")
    y_pos = np.arange(len(asset_order))
    axes[3].barh(y_pos, asset_order.to_numpy(), color=bar_colors, alpha=0.82, height=0.64)
    axes[3].axvline(0.0, color="#9AA3AF", lw=1.0)
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels(asset_order.index, fontsize=9)
    axes[3].set_xlabel("Cumulative tilt-return alignment score (bp proxy)")
    axes[3].set_title("Asset-level contribution: positive bars mean tilt direction matches future relative return", fontsize=11, loc="left")
    axes[3].grid(True, axis="x", alpha=0.9)
    max_abs = max(float(np.nanmax(np.abs(asset_order.to_numpy()))), 1.0)
    axes[3].set_xlim(-max_abs * 1.25, max_abs * 1.25)
    for y, asset, value in zip(y_pos, asset_order.index, asset_order.to_numpy()):
        hit = asset_hit_rate.loc[asset]
        ha = "left" if value >= 0 else "right"
        offset = max_abs * 0.035 if value >= 0 else -max_abs * 0.035
        axes[3].text(
            value + offset,
            y,
            f"{value:+.1f} | hit {hit:.0%}",
            va="center",
            ha=ha,
            fontsize=8.5,
            color="#293241",
        )
    axes[3].text(
        0.012,
        0.96,
        f"Mean corr(tilt, future relative return): {corr:.2f}\nPositive alignment days: {positive_ratio:.0%}",
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#293241",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.95},
    )

    out_dir = output_root / "04_inner_actor_interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / f"inner_actor_base_adjustment_future_return_{market}")

    return {
        "market": market,
        "start_date": str(idx[0].date()),
        "end_date": str(idx[-1].date()),
        "future_horizon_days": int(future_horizon),
        "assets": ";".join(assets),
        "mean_corr_tilt_future_relative_return": float(corr),
        "positive_alignment_ratio": float(positive_ratio),
        "mean_abs_inner_tilt": float(tilt_w.abs().mean(axis=1).mean()),
        "mean_abs_future_relative_return": float(fut_w.abs().mean(axis=1).mean()),
    }


def main() -> None:
    args = parse_args()
    setup_style()
    output_root = Path(args.output_dir)
    seed_map = parse_seed_specs(args.seeds)
    rows = []
    actions_by_market = {}
    for market in args.markets:
        actions = ensure_trace(args, output_root, market, seed_map[market])
        actions_by_market[market] = actions
        rows.append(plot_market(market, actions, output_root, future_horizon=args.future_horizon))
    summary = pd.DataFrame(rows)
    out_dir = output_root / "04_inner_actor_interpretability"
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "inner_actor_base_adjustment_future_return_summary.csv", index=False)
    summary.to_csv(tables_dir / "inner_actor_base_adjustment_future_return_summary.csv", index=False)
    display = summary.copy()
    display["mean_corr_tilt_future_relative_return"] = display["mean_corr_tilt_future_relative_return"].map(lambda x: f"{x:.2f}")
    for col in ["positive_alignment_ratio", "mean_abs_inner_tilt", "mean_abs_future_relative_return"]:
        display[col] = display[col].map(lambda x: f"{x * 100:.2f}%")
    display.to_csv(out_dir / "inner_actor_base_adjustment_future_return_summary_display.csv", index=False)
    display.to_csv(tables_dir / "inner_actor_base_adjustment_future_return_summary_display.csv", index=False)
    market_panels = {
        market: prepare_market_heatmap_data(
            market,
            actions,
            future_horizon=args.future_horizon,
        )
        for market, actions in actions_by_market.items()
    }
    plot_combined_market_heatmaps(
        market_panels,
        out_dir,
        future_horizon=args.future_horizon,
    )


if __name__ == "__main__":
    main()
