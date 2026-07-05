#!/usr/bin/env python3
"""Plot exploratory inner-actor weight/price resonance cases.

The plot uses daily action weights saved by the original HierAgent checkpoints
and aligns them with adjusted close prices. It is meant as an interpretability
visualization: within a monitored window, do weight moves co-vary with
cross-sectional price moves?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_experiments_outputs" / "paper_experiments_final" / "04_inner_actor_interpretability"

CONFIGS = {
    "nas": {
        "title": "NASDAQ Inner Actor Weight-Price Resonance",
        "actions": ROOT
        / "checkpoints/nas100/HierAgent-HierAgent-one-Hier-国内股市-sharpe框-纠错/test/actions/actions_epi1.csv",
        "prices": ROOT / "DeepAries/data/nas/nas_data.csv",
        "start": "2021-08-11",
        "end": "2021-10-06",
        "assets": ["CCEP.O", "ADSK.O", "COST.O", "BKNG.O", "MNST.O", "CPRT.O"],
    },
    "sh": {
        "title": "China A-Share Inner Actor Weight-Price Resonance",
        "actions": ROOT
        / "checkpoints/CN_wind_复权/HierAgent-HierAgent-one-Hier-国内股市-sharpe框-纠错/test/actions/actions_epi1.csv",
        "prices": ROOT / "DeepAries/data/sh/sh_data.csv",
        "start": "2023-01-05",
        "end": "2023-02-22",
        "assets": ["600875.SH", "000733.SZ", "600183.SH", "600111.SH", "600219.SH", "600150.SH"],
    },
}


PALETTE = ["#B83A4B", "#2A9D8F", "#E9A23B", "#496A81", "#7B5EA7", "#6A994E"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D8DEE9",
            "axes.labelcolor": "#394150",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#E8EDF3",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def load_market(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = pd.read_csv(cfg["actions"], index_col=0)
    actions.index = pd.to_datetime(actions.index)
    actions = actions.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    prices = pd.read_csv(cfg["prices"], usecols=["date", "tic", "adjclose"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.pivot(index="date", columns="tic", values="adjclose").sort_index()

    dates = actions.index.intersection(prices.index)
    assets = [asset for asset in cfg["assets"] if asset in actions.columns and asset in prices.columns]
    actions = actions.loc[dates, assets]
    prices = prices.loc[dates, assets].ffill()
    return actions, prices


def case_frame(cfg: dict) -> dict:
    actions, prices = load_market(cfg)
    start = pd.Timestamp(cfg["start"])
    end = pd.Timestamp(cfg["end"])
    weights = actions.loc[start:end].copy()
    prices = prices.loc[start:end].copy()

    raw_return = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rel_return = raw_return.sub(raw_return.mean(axis=1), axis=0)
    rel_price = (1.0 + rel_return).cumprod() * 100.0
    rel_price = rel_price.div(rel_price.iloc[0]).mul(100.0)

    delta_weight = weights.diff().fillna(0.0)
    turnover = delta_weight.abs().sum(axis=1)
    resonance = (delta_weight * rel_return).sum(axis=1) * 10000.0

    corr_values = []
    level_corr_values = []
    for asset in weights.columns:
        x = delta_weight[asset].to_numpy()
        y = rel_return[asset].to_numpy()
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            corr_values.append(float(np.corrcoef(x, y)[0, 1]))
        level_x = weights[asset].to_numpy()
        level_y = (rel_price[asset] - 100.0).to_numpy()
        if np.std(level_x) > 1e-12 and np.std(level_y) > 1e-12:
            level_corr_values.append(float(np.corrcoef(level_x, level_y)[0, 1]))
    mean_corr = float(np.mean(corr_values)) if corr_values else np.nan
    mean_level_corr = float(np.mean(level_corr_values)) if level_corr_values else np.nan
    positive_resonance_ratio = float((resonance > 0).mean())
    return {
        "weights": weights,
        "rel_price": rel_price,
        "rel_return": rel_return,
        "delta_weight": delta_weight,
        "turnover": turnover,
        "resonance": resonance,
        "mean_corr": mean_corr,
        "mean_level_corr": mean_level_corr,
        "positive_resonance_ratio": positive_resonance_ratio,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _date_ticks(index: pd.Index, max_ticks: int = 6) -> tuple[np.ndarray, list[str]]:
    if len(index) <= max_ticks:
        positions = np.arange(len(index))
    else:
        positions = np.linspace(0, len(index) - 1, max_ticks).round().astype(int)
    labels = [pd.Timestamp(index[i]).strftime("%Y-%m-%d") for i in positions]
    return positions, labels


def plot_heatmap_case(market: str, cfg: dict) -> None:
    data = case_frame(cfg)
    weights = data["weights"]
    rel_price = data["rel_price"]
    turnover = data["turnover"]
    resonance = data["resonance"]
    assets = list(weights.columns)
    xticks, xticklabels = _date_ticks(weights.index)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 7.8),
        gridspec_kw={"height_ratios": [1.05, 1.05, 0.72], "hspace": 0.26},
    )
    fig.suptitle(cfg["title"], fontsize=17, fontweight="bold", y=0.985)

    price_delta = rel_price.T - 100.0
    vmax = max(8.0, float(np.nanpercentile(np.abs(price_delta.to_numpy()), 92)))
    im0 = axes[0].imshow(price_delta, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    axes[0].set_yticks(np.arange(len(assets)))
    axes[0].set_yticklabels(assets, fontsize=9)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([])
    axes[0].set_title("Relative price move within the monitored holding window", fontsize=11, loc="left")
    cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.01, fraction=0.032)
    cbar0.set_label("Rel. price (%)", fontsize=9)

    im1 = axes[1].imshow(weights.T * 100.0, aspect="auto", cmap="YlGnBu")
    axes[1].set_yticks(np.arange(len(assets)))
    axes[1].set_yticklabels(assets, fontsize=9)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([])
    axes[1].set_title("Inner actor portfolio weights on the same assets", fontsize=11, loc="left")
    cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.01, fraction=0.032)
    cbar1.set_label("Weight (%)", fontsize=9)

    bar_colors = np.where(resonance >= 0, "#2A9D8F", "#C23B4B")
    axes[2].bar(np.arange(len(resonance)), resonance, color=bar_colors, width=0.82, alpha=0.78)
    ax2 = axes[2].twinx()
    ax2.plot(np.arange(len(turnover)), turnover * 100.0, color="#4C5B6C", lw=1.8, alpha=0.9)
    axes[2].axhline(0.0, color="#9AA3AF", lw=1.0)
    axes[2].set_ylabel("Resonance\nscore")
    ax2.set_ylabel("Turnover (%)")
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xticklabels, fontsize=9)
    axes[2].grid(True, axis="y", alpha=0.9)
    axes[2].set_title("Daily alignment between weight change and relative return", fontsize=11, loc="left")
    axes[2].text(
        0.012,
        0.92,
        (
            f"Mean corr(weight, relative price): {data['mean_level_corr']:.2f}\n"
            f"Positive resonance days: {data['positive_resonance_ratio']:.0%}"
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#293241",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.95},
    )

    save_figure(fig, OUT_DIR / f"inner_actor_weight_price_resonance_heatmap_{market}")


def plot_case(market: str, cfg: dict) -> dict:
    data = case_frame(cfg)
    weights = data["weights"]
    rel_price = data["rel_price"]
    turnover = data["turnover"]
    resonance = data["resonance"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 8.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.18, 1.0, 0.82], "hspace": 0.16},
    )
    fig.suptitle(cfg["title"], fontsize=17, fontweight="bold", y=0.985)

    for i, asset in enumerate(weights.columns):
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(rel_price.index, rel_price[asset], color=color, lw=2.1, label=asset)
        axes[1].plot(weights.index, weights[asset] * 100.0, color=color, lw=2.0)

    axes[0].axhline(100.0, color="#9AA3AF", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_ylabel("Relative price\n(start=100)")
    axes[0].grid(True, axis="both", alpha=0.9)
    axes[0].legend(ncol=3, loc="upper left", fontsize=9)

    axes[1].set_ylabel("Portfolio\nweight (%)")
    axes[1].grid(True, axis="both", alpha=0.9)

    bar_colors = np.where(resonance >= 0, "#2A9D8F", "#C23B4B")
    axes[2].bar(resonance.index, resonance, color=bar_colors, width=0.85, alpha=0.78, label="Weight-return resonance")
    ax2 = axes[2].twinx()
    ax2.plot(turnover.index, turnover * 100.0, color="#4C5B6C", lw=1.8, alpha=0.9, label="Daily weight turnover")
    axes[2].axhline(0.0, color="#9AA3AF", lw=1.0)
    axes[2].set_ylabel("Resonance\nscore")
    ax2.set_ylabel("Turnover (%)")
    axes[2].grid(True, axis="y", alpha=0.9)

    text = (
        f"Mean corr(weight, relative price): {data['mean_level_corr']:.2f}\n"
        f"Positive resonance days: {data['positive_resonance_ratio']:.0%}"
    )
    axes[2].text(
        0.012,
        0.92,
        text,
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        color="#293241",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.95},
    )

    axes[2].set_xlabel("Date")
    fig.autofmt_xdate(rotation=0)
    save_figure(fig, OUT_DIR / f"inner_actor_weight_price_resonance_{market}")

    row = {
        "market": market,
        "start_date": cfg["start"],
        "end_date": cfg["end"],
        "assets": ";".join(weights.columns),
        "mean_corr_weight_relative_price": data["mean_level_corr"],
        "mean_corr_delta_weight_relative_return": data["mean_corr"],
        "positive_resonance_ratio": data["positive_resonance_ratio"],
        "mean_daily_turnover": float(turnover.mean()),
        "mean_resonance_score": float(resonance.mean()),
    }
    return row


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for market, cfg in CONFIGS.items():
        rows.append(plot_case(market, cfg))
        plot_heatmap_case(market, cfg)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "inner_actor_weight_price_resonance_summary.csv", index=False)
    display = summary.copy()
    for col in ["mean_corr_delta_weight_relative_return", "positive_resonance_ratio", "mean_daily_turnover"]:
        if col == "positive_resonance_ratio" or col == "mean_daily_turnover":
            display[col] = display[col].map(lambda x: f"{x * 100:.2f}%")
        else:
            display[col] = display[col].map(lambda x: f"{x:.2f}")
    display["mean_corr_weight_relative_price"] = display["mean_corr_weight_relative_price"].map(lambda x: f"{x:.2f}")
    display["mean_resonance_score"] = display["mean_resonance_score"].map(lambda x: f"{x:.2f}")
    display.to_csv(OUT_DIR / "inner_actor_weight_price_resonance_summary_display.csv", index=False)


if __name__ == "__main__":
    main()
