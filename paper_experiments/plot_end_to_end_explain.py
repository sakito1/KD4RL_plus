import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "full_controller": "#b53a2f",
    "fixed_hrl": "#303030",
    "fixed_hrl_no_inner": "#2f6fbb",
    "Controller-PG checkpoint": "#d88724",
    "Final E2E checkpoint": "#b53a2f",
    "Fixed HRL checkpoint": "#303030",
    "random": "#b8b8b8",
}


def _save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_trace(input_dir: Path, market: str, seed: int, scenario: str):
    path = input_dir / "traces" / f"{market}_seed{seed}_{scenario}_portfolio.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _plot_curves(input_dir: Path, output_dir: Path, market: str, seed: int, scenarios, stem: str, title: str, drawdown=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, scenario in scenarios:
        df = _load_trace(input_dir, market, seed, scenario)
        if df.empty:
            continue
        x = pd.to_datetime(df["date"])
        if drawdown:
            y = pd.to_numeric(df["drawdown"], errors="coerce") * 100.0
        else:
            values = pd.to_numeric(df["portfolio_value"], errors="coerce")
            y = values / max(float(values.iloc[0]), 1e-12)
        ax.plot(x, y, label=label, linewidth=2.0, color=COLORS.get(scenario, COLORS.get(label, None)))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)" if drawdown else "Normalized portfolio value", fontsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, output_dir, stem)


def _plot_bar(df: pd.DataFrame, output_dir: Path, market: str, seed: int, stem: str, title: str):
    if df.empty:
        return
    metrics = ["total_return", "sharpe", "max_drawdown", "switch_count"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.6))
    methods = df["scenario"].astype(str).tolist()
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(df.get(metric), errors="coerce")
        scale = 100.0 if metric in {"total_return", "max_drawdown"} else 1.0
        bars = ax.bar(np.arange(len(vals)), vals * scale, color=["#2f6fbb", "#303030", "#b53a2f"][: len(vals)])
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        for bar, val in zip(bars, vals * scale):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    _save(fig, output_dir, stem)


def _plot_stage_bar(group: pd.DataFrame, output_dir: Path, market: str, seed: int):
    stage = group[group.get("stage").notna()].copy() if "stage" in group else pd.DataFrame()
    if stage.empty:
        return
    stage["method"] = stage["stage"].astype(str)
    metrics = ["total_return", "sharpe", "max_drawdown", "switch_count"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.7))
    for ax, metric in zip(axes, metrics):
        vals = pd.to_numeric(stage.get(metric), errors="coerce")
        scale = 100.0 if metric in {"total_return", "max_drawdown"} else 1.0
        colors = [COLORS.get(x, "#808080") for x in stage["stage"]]
        bars = ax.bar(np.arange(len(vals)), vals * scale, color=colors)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(stage["method"], rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        for bar, val in zip(bars, vals * scale):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Stage progression metrics ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
    _save(fig, output_dir, f"fig02b_stage_progression_bar_{market}_seed{seed}")


def plot_inner_alpha(input_dir: Path, output_dir: Path, market: str, seed: int):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for scenario in ["fixed_hrl", "full_controller"]:
        df = _load_trace(input_dir, market, seed, scenario)
        if df.empty or "inner_alpha" not in df:
            continue
        ax.plot(pd.to_datetime(df["date"]), pd.to_numeric(df["inner_alpha"], errors="coerce").fillna(0).cumsum(), label=scenario, linewidth=2.0, color=COLORS.get(scenario))
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.set_title(f"Cumulative inner alpha ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative log alpha", fontsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, output_dir, f"fig04_cumulative_inner_alpha_{market}_seed{seed}")

    df = _load_trace(input_dir, market, seed, "full_controller")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        alpha = pd.to_numeric(df.get("inner_alpha"), errors="coerce").dropna()
        ax.hist(alpha, bins=40, color="#2f6fbb", alpha=0.72)
        ax.axvline(0, color="#202020", linewidth=1.0)
        ax.axvline(alpha.mean(), color="#b53a2f", linewidth=1.6, label=f"mean={alpha.mean():.4f}")
        ax.set_title(f"Daily inner alpha distribution ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily inner alpha", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(frameon=False)
        _save(fig, output_dir, f"fig04b_inner_alpha_distribution_{market}_seed{seed}")


def plot_alignment(input_dir: Path, output_dir: Path, market: str, seed: int):
    path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_actions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    free = df[df.get("decision_type") == "free_decision"].copy()
    if free.empty:
        return
    free["exit_prob"] = pd.to_numeric(free["exit_prob"], errors="coerce")
    free["controller_switch_advantage"] = pd.to_numeric(free["controller_switch_advantage"], errors="coerce")
    free["bin"] = pd.cut(free["exit_prob"], bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = free.groupby("bin", observed=False)["controller_switch_advantage"].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(np.arange(len(grouped)), grouped["mean"], color="#b53a2f", alpha=0.75)
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels([str(x) for x in grouped["bin"]], rotation=35, ha="right", fontsize=8)
    ax.set_title(f"Exit probability calibration ({market.upper()} seed {seed}, n={len(free)})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Exit probability bin", fontsize=12)
    ax.set_ylabel("Average switch advantage", fontsize=12)
    _save(fig, output_dir, f"fig05_exit_prob_calibration_{market}_seed{seed}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    data = [
        free.loc[free["is_switch"] == 0, "controller_switch_advantage"].dropna(),
        free.loc[free["is_switch"] == 1, "controller_switch_advantage"].dropna(),
    ]
    ax.boxplot(data, labels=["Held", "Switched"], showfliers=False)
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.set_title(f"Switch advantage: held vs switched ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Switch advantage", fontsize=12)
    _save(fig, output_dir, f"fig05b_switch_advantage_switched_vs_held_{market}_seed{seed}")


def plot_switch_events(input_dir: Path, output_dir: Path, market: str, seed: int):
    path = input_dir / "traces" / f"{market}_seed{seed}_full_controller_switch_events.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    metrics = [
        ("pre_return_20", "Pre-switch realized"),
        ("post_actual_return_20", "Actual post-switch"),
        ("post_hold_return_20", "Continue-hold counterfactual"),
        ("post_switch_return_20", "Switch counterfactual"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    means = [pd.to_numeric(df[col], errors="coerce").mean() * 100.0 for col, _ in metrics]
    bars = ax.bar(np.arange(len(metrics)), means, color=["#1b4d5c", "#b53a2f", "#7b8187", "#d88724"])
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean 20-day log return (%)", fontsize=12)
    ax.set_title(f"Switch event study summary ({market.upper()} seed {seed}, n={len(df)})", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, means):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    _save(fig, output_dir, f"fig06_switch_event_study_{market}_seed{seed}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    avoided = pd.to_numeric(df.get("avoided_loss_20"), errors="coerce").dropna()
    ax.hist(avoided * 100.0, bins=min(30, max(5, len(avoided))), color="#b53a2f", alpha=0.72)
    ax.axvline(0, color="#202020", linewidth=0.9)
    if len(avoided):
        ax.axvline(avoided.mean() * 100.0, color="#1b4d5c", linewidth=1.6, label=f"mean={avoided.mean()*100:.2f}%")
    ax.set_title(f"Avoided-loss distribution ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Switch advantage / avoided loss (%)", fontsize=12)
    ax.set_ylabel("Event count", fontsize=12)
    ax.legend(frameon=False)
    _save(fig, output_dir, f"fig06b_switch_avoided_loss_distribution_{market}_seed{seed}")


def plot_random(input_dir: Path, output_dir: Path, market: str, seed: int):
    full = _load_trace(input_dir, market, seed, "full_controller")
    fixed = _load_trace(input_dir, market, seed, "fixed_hrl")
    random_paths = sorted((input_dir / "traces").glob(f"{market}_seed{seed}_random_switch_matched_count_*_portfolio.csv"))
    if full.empty or not random_paths:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for path in random_paths:
        df = pd.read_csv(path)
        values = pd.to_numeric(df["portfolio_value"], errors="coerce")
        ax.plot(pd.to_datetime(df["date"]), values / max(values.iloc[0], 1e-12), color="#b8b8b8", alpha=0.35, linewidth=0.8)
    for df, label, color, lw, ls in [(fixed, "Fixed HRL", "#303030", 1.6, "--"), (full, "Full controller", "#b53a2f", 2.4, "-")]:
        if df.empty:
            continue
        values = pd.to_numeric(df["portfolio_value"], errors="coerce")
        ax.plot(pd.to_datetime(df["date"]), values / max(values.iloc[0], 1e-12), label=label, color=color, linewidth=lw, linestyle=ls)
    ax.set_title(f"Random switch matched-count comparison ({market.upper()} seed {seed})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized portfolio value", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    _save(fig, output_dir, f"fig07_random_switch_comparison_{market}_seed{seed}")


def main_from_paths(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metrics = input_dir / "metrics" / "all_metrics.csv"
    if not metrics.exists():
        return
    all_metrics = pd.read_csv(metrics)
    for (market, seed), group in all_metrics.groupby(["market", "seed"]):
        market = str(market)
        seed = int(seed)
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [
                ("Fixed HRL checkpoint", "stage_hrl_fixed_best_fixed_hrl"),
                ("Controller-PG checkpoint", "stage_controller_best_full_controller"),
                ("Final E2E checkpoint", "stage_best_model_full_controller"),
            ],
            f"fig01_stage_progression_cumulative_{market}_seed{seed}",
            f"Stage progression cumulative return ({market.upper()} seed {seed})",
            drawdown=False,
        )
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [
                ("Fixed HRL checkpoint", "stage_hrl_fixed_best_fixed_hrl"),
                ("Controller-PG checkpoint", "stage_controller_best_full_controller"),
                ("Final E2E checkpoint", "stage_best_model_full_controller"),
            ],
            f"fig02_stage_progression_drawdown_{market}_seed{seed}",
            f"Stage progression drawdown ({market.upper()} seed {seed})",
            drawdown=True,
        )
        _plot_stage_bar(group, output_dir, market, seed)
        ablation = group[group["scenario"].isin(["fixed_hrl_no_inner", "fixed_hrl", "full_controller"])]
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [("Fixed HRL w/o Inner", "fixed_hrl_no_inner"), ("Fixed HRL", "fixed_hrl"), ("Full controller", "full_controller")],
            f"fig03_inference_ablation_cumulative_{market}_seed{seed}",
            f"Inference ablation cumulative return ({market.upper()} seed {seed})",
            drawdown=False,
        )
        _plot_curves(
            input_dir,
            output_dir,
            market,
            seed,
            [("Fixed HRL w/o Inner", "fixed_hrl_no_inner"), ("Fixed HRL", "fixed_hrl"), ("Full controller", "full_controller")],
            f"fig03b_inference_ablation_drawdown_{market}_seed{seed}",
            f"Inference ablation drawdown ({market.upper()} seed {seed})",
            drawdown=True,
        )
        _plot_bar(ablation, output_dir, market, seed, f"fig03c_inference_ablation_bar_{market}_seed{seed}", f"Inference ablation metrics ({market.upper()} seed {seed})")
        plot_inner_alpha(input_dir, output_dir, market, seed)
        plot_alignment(input_dir, output_dir, market, seed)
        plot_switch_events(input_dir, output_dir, market, seed)
        plot_random(input_dir, output_dir, market, seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate paper explanation figures.")
    parser.add_argument("--input_dir", default="paper_experiments_outputs/end_to_end_explain")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    main_from_paths(input_dir, output_dir)


if __name__ == "__main__":
    main()
