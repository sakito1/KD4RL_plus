#!/usr/bin/env python3
"""Render Appendix B.2 dense fixed holding-window sensitivity figures."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_fixed_window_sensitivity import APPENDIX_ROOT, MARKETS, WINDOWS


CONTROLLER_COLOR = "#B63A4A"
REFERENCE_COLOR = "#222222"
INK = "#273142"
GRID = "#D8DDE6"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _style_axis(axis, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AEB5C0")
    axis.spines["bottom"].set_color("#AEB5C0")
    axis.tick_params(colors=INK, labelsize=8.5)
    axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)


def _market_summary(
    summary_rows: list[dict[str, str]],
    market_key: str,
) -> dict[str, str]:
    selected = [row for row in summary_rows if row["market_key"] == market_key]
    if len(selected) != 1:
        raise ValueError(f"expected one summary row for {market_key}")
    return selected[0]


def plot_market(
    public_key: str,
    tables_dir: Path,
    output_dir: Path,
) -> None:
    spec = MARKETS[public_key]
    summary = _market_summary(
        _read_csv(Path(tables_dir) / "fixed_window_summary.csv"),
        str(spec["market_key"]),
    )
    wealth_rows = _read_csv(
        Path(tables_dir) / f"fixed_window_wealth_{public_key}.csv"
    )
    dates = [datetime.fromisoformat(row["date"]) for row in wealth_rows]

    figure, (wealth_axis, rank_axis) = plt.subplots(
        1,
        2,
        figsize=(10.4, 5.0),
        gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.22},
    )
    purple = plt.cm.Purples
    for window in WINDOWS:
        shade = 0.25 + 0.45 * (window - WINDOWS[0]) / (WINDOWS[-1] - WINDOWS[0])
        wealth_axis.plot(
            dates,
            [float(row[f"fixed_w{window:02d}"]) for row in wealth_rows],
            color=purple(shade),
            alpha=0.30,
            linewidth=0.76,
            zorder=1,
        )
    wealth_axis.plot(
        dates,
        [float(row["fixed_w30"]) for row in wealth_rows],
        color=REFERENCE_COLOR,
        alpha=0.82,
        linewidth=1.55,
        linestyle="--",
        label="Fixed HRL (30d)",
        zorder=2,
    )
    controller = [float(row["controller"]) for row in wealth_rows]
    wealth_axis.plot(
        dates,
        controller,
        color=CONTROLLER_COLOR,
        linewidth=2.85,
        label="Learned Controller",
        zorder=4,
    )
    wealth_axis.annotate(
        "Controller",
        xy=(dates[-1], controller[-1]),
        xytext=(-6, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=CONTROLLER_COLOR,
        fontsize=8.4,
        fontweight="bold",
    )
    metric_specs = [
        ("TR", "total_return"),
        ("SR", "sharpe"),
        ("MDD", "max_drawdown"),
        ("CR", "calmar"),
    ]
    count = int(summary["fixed_window_count"])
    wealth_axis.text(
        0.98,
        0.04,
        "\n".join(
            [
                f"Fixed windows: 1–60d, n={count}",
                " · ".join(
                    f"{label} wins {int(float(summary[f'controller_{key}_wins']))}/{count}"
                    for label, key in metric_specs[:2]
                ),
                " · ".join(
                    f"{label} wins {int(float(summary[f'controller_{key}_wins']))}/{count}"
                    for label, key in metric_specs[2:]
                ),
            ]
        ),
        transform=wealth_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#BFC5CE",
            "alpha": 0.92,
        },
    )
    wealth_axis.set_title(
        "Wealth paths across fixed holding windows",
        fontsize=10.8,
        pad=8,
    )
    wealth_axis.set_xlabel("Date")
    wealth_axis.set_ylabel("Normalized portfolio value")
    wealth_axis.legend(frameon=False, loc="upper left")
    _style_axis(wealth_axis)

    percentages = [
        float(summary[f"controller_{key}_win_pct"])
        for _, key in metric_specs
    ]
    positions = np.arange(len(metric_specs))
    rank_axis.barh(
        positions,
        percentages,
        color=[CONTROLLER_COLOR, CONTROLLER_COLOR, "#2A9D8F", CONTROLLER_COLOR],
        alpha=0.88,
        height=0.58,
    )
    rank_axis.axvline(
        50.0,
        color="#7A8392",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
    )
    rank_axis.set_yticks(
        positions,
        [label for label, _ in metric_specs],
    )
    rank_axis.set_xlim(0.0, 100.0)
    rank_axis.set_xlabel("Controller win rate (%)")
    rank_axis.set_title("Controller percentile", fontsize=10.8, pad=8)
    for index, value in enumerate(percentages):
        rank_axis.text(
            min(value + 2.0, 98.0),
            index,
            f"{value:.0f}%",
            va="center",
            ha="left" if value < 96.0 else "right",
            fontsize=8.7,
            color=INK,
        )
    rank_axis.invert_yaxis()
    _style_axis(rank_axis, grid_axis="x")

    figure.suptitle(
        f"Dense Fixed Holding-Window Timing Baseline · {spec['market']}",
        y=0.985,
        fontsize=13.6,
        fontweight="bold",
        color=INK,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.145,
        top=0.84,
        wspace=0.22,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"fixed_window_sensitivity_{public_key}"
    figure.savefig(
        stem.with_suffix(".png"),
        dpi=240,
        facecolor="white",
    )
    figure.savefig(
        stem.with_suffix(".pdf"),
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def render(tables_dir: Path, output_dir: Path) -> None:
    for public_key in MARKETS:
        plot_market(public_key, tables_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs/tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs/figures",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    render(args.tables_dir, args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
