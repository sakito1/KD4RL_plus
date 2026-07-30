#!/usr/bin/env python3
"""Render the two Appendix C.1 Controller hold/switch case figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APPENDIX_ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "state": "#2B6CB0",
    "base": "#2B6CB0",
    "adv": "#E69F00",
    "final": "#8C4F9E",
    "hold": "#D55E5A",
    "candidate": "#009E73",
}


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if len(cases) != 4:
        raise ValueError("Controller manifest must contain exactly four cases")
    return data


def load_case(appendix_root: Path, case: dict[str, object]) -> dict[str, object]:
    input_path = appendix_root / str(case["input"])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    required = {
        "past_curve",
        "hold_curve_30",
        "candidate_curve_30",
        "base_logit",
        "adv_correction",
        "final_logit",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{input_path} is missing fields: {sorted(missing)}")
    return {**case, **payload}


def _percent_curve(values: list[float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - 1.0) * 100.0


def plot_market(
    market_key: str,
    cases: list[dict[str, object]],
    output_dir: Path,
) -> None:
    if len(cases) != 2:
        raise ValueError(f"{market_key} requires one hold and one switch case")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.2),
        constrained_layout=True,
    )
    panel_letters = ("A", "B", "C", "D", "E", "F")
    for row_index, case in enumerate(cases):
        decision = str(case["decision"])
        winner = "Hold" if decision == "HOLD" else "Candidate"

        ax = axes[row_index, 0]
        past = _percent_curve(case["past_curve"])
        ax.plot(np.arange(len(past)), past, color=COLORS["state"], linewidth=2.3)
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=0.9)
        ax.set_title(
            f"{panel_letters[row_index * 3]}. Pre-decision state\n"
            f"Correct {decision.title()} · {case['date']}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Days in current segment")
        ax.set_ylabel("Cumulative return (%)")
        ax.text(
            0.03,
            0.05,
            f"Held {int(case['hold_duration'])} days\n"
            f"Return {float(case['segment_return']) * 100:+.2f}% · "
            f"DD {float(case['segment_drawdown']) * 100:.2f}%",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )

        ax = axes[row_index, 1]
        base = float(case["base_logit"])
        correction = float(case["adv_correction"])
        final = float(case["final_logit"])
        ax.bar(0, base, width=0.58, color=COLORS["base"], label="Base")
        ax.bar(
            1,
            correction,
            bottom=base,
            width=0.58,
            color=COLORS["adv"],
            label="Adv correction",
        )
        ax.bar(2, final, width=0.58, color=COLORS["final"], label="Final")
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xticks([0, 1, 2], ["Base", "+ Adv", "Final"])
        ax.set_ylabel("Controller logit")
        ax.set_title(
            f"{panel_letters[row_index * 3 + 1]}. Gate decomposition\n"
            f"Decision: {decision} · p = "
            f"{float(case['switch_probability']) * 100:.2f}%",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.text(0, base, f"{base:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(
            1,
            base + correction / 2.0,
            f"{correction:+.3f}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(2, final, f"{final:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="best")

        ax = axes[row_index, 2]
        hold = _percent_curve(case["hold_curve_30"])
        candidate = _percent_curve(case["candidate_curve_30"])
        days = np.arange(len(hold))
        ax.plot(days, hold, color=COLORS["hold"], linewidth=2.2, label="Hold")
        ax.plot(
            days,
            candidate,
            color=COLORS["candidate"],
            linewidth=2.2,
            label="Candidate",
        )
        ax.axhline(0.0, color="#666666", linestyle=":", linewidth=0.9)
        ax.axvline(20, color="#999999", linestyle="--", linewidth=0.8)
        advantage_20 = float(case["winner_advantage_20_pp"])
        advantage_30 = float(case["winner_advantage_30_pp"])
        ax.set_title(
            f"{panel_letters[row_index * 3 + 2]}. Frozen counterfactual\n"
            f"{winner} wins · +{advantage_20:.2f} / +{advantage_30:.2f} pp",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Trading days after decision")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(frameon=False, fontsize=8, loc="best")

    market_label = str(cases[0]["market"])
    figure.suptitle(
        f"{market_label}: Controller decision loop",
        fontsize=13,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"controller_cases_{market_key}"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None},
    )
    plt.close(figure)


def render(manifest_path: Path, output_dir: Path) -> None:
    manifest = load_manifest(manifest_path)
    loaded = [
        load_case(Path(manifest_path).resolve().parents[1], case)
        for case in manifest["cases"]
    ]
    for market_key in ("nas", "sh"):
        selected = [case for case in loaded if case["market_key"] == market_key]
        selected.sort(key=lambda case: 0 if case["decision"] == "HOLD" else 1)
        plot_market(market_key, selected, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=APPENDIX_ROOT / "configs/controller_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs/figures",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    render(args.manifest, args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
