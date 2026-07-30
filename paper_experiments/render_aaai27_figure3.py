#!/usr/bin/env python3
"""Render the paper-fixed Figure 3 from frozen selected-case CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.run_paper_experiments_final import (
    parse_curve_json,
    plot_combined_controller_case,
)


def build_case_inputs(
    selected_path: Path,
    case_spec: Mapping[str, object],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Create the minimal portfolio/action inputs needed by the Figure 3 plotter."""

    actions = pd.read_csv(selected_path)
    case_id = int(case_spec["case_id"])
    if case_id < 1 or case_id > len(actions):
        raise ValueError(f"case_id {case_id} is unavailable in {selected_path}")
    case = actions.iloc[case_id - 1].copy()
    decision_date = str(case_spec["decision_date"])
    if str(pd.Timestamp(case["date"]).date()) != decision_date:
        raise ValueError(
            f"{selected_path} case {case_id} has date {case['date']}, "
            f"expected {decision_date}"
        )

    hold_curve = parse_curve_json(case.get("hold_curve_30", ""))
    switch_curve = parse_curve_json(case.get("switch_curve_30", ""))
    if not len(hold_curve) or not len(switch_curve):
        hold_curve = parse_curve_json(case.get("hold_curve_20", ""))
        switch_curve = parse_curve_json(case.get("switch_curve_20", ""))
    curve_len = min(len(hold_curve), len(switch_curve))
    if curve_len < 2:
        raise ValueError(f"{selected_path} case {case_id} has no usable curves")

    plotted_dates = pd.date_range(
        start=str(case_spec["plot_start_date"]),
        end=str(case_spec["plot_end_date"]),
        periods=curve_len - 1,
    )
    dates = [pd.Timestamp(decision_date), *plotted_dates]
    key_step = int(case["step"])
    portfolio = pd.DataFrame(
        {
            "step": range(key_step, key_step + curve_len),
            "date": [timestamp.strftime("%Y-%m-%d") for timestamp in dates],
        }
    )
    return case, portfolio, actions


def render_figure3(
    *,
    trace_root: Path,
    case_manifest: Path,
    output_dir: Path,
) -> None:
    specs = json.loads(case_manifest.read_text(encoding="utf-8"))
    if set(specs) != {"nas", "sh"}:
        raise ValueError("Figure 3 manifest must contain exactly nas and sh")
    inputs = {}
    for market in ("sh", "nas"):
        inputs[market] = build_case_inputs(
            trace_root / f"selected_controller_cases_{market}.csv",
            specs[market],
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    sh_case, sh_portfolio, sh_actions = inputs["sh"]
    nas_case, nas_portfolio, nas_actions = inputs["nas"]
    plot_combined_controller_case(
        sh_case_id=int(specs["sh"]["case_id"]),
        sh_case=sh_case,
        sh_portfolio=sh_portfolio,
        sh_actions=sh_actions,
        nas_case_id=int(specs["nas"]["case_id"]),
        nas_case=nas_case,
        nas_portfolio=nas_portfolio,
        nas_actions=nas_actions,
        out_dir=output_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_root", type=Path, required=True)
    parser.add_argument("--case_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    render_figure3(
        trace_root=args.trace_root,
        case_manifest=args.case_manifest,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
