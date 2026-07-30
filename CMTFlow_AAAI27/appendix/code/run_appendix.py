#!/usr/bin/env python3
"""Generate all directly packaged AAAI appendix tables and case figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_fixed_window_sensitivity import build_fixed_window_outputs
from analyze_transaction_cost import APPENDIX_ROOT, PACKAGE_ROOT, build_table
from plot_fixed_window_sensitivity import render as render_fixed_windows
from plot_controller_cases import render as render_controller_cases
from render_statistical_tables import (
    build_controller_table,
    build_trader_table,
    render_combined,
)


def run(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cost_path = tables_dir / "transaction_cost_sensitivity.csv"
    controller_path = tables_dir / "controller_decision_validation.csv"
    trader_path = tables_dir / "trader_refinement_validation.csv"
    build_table(
        PACKAGE_ROOT / "traces/transaction_cost/tables",
        cost_path,
    )
    build_fixed_window_outputs(
        APPENDIX_ROOT / "inputs/fixed_window",
        PACKAGE_ROOT / "traces/transaction_cost/tables",
        tables_dir,
    )
    build_controller_table(
        APPENDIX_ROOT / "inputs/controller_statistics",
        controller_path,
    )
    build_trader_table(
        APPENDIX_ROOT / "inputs/trader_statistics",
        trader_path,
    )
    render_combined(
        cost_path,
        controller_path,
        trader_path,
        tables_dir / "appendix_tables.md",
        tables_dir / "appendix_tables.tex",
    )
    render_controller_cases(
        APPENDIX_ROOT / "configs/controller_cases.json",
        figures_dir,
    )
    render_fixed_windows(tables_dir, figures_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args.output_dir)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
