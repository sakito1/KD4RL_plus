#!/usr/bin/env python3
"""Recalculate Appendix B.1 from the packaged fixed-path daily replays."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


APPENDIX_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = APPENDIX_ROOT.parent
MARKETS = {
    "nas": ("NASDAQ-100", 49),
    "sh": ("CSI-300", 90),
}
COST_SPECS = (
    (0.005, "tc_0p0050pct"),
    (0.010, "tc_0p0100pct"),
    (0.015, "tc_0p0150pct"),
    (0.020, "tc_0p0200pct"),
    (0.050, "tc_0p0500pct"),
)


def path_metrics(growth: Iterable[float]) -> dict[str, float]:
    """Compute the paper TR/SR/MDD/CR definitions from daily growth."""

    values = [float(value) for value in growth]
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("daily growth must be finite, positive, and non-empty")
    returns = [value - 1.0 for value in values]
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        wealth *= value
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
    daily_std = stdev(returns) if len(returns) > 1 else 0.0
    sharpe = mean(returns) / daily_std * math.sqrt(252.0) if daily_std else 0.0
    annualized_return = mean(returns) * 252.0
    calmar = annualized_return / max_drawdown if max_drawdown else 0.0
    # Python's statistics implementation can differ by one final binary digit
    # across supported interpreters. Fifteen significant digits retain far
    # more precision than the paper reports while keeping public CSVs stable.
    return {
        "total_return": float(format(wealth - 1.0, ".15g")),
        "sharpe": float(format(sharpe, ".15g")),
        "max_drawdown": float(format(max_drawdown, ".15g")),
        "calmar": float(format(calmar, ".15g")),
    }


def load_replay(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty daily replay: {path}")
    return rows


def build_table(input_dir: Path, output_path: Path) -> list[dict[str, object]]:
    """Aggregate both market replays and write the public Appendix B.1 CSV."""

    output_rows: list[dict[str, object]] = []
    for market_key, (market_label, seed) in MARKETS.items():
        replay = load_replay(Path(input_dir) / f"{market_key}_daily_replay.csv")
        market_rows: list[dict[str, object]] = []
        for cost_pct, column_label in COST_SPECS:
            column = f"net_growth_{column_label}"
            if column not in replay[0]:
                raise ValueError(f"{market_key} replay is missing {column}")
            metrics = path_metrics(float(row[column]) for row in replay)
            market_rows.append(
                {
                    "market": market_label,
                    "market_key": market_key,
                    "seed": seed,
                    "cost_pct": cost_pct,
                    "total_return_pct": metrics["total_return"] * 100.0,
                    "sharpe": metrics["sharpe"],
                    "max_drawdown_pct": metrics["max_drawdown"] * 100.0,
                    "calmar": metrics["calmar"],
                }
            )
        reference = float(market_rows[0]["total_return_pct"])
        for row in market_rows:
            row["delta_tr_pp"] = float(row["total_return_pct"]) - reference
        output_rows.extend(market_rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "market",
        "market_key",
        "seed",
        "cost_pct",
        "total_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "calmar",
        "delta_tr_pp",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PACKAGE_ROOT / "traces/transaction_cost/tables",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=APPENDIX_ROOT
        / "outputs/tables/transaction_cost_sensitivity.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_table(args.input_dir, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
