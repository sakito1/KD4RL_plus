#!/usr/bin/env python3
"""Reprice and summarize the dense 1--60 day fixed-window evaluation paths."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Iterable

from analyze_transaction_cost import APPENDIX_ROOT, PACKAGE_ROOT, path_metrics


ORIGINAL_FEE = 0.00005
EVALUATION_FEE = 0.00010
WINDOWS = tuple(range(1, 61))
MARKETS = {
    "nasdaq100": {
        "market": "NASDAQ-100",
        "market_key": "nas",
        "controller_file": "nas_daily_replay.csv",
    },
    "csi300": {
        "market": "CSI-300",
        "market_key": "sh",
        "controller_file": "sh_daily_replay.csv",
    },
}
METRICS = (
    ("total_return_pct", "max"),
    ("sharpe", "max"),
    ("max_drawdown_pct", "min"),
    ("calmar", "max"),
)


def reprice_growth(
    original_growth: float,
    turnover: float,
    original_fee: float = ORIGINAL_FEE,
    evaluation_fee: float = EVALUATION_FEE,
) -> float:
    """Remove the recorded proportional fee and apply the evaluation fee."""

    original_growth = float(original_growth)
    turnover = float(turnover)
    original_fee = float(original_fee)
    evaluation_fee = float(evaluation_fee)
    if not math.isfinite(original_growth) or original_growth <= 0.0:
        raise ValueError("growth must be finite and positive")
    if not math.isfinite(turnover) or turnover < 0.0:
        raise ValueError("turnover must be finite and non-negative")
    if original_fee < 0.0 or evaluation_fee < 0.0:
        raise ValueError("fees must be non-negative")
    charged = turnover * original_fee
    replacement = turnover * evaluation_fee
    if charged >= 1.0 or replacement >= 1.0:
        raise ValueError("turnover-adjusted fee must be below one")
    return original_growth / (1.0 - charged) * (1.0 - replacement)


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return fieldnames, rows


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cumulative_wealth(growth: Iterable[float]) -> list[float]:
    wealth = 1.0
    values = []
    for value in growth:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("daily growth must be finite and positive")
        wealth *= value
        values.append(wealth)
    return values


def _expected_replay_columns() -> list[str]:
    return [
        "date",
        *[
            column
            for window in WINDOWS
            for column in (
                f"net_growth_w{window:02d}",
                f"turnover_w{window:02d}",
            )
        ],
    ]


def _load_audit_metrics(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    _, rows = _load_csv(path)
    indexed = {
        (row["market_key"], int(row["fixed_window_days"])): row
        for row in rows
    }
    expected = {
        (spec["market_key"], window)
        for spec in MARKETS.values()
        for window in WINDOWS
    }
    if set(indexed) != expected:
        raise ValueError("fixed-window audit must contain exactly 1--60 for both markets")
    return indexed


def _check_original_metrics(
    market_key: str,
    window: int,
    growth: list[float],
    audit: dict[tuple[str, int], dict[str, str]],
) -> None:
    observed = path_metrics(growth)
    expected = audit[(market_key, window)]
    pairs = {
        "total_return": float(expected["total_return_pct"]) / 100.0,
        "sharpe": float(expected["sharpe"]),
        "max_drawdown": float(expected["max_drawdown_pct"]) / 100.0,
        "calmar": float(expected["calmar"]),
    }
    for metric, expected_value in pairs.items():
        if not math.isclose(
            observed[metric],
            expected_value,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                f"{market_key} window {window} does not reproduce "
                f"the 0.005% {metric}"
            )


def _load_controller(
    path: Path,
    expected_dates: list[str],
) -> tuple[dict[str, float], list[float]]:
    fieldnames, rows = _load_csv(path)
    growth_column = "net_growth_tc_0p0100pct"
    if "date" not in fieldnames or growth_column not in fieldnames:
        raise ValueError(f"{path} is missing the 0.01% Controller replay")
    dates = [row["date"] for row in rows]
    if dates != expected_dates:
        raise ValueError(f"{path} date grid differs from fixed-window replay")
    growth = [float(row[growth_column]) for row in rows]
    metrics = path_metrics(growth)
    return (
        {
            "total_return_pct": metrics["total_return"] * 100.0,
            "sharpe": metrics["sharpe"],
            "max_drawdown_pct": metrics["max_drawdown"] * 100.0,
            "calmar": metrics["calmar"],
        },
        _cumulative_wealth(growth),
    )


def _summary_row(
    market: str,
    market_key: str,
    controller: dict[str, float],
    fixed_rows: list[dict[str, object]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "market": market,
        "market_key": market_key,
        "evaluation_cost_pct": EVALUATION_FEE * 100.0,
        "fixed_window_count": len(fixed_rows),
    }
    for metric, direction in METRICS:
        values = [float(item[metric]) for item in fixed_rows]
        if direction == "min":
            best_index = min(range(len(values)), key=values.__getitem__)
            wins = sum(value > controller[metric] for value in values)
        else:
            best_index = max(range(len(values)), key=values.__getitem__)
            wins = sum(value < controller[metric] for value in values)
        prefix = metric.removesuffix("_pct")
        row[f"controller_{prefix}"] = controller[metric]
        row[f"best_fixed_{prefix}"] = values[best_index]
        row[f"best_fixed_{prefix}_window"] = int(
            fixed_rows[best_index]["fixed_window_days"]
        )
        row[f"median_fixed_{prefix}"] = median(values)
        row[f"controller_{prefix}_wins"] = wins
        row[f"controller_{prefix}_win_pct"] = wins / len(values) * 100.0
    return row


def build_fixed_window_outputs(
    input_dir: Path,
    controller_replay_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    """Build the public 0.01% B.2 metrics, summary, and wealth matrices."""

    input_dir = Path(input_dir)
    controller_replay_dir = Path(controller_replay_dir)
    output_dir = Path(output_dir)
    audit = _load_audit_metrics(input_dir / "fixed_window_metrics.csv")
    result_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    wealth_paths: dict[str, Path] = {}

    for public_key, spec in MARKETS.items():
        replay_path = input_dir / f"daily_replay_{public_key}.csv"
        fieldnames, replay = _load_csv(replay_path)
        if fieldnames != _expected_replay_columns():
            raise ValueError(f"{replay_path} has unexpected disclosure columns")
        dates = [row["date"] for row in replay]
        if len(set(dates)) != len(dates):
            raise ValueError(f"{replay_path} contains duplicate dates")

        controller_metrics, controller_wealth = _load_controller(
            controller_replay_dir / str(spec["controller_file"]),
            dates,
        )
        wealth_columns: dict[str, list[float]] = {
            "controller": controller_wealth,
        }
        market_rows: list[dict[str, object]] = []
        for window in WINDOWS:
            growth_column = f"net_growth_w{window:02d}"
            turnover_column = f"turnover_w{window:02d}"
            original_growth = [float(row[growth_column]) for row in replay]
            turnover = [float(row[turnover_column]) for row in replay]
            _check_original_metrics(
                str(spec["market_key"]),
                window,
                original_growth,
                audit,
            )
            repriced = [
                reprice_growth(growth, daily_turnover)
                for growth, daily_turnover in zip(
                    original_growth,
                    turnover,
                    strict=True,
                )
            ]
            metrics = path_metrics(repriced)
            result = {
                "market": spec["market"],
                "market_key": spec["market_key"],
                "fixed_window_days": window,
                "evaluation_cost_pct": EVALUATION_FEE * 100.0,
                "total_return_pct": metrics["total_return"] * 100.0,
                "sharpe": metrics["sharpe"],
                "max_drawdown_pct": metrics["max_drawdown"] * 100.0,
                "calmar": metrics["calmar"],
            }
            result_rows.append(result)
            market_rows.append(result)
            wealth_columns[f"fixed_w{window:02d}"] = _cumulative_wealth(repriced)

        summary_rows.append(
            _summary_row(
                str(spec["market"]),
                str(spec["market_key"]),
                controller_metrics,
                market_rows,
            )
        )
        wealth_rows = []
        for index, date in enumerate(dates):
            wealth_rows.append(
                {
                    "date": date,
                    **{
                        column: values[index]
                        for column, values in wealth_columns.items()
                    },
                }
            )
        wealth_path = output_dir / f"fixed_window_wealth_{public_key}.csv"
        wealth_paths[public_key] = _write_csv(
            wealth_path,
            ["date", "controller", *[f"fixed_w{w:02d}" for w in WINDOWS]],
            wealth_rows,
        )

    metrics_path = _write_csv(
        output_dir / "fixed_window_sensitivity.csv",
        [
            "market",
            "market_key",
            "fixed_window_days",
            "evaluation_cost_pct",
            "total_return_pct",
            "sharpe",
            "max_drawdown_pct",
            "calmar",
        ],
        result_rows,
    )
    summary_path = _write_csv(
        output_dir / "fixed_window_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    return metrics_path, summary_path, wealth_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=APPENDIX_ROOT / "inputs/fixed_window",
    )
    parser.add_argument(
        "--controller-replay-dir",
        type=Path,
        default=PACKAGE_ROOT / "traces/transaction_cost/tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs/tables",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_fixed_window_outputs(
        args.input_dir,
        args.controller_replay_dir,
        args.output_dir,
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
