#!/usr/bin/env python3
"""Render compact Appendix B.1, C.2, and C.3 tables from packaged inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


APPENDIX_ROOT = Path(__file__).resolve().parents[1]
MARKET_LABELS = {"nas": "NASDAQ-100", "sh": "CSI-300"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_controller_table(input_dir: Path, output_path: Path) -> list[dict[str, object]]:
    rows = read_rows(input_dir / "adaptive_horizon_decision_value.csv")
    output = []
    for row in rows:
        output.append(
            {
                "market": row["market_label"],
                "decisions": int(row["free_decisions"]),
                "switch_rate_pct": f"{float(row['switch_rate']) * 100:.2f}",
                "return_value_bp_day": f"{float(row['mean_return_decision_value_bp_day']):.3f}",
                "mdd_value_pp": f"{float(row['mean_mdd_decision_value']) * 100:.3f}",
            }
        )
    fields = [
        "market",
        "decisions",
        "switch_rate_pct",
        "return_value_bp_day",
        "mdd_value_pp",
    ]
    write_rows(output_path, fields, output)
    return output


def build_trader_table(input_dir: Path, output_path: Path) -> list[dict[str, object]]:
    configuration = {
        row["market"]: row
        for row in read_rows(input_dir / "configuration_refinement.csv")
    }
    placebo = {
        row["market"]: row for row in read_rows(input_dir / "placebo_analysis.csv")
    }
    output = []
    for market_key in ("nas", "sh"):
        config = configuration[market_key]
        random = placebo[market_key]
        output.append(
            {
                "market": MARKET_LABELS[market_key],
                "mean_active_share_pct": f"{float(config['mean_active_share']) * 100:.3f}",
                "active_share_gt_1pct_pct": f"{float(config['active_share_gt_1pct']) * 100:.1f}",
                "observed_delta_sigma_pp": f"{float(config['mean_delta_ex_ante_vol']) * 100:.4f}",
                "random_delta_sigma_pp": f"{float(random['placebo_mean_delta_exante_vol']) * 100:.4f}",
                "p_value": f"{float(random['negative_risk_permutation_p']):.4f}",
            }
        )
    fields = [
        "market",
        "mean_active_share_pct",
        "active_share_gt_1pct_pct",
        "observed_delta_sigma_pp",
        "random_delta_sigma_pp",
        "p_value",
    ]
    write_rows(output_path, fields, output)
    return output


def _cost_markdown(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "### Table B.1 Transaction-cost sensitivity",
        "",
        "| Market | Cost | TR | SR | MDD | CR | ΔTR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = "—" if abs(float(row["delta_tr_pp"])) < 5e-6 else f"{float(row['delta_tr_pp']):+.2f} pp"
        lines.append(
            f"| {row['market']} | {float(row['cost_pct']):.3f}% "
            f"| {float(row['total_return_pct']):.2f}% "
            f"| {float(row['sharpe']):.3f} "
            f"| {float(row['max_drawdown_pct']):.2f}% "
            f"| {float(row['calmar']):.3f} | {delta} |"
        )
    return lines


def _controller_markdown(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "### Table C.2 Counterfactual validation of Controller decisions",
        "",
        "| Market | Dec. | Switch | Ret. (bp/day) | MDD (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['market']} | {row['decisions']} "
            f"| {float(row['switch_rate_pct']):.2f}% "
            f"| {float(row['return_value_bp_day']):.3f} "
            f"| {float(row['mdd_value_pp']):.3f} |"
        )
    return lines


def _trader_markdown(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "### Table C.3 Statistical validation of Trader refinement",
        "",
        "| Market | AS | AS > 1% | Observed Δσ (pp) | Random Δσ (pp) | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['market']} | {float(row['mean_active_share_pct']):.3f}% "
            f"| {float(row['active_share_gt_1pct_pct']):.1f}% "
            f"| {float(row['observed_delta_sigma_pp']):.4f} "
            f"| {float(row['random_delta_sigma_pp']):.4f} "
            f"| {float(row['p_value']):.4f} |"
        )
    return lines


def _latex_table(
    headers: list[str],
    align: str,
    body: list[list[str]],
    caption: str,
) -> list[str]:
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in body)
    lines.extend(["\\bottomrule", "\\end{tabular}", f"% {caption}", ""])
    return lines


def render_combined(
    cost_path: Path,
    controller_path: Path,
    trader_path: Path,
    markdown_path: Path,
    latex_path: Path,
) -> None:
    cost = read_rows(cost_path)
    controller = read_rows(controller_path)
    trader = read_rows(trader_path)
    markdown = (
        _cost_markdown(cost)
        + [""]
        + _controller_markdown(controller)
        + [""]
        + _trader_markdown(trader)
        + [""]
    )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    cost_body = []
    for row in cost:
        delta = "--" if abs(float(row["delta_tr_pp"])) < 5e-6 else f"{float(row['delta_tr_pp']):+.2f}"
        cost_body.append(
            [
                row["market"],
                f"{float(row['cost_pct']):.3f}\\%",
                f"{float(row['total_return_pct']):.2f}\\%",
                f"{float(row['sharpe']):.3f}",
                f"{float(row['max_drawdown_pct']):.2f}\\%",
                f"{float(row['calmar']):.3f}",
                delta,
            ]
        )
    controller_body = [
        [
            row["market"],
            row["decisions"],
            f"{float(row['switch_rate_pct']):.2f}\\%",
            f"{float(row['return_value_bp_day']):.3f}",
            f"{float(row['mdd_value_pp']):.3f}",
        ]
        for row in controller
    ]
    trader_body = [
        [
            row["market"],
            f"{float(row['mean_active_share_pct']):.3f}\\%",
            f"{float(row['active_share_gt_1pct_pct']):.1f}\\%",
            f"{float(row['observed_delta_sigma_pp']):.4f}",
            f"{float(row['random_delta_sigma_pp']):.4f}",
            f"{float(row['p_value']):.4f}",
        ]
        for row in trader
    ]
    latex = (
        _latex_table(
            ["Market", "Cost", "TR", "SR", "MDD", "CR", "$\\Delta$TR"],
            "lrrrrrr",
            cost_body,
            "Transaction-cost sensitivity.",
        )
        + _latex_table(
            ["Market", "Dec.", "Switch", "Ret. (bp/day)", "MDD (pp)"],
            "lrrrr",
            controller_body,
            "Counterfactual validation of Controller decisions.",
        )
        + _latex_table(
            ["Market", "AS", "AS $>$ 1\\%", "Observed $\\Delta\\sigma$", "Random $\\Delta\\sigma$", "$p$"],
            "lrrrrr",
            trader_body,
            "Statistical validation of Trader refinement.",
        )
    )
    latex_path.write_text("\n".join(latex), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=APPENDIX_ROOT / "outputs/tables/transaction_cost_sensitivity.csv",
    )
    parser.add_argument(
        "--controller-input",
        type=Path,
        default=APPENDIX_ROOT / "inputs/controller_statistics",
    )
    parser.add_argument(
        "--trader-input",
        type=Path,
        default=APPENDIX_ROOT / "inputs/trader_statistics",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APPENDIX_ROOT / "outputs/tables",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    controller = args.output_dir / "controller_decision_validation.csv"
    trader = args.output_dir / "trader_refinement_validation.csv"
    build_controller_table(args.controller_input, controller)
    build_trader_table(args.trader_input, trader)
    render_combined(
        args.cost_table,
        controller,
        trader,
        args.output_dir / "appendix_tables.md",
        args.output_dir / "appendix_tables.tex",
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
