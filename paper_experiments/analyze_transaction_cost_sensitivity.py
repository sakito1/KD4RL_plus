#!/usr/bin/env python3
"""Replay a recorded fixed path at alternative transaction-cost rates.

This module changes only the mechanical cost multiplier. It does not rerun the
policy, change Controller decisions, or retrain any network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_COST_RATES = (0.00005, 0.00010, 0.00015, 0.00020, 0.00050)


def cost_label(rate: float) -> str:
    """Return a stable column suffix for a decimal transaction-cost rate."""

    return f"tc_{float(rate) * 100:.4f}pct".replace(".", "p")


def _validate_weight_and_price_frames(
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    cost_rates: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    if executed.empty:
        raise ValueError("executed weight path must not be empty")
    if list(executed.columns) != list(prices.columns):
        raise ValueError("executed weights and prices must use the same ordered assets")
    rates = [float(rate) for rate in cost_rates]
    if not rates:
        raise ValueError("at least one transaction-cost rate is required")
    if any(rate < 0.0 for rate in rates):
        raise ValueError("transaction-cost rates must be non-negative")

    weights = executed.copy()
    market_prices = prices.copy()
    weights.index = pd.to_datetime(weights.index)
    market_prices.index = pd.to_datetime(market_prices.index)
    if weights.index.has_duplicates or market_prices.index.has_duplicates:
        raise ValueError("executed weights and prices require unique dates")
    weights = weights.sort_index().astype("float64")
    market_prices = market_prices.sort_index().astype("float64")
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("executed weights contain non-finite values")
    if not np.isfinite(market_prices.to_numpy()).all():
        raise ValueError("prices contain non-finite values")
    if (market_prices <= 0.0).any().any():
        raise ValueError("prices must be strictly positive")
    if (weights < -1e-10).any().any():
        raise ValueError("executed weights must be long-only")

    row_sums = weights.sum(axis=1)
    if (row_sums <= 0.0).any():
        raise ValueError("each executed-weight row must have positive mass")
    weights = weights.div(row_sums, axis=0)
    return weights, market_prices, rates


def replay_executed_weight_path(
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    cost_rates: Sequence[float],
) -> pd.DataFrame:
    """Replay common weights and prices while varying only charged cost.

    The first executed row supplies the previous target for the first replay
    date. The final executed row is excluded because no next-day price exists
    inside the common executed path.
    """

    weights, market_prices, rates = _validate_weight_and_price_frames(
        executed,
        prices,
        cost_rates,
    )
    missing_dates = weights.index.difference(market_prices.index)
    if len(missing_dates):
        raise ValueError(f"prices are missing {len(missing_dates)} executed dates")
    if len(weights) < 2:
        raise ValueError("at least two executed rows are required")

    aligned_prices = market_prices.loc[weights.index, weights.columns]
    previous_weights = weights.shift(1)
    previous_to_current = aligned_prices / aligned_prices.shift(1)
    drifted = previous_weights * previous_to_current
    drifted = drifted.div(drifted.sum(axis=1), axis=0)
    turnover = (weights - drifted).abs().sum(axis=1, min_count=1)

    price_positions = market_prices.index.get_indexer(weights.index)
    if (price_positions < 0).any():
        raise ValueError("could not locate every executed date in the price panel")
    has_next_price = price_positions + 1 < len(market_prices)
    gross_growth = pd.Series(np.nan, index=weights.index, dtype="float64")
    if has_next_price.any():
        valid_dates = weights.index[has_next_price]
        current_prices = market_prices.iloc[
            price_positions[has_next_price]
        ].set_axis(valid_dates)
        next_prices = market_prices.iloc[
            price_positions[has_next_price] + 1
        ].set_axis(valid_dates)
        current_to_next = next_prices / current_prices
        gross_growth.loc[valid_dates] = (
            weights.loc[valid_dates] * current_to_next
        ).sum(axis=1)
    valid = turnover.notna() & gross_growth.notna()
    replay = pd.DataFrame(
        {
            "turnover": turnover.loc[valid],
            "gross_growth": gross_growth.loc[valid],
        }
    )
    replay["gross_return"] = replay["gross_growth"] - 1.0

    for rate in rates:
        label = cost_label(rate)
        charged = replay["turnover"] * rate
        net_growth = (1.0 - charged) * replay["gross_growth"]
        if (net_growth <= 0.0).any() or not np.isfinite(net_growth).all():
            raise ValueError(f"non-positive or non-finite net growth at rate {rate}")
        replay[f"charged_cost_rate_{label}"] = charged
        replay[f"net_growth_{label}"] = net_growth
        replay[f"net_log_return_{label}"] = np.log(net_growth)
        replay[f"wealth_{label}"] = net_growth.cumprod()

    replay.index.name = "date"
    return replay


def replay_recorded_trace(
    recorded: pd.DataFrame,
    *,
    cost_rates: Sequence[float],
    reference_rate: float,
) -> pd.DataFrame:
    """Reprice an environment-recorded path while holding its actions fixed.

    ``exec_log_return`` is the net path actually emitted by the evaluation
    environment at ``reference_rate``. Recovering the corresponding pre-cost
    growth from that field preserves the exact first day and the environment's
    return timing, which cannot be inferred reliably from end-of-day weights.
    """

    required = {"exec_log_return", "turnover", "cost_rate"}
    missing = sorted(required.difference(recorded.columns))
    if missing:
        raise ValueError(f"recorded trace is missing columns: {missing}")
    if recorded.empty:
        raise ValueError("recorded trace must not be empty")

    trace = recorded.loc[:, sorted(required)].copy()
    trace.index = pd.to_datetime(trace.index)
    trace = trace.sort_index().astype("float64")
    if trace.index.has_duplicates:
        raise ValueError("recorded trace dates must be unique")
    if not np.isfinite(trace.to_numpy()).all():
        raise ValueError("recorded trace contains non-finite values")
    if (trace["turnover"] < 0.0).any():
        raise ValueError("recorded turnover must be non-negative")

    rates = [float(rate) for rate in cost_rates]
    reference_rate = float(reference_rate)
    if not rates or reference_rate not in rates:
        raise ValueError("reference_rate must be included in cost_rates")
    if any(not np.isfinite(rate) or rate < 0.0 for rate in rates):
        raise ValueError("transaction-cost rates must be finite and non-negative")

    expected_reference_cost = reference_rate * trace["turnover"]
    if not np.allclose(
        trace["cost_rate"],
        expected_reference_cost,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(
            "recorded cost_rate does not match reference_rate * turnover"
        )
    reference_multiplier = 1.0 - expected_reference_cost
    if (reference_multiplier <= 0.0).any():
        raise ValueError("reference transaction cost consumes all portfolio value")

    replay = pd.DataFrame(index=trace.index.copy())
    replay.index.name = "date"
    replay["turnover"] = trace["turnover"]
    replay["recorded_cost_rate"] = trace["cost_rate"]
    replay["gross_growth"] = (
        np.exp(trace["exec_log_return"]) / reference_multiplier
    )
    replay["gross_return"] = replay["gross_growth"] - 1.0
    for rate in rates:
        label = cost_label(rate)
        charged = replay["turnover"] * rate
        net_growth = replay["gross_growth"] * (1.0 - charged)
        if (net_growth <= 0.0).any() or not np.isfinite(net_growth).all():
            raise ValueError(f"non-positive or non-finite net growth at rate {rate}")
        replay[f"charged_cost_rate_{label}"] = charged
        replay[f"net_growth_{label}"] = net_growth
        replay[f"net_log_return_{label}"] = np.log(net_growth)
        replay[f"wealth_{label}"] = net_growth.cumprod()
    return replay


def _path_metrics(net_growth: pd.Series) -> Mapping[str, float]:
    returns = net_growth.astype("float64") - 1.0
    wealth = net_growth.cumprod()
    wealth_with_initial = pd.concat(
        [pd.Series([1.0], index=[wealth.index.min() - pd.Timedelta(days=1)]), wealth]
    )
    running_max = wealth_with_initial.cummax()
    drawdown = 1.0 - wealth_with_initial / running_max
    total_return = float(wealth.iloc[-1] - 1.0)
    std = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean()) / std * math.sqrt(252.0)
        if std > 0.0
        else 0.0
    )
    max_drawdown = float(drawdown.max())
    annualized_return = float(returns.mean()) * 252.0
    calmar = annualized_return / max_drawdown if max_drawdown > 0.0 else 0.0
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def summarize_replay(
    replay: pd.DataFrame,
    *,
    market: str,
    seed: int,
    cost_rates: Sequence[float],
    reference_rate: float,
) -> pd.DataFrame:
    """Summarize every cost path and express changes from a reference rate."""

    rates = [float(rate) for rate in cost_rates]
    if float(reference_rate) not in rates:
        raise ValueError("reference_rate must be included in cost_rates")
    rows: list[dict[str, float | int | str]] = []
    for rate in rates:
        label = cost_label(rate)
        net_growth_column = f"net_growth_{label}"
        charged_column = f"charged_cost_rate_{label}"
        if net_growth_column not in replay or charged_column not in replay:
            raise ValueError(f"replay is missing columns for rate {rate}")
        metrics = _path_metrics(replay[net_growth_column])
        rows.append(
            {
                "market": str(market),
                "seed": int(seed),
                "transaction_cost_rate": rate,
                "transaction_cost_pct": rate * 100.0,
                "replay_days": int(len(replay)),
                **metrics,
                "mean_daily_turnover": float(replay["turnover"].mean()),
                "cumulative_charged_cost_rate": float(
                    replay[charged_column].sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    reference = summary.loc[
        np.isclose(summary["transaction_cost_rate"], float(reference_rate))
    ].iloc[0]
    summary["delta_total_return_pp"] = (
        summary["total_return"] - float(reference["total_return"])
    ) * 100.0
    summary["delta_sharpe"] = summary["sharpe"] - float(reference["sharpe"])
    summary["delta_max_drawdown_pp"] = (
        summary["max_drawdown"] - float(reference["max_drawdown"])
    ) * 100.0
    summary["delta_calmar"] = summary["calmar"] - float(reference["calmar"])
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_action_weights(path: Path) -> tuple[pd.DataFrame, pd.Series | None]:
    actions = pd.read_csv(path)
    required = {"date", "asset_names_json", "exec_weights_json"}
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"{path} is missing action columns: {sorted(missing)}")
    names = json.loads(actions["asset_names_json"].dropna().iloc[0])
    weights = pd.DataFrame(
        [json.loads(value) for value in actions["exec_weights_json"]],
        columns=names,
        index=pd.to_datetime(actions["date"]),
    )
    recorded = None
    if "cost_rate" in actions:
        recorded = pd.Series(
            actions["cost_rate"].to_numpy(dtype="float64"),
            index=pd.to_datetime(actions["date"]),
            name="recorded_cost_rate",
        )
    return weights, recorded


def load_recorded_trace(path: Path) -> pd.DataFrame:
    actions = pd.read_csv(
        path,
        usecols=["date", "exec_log_return", "turnover", "cost_rate"],
    )
    actions["date"] = pd.to_datetime(actions["date"])
    return actions.set_index("date").sort_index()


def load_price_panel(path: Path, assets: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["date", "tic", "adjclose"])
    frame["date"] = pd.to_datetime(frame["date"])
    panel = frame.pivot(index="date", columns="tic", values="adjclose").sort_index()
    missing = [asset for asset in assets if asset not in panel.columns]
    if missing:
        raise ValueError(f"{path} is missing price assets: {missing}")
    return panel.loc[:, list(assets)].ffill()


def _markdown_report(summary: pd.DataFrame, reference_rate: float) -> str:
    lines = [
        "# Fixed-Path Transaction-Cost Sensitivity",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        "- Scope: paper-selected environment-recorded paths held fixed",
        f"- Reference cost: {reference_rate * 100:.3f}%",
        "- Replay: recover pre-cost growth from recorded net return and turnover",
        "- Limitation: policy actions and Controller switches are not recomputed",
        "",
        "## Results",
        "",
        "| Market | Cost | TR | SR | MDD | CR | ΔTR (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"nas": "Nasdaq-100", "sh": "CSI-300"}
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.market, row.market)} "
            f"| {row.transaction_cost_pct:.3f}% "
            f"| {row.total_return * 100:.2f}% "
            f"| {row.sharpe:.3f} "
            f"| {row.max_drawdown * 100:.2f}% "
            f"| {row.calmar:.3f} "
            f"| {row.delta_total_return_pp:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This table isolates the mechanical effect of charging alternative costs "
            "to the same environment-recorded path. It does not represent inference "
            "or training under the alternative cost rates.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    replays: Mapping[str, pd.DataFrame],
    summaries: Sequence[pd.DataFrame],
    manifest: Mapping[str, object],
    reference_rate: float,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    metadata_dir = output_dir / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for market, replay in replays.items():
        replay.reset_index().to_csv(
            tables_dir / f"{market}_daily_replay.csv",
            index=False,
        )
    summary = pd.concat(list(summaries), ignore_index=True)
    summary.to_csv(tables_dir / "transaction_cost_sensitivity.csv", index=False)
    (output_dir / "TRANSACTION_COST_SENSITIVITY.md").write_text(
        _markdown_report(summary, reference_rate),
        encoding="utf-8",
    )
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_seed_specs(specs: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"invalid seed specification: {spec}")
        market, seed = spec.split(":", 1)
        result[market] = int(seed)
    return result


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-path transaction-cost replay for paper-selected models."
    )
    parser.add_argument("--full_actions_root", type=Path, required=True)
    parser.add_argument("--prices_root", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--markets", nargs="+", default=["nas", "sh"])
    parser.add_argument("--seeds", nargs="+", default=["nas:49", "sh:90"])
    parser.add_argument(
        "--cost_rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_COST_RATES),
    )
    parser.add_argument("--reference_rate", type=float, default=0.00005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_specs(args.seeds)
    replays: dict[str, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    manifest_markets: dict[str, dict[str, object]] = {}

    for market in args.markets:
        seed = seeds[market]
        action_path = (
            args.full_actions_root
            / f"{market}_seed{seed}_full_controller_inner_base_actions.csv"
        )
        price_path = args.prices_root / market / f"{market}_data.csv"
        run_root = args.results_root / f"{market}_seed{seed}"
        checkpoint_path = run_root / "checkpoints" / "best_model.pth"
        command_path = run_root / f"seed_{seed}_command.json"

        recorded = load_recorded_trace(action_path)
        replay = replay_recorded_trace(
            recorded,
            cost_rates=args.cost_rates,
            reference_rate=args.reference_rate,
        )
        replays[market] = replay
        summaries.append(
            summarize_replay(
                replay,
                market=market,
                seed=seed,
                cost_rates=args.cost_rates,
                reference_rate=args.reference_rate,
            )
        )
        manifest_markets[market] = {
            "seed": seed,
            "action_trace_path": str(action_path.resolve()),
            "action_trace_sha256": sha256_file(action_path),
            "price_path": str(price_path.resolve()),
            "price_sha256": sha256_file(price_path),
            "replay_days": len(replay),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "command_json_path": str(command_path.resolve()),
            "command_json_sha256": sha256_file(command_path),
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "full_actions_root": str(args.full_actions_root.resolve()),
            "prices_root": str(args.prices_root.resolve()),
            "results_root": str(args.results_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "markets": list(args.markets),
            "seeds": list(args.seeds),
            "cost_rates": [float(rate) for rate in args.cost_rates],
            "reference_rate": float(args.reference_rate),
        },
        "markets": manifest_markets,
        "replay_method": (
            "trace-calibrated fixed path: gross_growth = "
            "exp(exec_log_return) / (1 - reference_rate * turnover)"
        ),
        "scope": "fixed-path sensitivity; no policy reinference or retraining",
        "code_commit": _git_commit(Path(__file__).resolve().parents[1]),
    }
    summary = write_outputs(
        output_dir=args.output_dir,
        replays=replays,
        summaries=summaries,
        manifest=manifest,
        reference_rate=args.reference_rate,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
