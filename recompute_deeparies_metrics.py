#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute DeepAries backtest metrics from saved actions without retraining."
    )
    parser.add_argument(
        "--run_root",
        nargs="+",
        default=["checkpoints/deeparies_baseline"],
        help="One or more run roots to scan for DeepAries result folders.",
    )
    parser.add_argument(
        "--result_dir",
        nargs="+",
        default=None,
        help="Optional explicit DeepAries setting result dir(s), each containing actions/test.csv.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Optional explicit DeepAries input CSV. Use only with one --result_dir.",
    )
    parser.add_argument("--initial_amount", type=float, default=1000.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite backtest_metrics.csv, portfolio_performance.csv, and daily_weights.csv.",
    )
    parser.add_argument(
        "--write_artifacts",
        action="store_true",
        help="Also write recomputed daily portfolio and weight files.",
    )
    parser.add_argument(
        "--summary_name",
        default="deeparies_recomputed_metrics_summary.csv",
        help="Summary CSV name written under each run root.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Only process result dirs whose path contains this substring, e.g. 'sl(240)'.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Skip result dirs whose path contains this substring.",
    )
    return parser.parse_args()


def find_result_dirs(run_roots):
    result_dirs = []
    for root in run_roots:
        root = Path(root)
        if not root.exists():
            continue
        for actions_path in root.glob("**/actions/test.csv"):
            result_dirs.append(actions_path.parents[1])
    return sorted(set(result_dirs))


def infer_market_seed(result_dir):
    parts = result_dir.parts
    market = ""
    seed = ""
    for i, part in enumerate(parts):
        if part in {"sh", "nas"}:
            market = part
        if part.startswith("seed_"):
            seed = part.replace("seed_", "")
    return market, seed


def find_input_csv(result_dir, explicit_input=None):
    if explicit_input:
        path = Path(explicit_input)
        if path.exists():
            return path
        raise FileNotFoundError(f"input_csv not found: {path}")

    for parent in [result_dir, *result_dir.parents]:
        input_dir = parent / "input"
        if not input_dir.exists():
            continue
        candidates = [
            p for p in input_dir.glob("*/*_data.csv")
            if not p.name.endswith("_general_data.csv")
        ]
        if candidates:
            return sorted(candidates)[0]
    raise FileNotFoundError(f"Cannot find DeepAries input CSV for {result_dir}")


def load_close_panel(input_csv, tickers):
    raw = pd.read_csv(input_csv)
    price_col = "adjclose" if "adjclose" in raw.columns else "close" if "close" in raw.columns else None
    if price_col is None:
        raise ValueError(f"{input_csv} must contain adjclose or close.")
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw = raw[raw["tic"].isin(tickers)]
    close_panel = (
        raw.pivot(index="date", columns="tic", values=price_col)
        .sort_index()
        .reindex(columns=tickers)
        .ffill()
        .dropna(how="any")
    )
    if close_panel.empty:
        raise ValueError(f"No close panel rows after aligning tickers for {input_csv}")
    return close_panel


def load_weights(result_dir):
    actions_path = result_dir / "actions" / "test.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions/test.csv under {result_dir}")
    weights = pd.read_csv(actions_path, index_col=0)
    weights.index = pd.to_datetime(weights.index).tz_localize(None)
    weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return weights.sort_index()


def load_decisions(result_dir):
    path = result_dir / "rebalance_decisions.csv"
    if not path.exists():
        return None
    decisions = pd.read_csv(path)
    if "date" not in decisions.columns:
        return None
    decisions["date"] = pd.to_datetime(decisions["date"]).dt.tz_localize(None)
    if "end_date" in decisions.columns:
        decisions["end_date"] = pd.to_datetime(decisions["end_date"]).dt.tz_localize(None)
    return decisions.sort_values("date").reset_index(drop=True)


def decision_weight(weights, decisions, row_index, date):
    if date in weights.index:
        return weights.loc[date].to_numpy(dtype=np.float64)
    if len(weights) == len(decisions):
        return weights.iloc[row_index].to_numpy(dtype=np.float64)
    loc = weights.index.searchsorted(date, side="right") - 1
    if loc >= 0:
        return weights.iloc[loc].to_numpy(dtype=np.float64)
    return weights.iloc[0].to_numpy(dtype=np.float64)


def expand_daily_from_decisions(weights, decisions, close_panel):
    close_dates = close_panel.index
    returns = []
    weight_dates = []
    return_dates = []
    weight_rows = []

    for row_idx, row in decisions.iterrows():
        dec_date = row["date"]
        start_loc = int(close_dates.searchsorted(dec_date, side="left"))
        if start_loc < 0 or start_loc >= len(close_dates) - 1:
            continue

        if "end_date" in decisions.columns and pd.notna(row.get("end_date")):
            end_loc = int(close_dates.searchsorted(row["end_date"], side="right")) - 1
        else:
            horizon = int(row.get("chosen_horizon", 1))
            end_loc = start_loc + max(horizon, 1)
        end_loc = min(end_loc, len(close_dates) - 1)
        if end_loc <= start_loc:
            continue

        w = decision_weight(weights, decisions, row_idx, dec_date)
        for loc in range(start_loc, end_loc):
            current_close = close_panel.iloc[loc].to_numpy(dtype=np.float64)
            next_close = close_panel.iloc[loc + 1].to_numpy(dtype=np.float64)
            ret_vec = next_close / (current_close + 1e-12) - 1.0
            returns.append(float(np.sum(w * ret_vec)))
            weight_dates.append(close_dates[loc])
            return_dates.append(close_dates[loc + 1])
            weight_rows.append(w)

    return returns, weight_dates, return_dates, weight_rows


def expand_daily_from_weights(weights, close_panel):
    close_dates = close_panel.index
    weights = weights.reindex(close_dates).ffill().dropna(how="all")
    returns = []
    weight_dates = []
    return_dates = []
    weight_rows = []

    for date, w_row in weights.iloc[:-1].iterrows():
        loc = close_dates.get_loc(date)
        current_close = close_panel.iloc[loc].to_numpy(dtype=np.float64)
        next_close = close_panel.iloc[loc + 1].to_numpy(dtype=np.float64)
        ret_vec = next_close / (current_close + 1e-12) - 1.0
        w = w_row.to_numpy(dtype=np.float64)
        returns.append(float(np.sum(w * ret_vec)))
        weight_dates.append(date)
        return_dates.append(close_dates[loc + 1])
        weight_rows.append(w)
    return returns, weight_dates, return_dates, weight_rows


def compute_metrics(returns, portfolio_values, portfolio_dates):
    returns = np.asarray(returns, dtype=np.float64)
    values = np.asarray(portfolio_values, dtype=np.float64)
    initial = float(values[0])
    final = float(values[-1])
    cumulative_return = final / initial - 1.0
    ann_return = float(returns.mean() * 252.0) if len(returns) else 0.0
    ann_vol = float(returns.std() * np.sqrt(252.0)) if len(returns) else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    drawdown = values / np.maximum.accumulate(values) - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    dates = pd.to_datetime(portfolio_dates)
    days = int((dates[-1] - dates[0]).days) if len(dates) >= 2 else 0
    cagr = float((final / initial) ** (365.0 / days) - 1.0) if days > 0 else 0.0
    return {
        "initial_value": initial,
        "final_value": final,
        "cumulative_return": float(cumulative_return),
        "annualized_return": ann_return,
        "annualized_return_cagr": cagr,
        "annualized_return_mean": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": max_drawdown,
        "metric_frequency": "daily_close_to_close",
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "num_daily_returns": int(len(returns)),
    }


def recompute_result(result_dir, input_csv=None, initial_amount=1000.0, overwrite=False, write_artifacts=False):
    result_dir = Path(result_dir)
    weights = load_weights(result_dir)
    input_csv = find_input_csv(result_dir, input_csv)
    close_panel = load_close_panel(input_csv, list(weights.columns))
    weights = weights.reindex(columns=close_panel.columns).fillna(0.0)
    decisions = load_decisions(result_dir)

    if decisions is not None:
        returns, weight_dates, return_dates, weight_rows = expand_daily_from_decisions(weights, decisions, close_panel)
        num_rebalances = int(len(decisions))
    else:
        returns, weight_dates, return_dates, weight_rows = expand_daily_from_weights(weights, close_panel)
        num_rebalances = int(len(weights))

    if not returns:
        raise ValueError(f"No daily returns could be reconstructed for {result_dir}")

    portfolio_values = [float(initial_amount)]
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1.0 + float(ret)))
    portfolio_dates = [pd.to_datetime(weight_dates[0]), *pd.to_datetime(return_dates).tolist()]

    metrics = compute_metrics(returns, portfolio_values, portfolio_dates)
    metrics["num_rebalances"] = num_rebalances
    metrics["result_dir"] = str(result_dir)
    metrics["input_csv"] = str(input_csv)

    suffix = "" if overwrite else "_recomputed"
    metrics_path = result_dir / f"backtest_metrics{suffix}.csv"
    json_path = result_dir / f"backtest_metrics{suffix}.json"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.Series(metrics).to_json(json_path, force_ascii=False, indent=2)

    if write_artifacts or overwrite:
        portfolio_df = pd.DataFrame({
            "date": pd.to_datetime(portfolio_dates),
            "portfolio_value": portfolio_values,
        })
        portfolio_df["period_return"] = portfolio_df["portfolio_value"].pct_change().fillna(0.0)
        portfolio_df.to_csv(result_dir / f"portfolio_performance{suffix}.csv", index=False)
        weights_df = pd.DataFrame(weight_rows, index=pd.to_datetime(weight_dates), columns=close_panel.columns)
        weights_df.to_csv(result_dir / f"daily_weights{suffix}.csv")

    return metrics


def write_summary(root, rows, summary_name):
    if not rows:
        return None
    path = Path(root) / summary_name
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    if args.result_dir:
        result_dirs = [Path(p) for p in args.result_dir]
        if args.input_csv and len(result_dirs) != 1:
            raise ValueError("--input_csv can only be used with exactly one --result_dir")
    else:
        result_dirs = find_result_dirs(args.run_root)
    if args.include:
        result_dirs = [p for p in result_dirs if args.include in str(p)]
    if args.exclude:
        result_dirs = [p for p in result_dirs if args.exclude not in str(p)]

    rows = []
    for result_dir in result_dirs:
        market, seed = infer_market_seed(result_dir)
        metrics = recompute_result(
            result_dir,
            input_csv=args.input_csv,
            initial_amount=args.initial_amount,
            overwrite=args.overwrite,
            write_artifacts=args.write_artifacts,
        )
        row = {
            "market": market,
            "seed": seed,
            "setting": result_dir.name,
            **metrics,
        }
        rows.append(row)
        print(
            f"{result_dir}: cumulative={metrics['cumulative_return']:.4f}, "
            f"ann={metrics['annualized_return']:.4f}, sharpe={metrics['sharpe']:.4f}, "
            f"mdd={metrics['max_drawdown']:.4f}"
        )

    for root in args.run_root:
        root_path = Path(root)
        root_rows = [r for r in rows if Path(r["result_dir"]).is_relative_to(root_path)]
        summary_path = write_summary(root_path, root_rows, args.summary_name)
        if summary_path:
            print(f"summary: {summary_path}")

    if args.result_dir and rows:
        print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
