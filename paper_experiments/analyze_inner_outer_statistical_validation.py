#!/usr/bin/env python3
"""Validate how the Inner actor refines and complements the Outer portfolio."""

from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class WeightTrace:
    base: pd.DataFrame
    executed: pd.DataFrame
    tilt: pd.DataFrame


def _parse_weight_matrix(actions: pd.DataFrame, column: str, names: list[str]) -> pd.DataFrame:
    rows = []
    for row_number, value in enumerate(actions[column]):
        parsed = json.loads(value)
        if len(parsed) != len(names):
            raise ValueError(
                f"{column} row {row_number} has {len(parsed)} values; expected {len(names)}"
            )
        rows.append(parsed)
    matrix = pd.DataFrame(rows, columns=names, index=pd.to_datetime(actions["date"]))
    matrix = matrix.astype("float64")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(f"{column} contains non-finite weights")
    return matrix


def parse_weight_trace(actions: pd.DataFrame) -> WeightTrace:
    required = {
        "date",
        "asset_names_json",
        "base_weights_json",
        "exec_weights_json",
        "inner_tilt_json",
    }
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(f"action trace is missing columns: {missing}")
    if actions.empty:
        raise ValueError("action trace is empty")

    dates = pd.to_datetime(actions["date"])
    if dates.duplicated().any():
        raise ValueError("action trace contains duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("action trace dates must be sorted")

    names = json.loads(actions["asset_names_json"].dropna().iloc[0])
    if len(names) != len(set(names)):
        raise ValueError("asset_names_json contains duplicate assets")

    return WeightTrace(
        base=_parse_weight_matrix(actions, "base_weights_json", names),
        executed=_parse_weight_matrix(actions, "exec_weights_json", names),
        tilt=_parse_weight_matrix(actions, "inner_tilt_json", names),
    )


def validate_weight_invariants(
    trace: WeightTrace,
    *,
    atol: float = 1e-7,
) -> dict[str, float | int]:
    base = trace.base.to_numpy(dtype="float64")
    executed = trace.executed.to_numpy(dtype="float64")
    tilt = trace.tilt.to_numpy(dtype="float64")
    identity_error = tilt - (executed - base)
    weight_sum_error = np.concatenate(
        [base.sum(axis=1) - 1.0, executed.sum(axis=1) - 1.0, tilt.sum(axis=1)]
    )
    support_violations = (base <= atol) & (executed > atol)
    negative_weights = (base < -atol) | (executed < -atol)
    return {
        "max_abs_tilt_identity_error": float(np.max(np.abs(identity_error))),
        "max_abs_weight_sum_error": float(np.max(np.abs(weight_sum_error))),
        "support_violation_count": int(support_violations.sum()),
        "negative_weight_count": int(negative_weights.sum()),
    }


def _require_aligned_weights(base: pd.DataFrame, executed: pd.DataFrame) -> None:
    if not base.index.equals(executed.index):
        raise ValueError("base and executed weights must have identical dates")
    if list(base.columns) != list(executed.columns):
        raise ValueError("base and executed weights must have identical assets")


def configuration_shape_metrics(
    base: pd.DataFrame,
    executed: pd.DataFrame,
) -> pd.DataFrame:
    _require_aligned_weights(base, executed)
    base_values = base.to_numpy(dtype="float64")
    exec_values = executed.to_numpy(dtype="float64")
    base_hhi = np.square(base_values).sum(axis=1)
    exec_hhi = np.square(exec_values).sum(axis=1)
    return pd.DataFrame(
        {
            "active_share": 0.5 * np.abs(exec_values - base_values).sum(axis=1),
            "base_hhi": base_hhi,
            "exec_hhi": exec_hhi,
            "delta_hhi": exec_hhi - base_hhi,
            "base_effective_n": np.divide(
                1.0,
                base_hhi,
                out=np.full_like(base_hhi, np.nan),
                where=base_hhi > 0,
            ),
            "exec_effective_n": np.divide(
                1.0,
                exec_hhi,
                out=np.full_like(exec_hhi, np.nan),
                where=exec_hhi > 0,
            ),
        },
        index=base.index,
    ).assign(
        delta_effective_n=lambda frame: frame["exec_effective_n"] - frame["base_effective_n"]
    )


def estimate_covariance(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype="float64")
    if values.ndim != 2:
        raise ValueError("covariance window must be two-dimensional")
    if values.shape[0] < 2:
        return np.zeros((values.shape[1], values.shape[1]), dtype="float64")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(values - values.mean(axis=0, keepdims=True))) < 1e-15:
        return np.zeros((values.shape[1], values.shape[1]), dtype="float64")
    try:
        from sklearn.covariance import LedoitWolf

        covariance = LedoitWolf().fit(values).covariance_
    except (ImportError, ValueError, np.linalg.LinAlgError):
        sample = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        diagonal = np.diag(np.diag(sample))
        covariance = 0.9 * sample + 0.1 * diagonal
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _portfolio_volatility(weight: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weight @ covariance @ weight)
    return float(np.sqrt(max(variance, 0.0)) * np.sqrt(252.0))


def ex_ante_risk_metrics(
    base: pd.DataFrame,
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    _require_aligned_weights(base, executed)
    if lookback < 2:
        raise ValueError("lookback must be at least two days")
    missing_assets = sorted(set(base.columns).difference(asset_returns.columns))
    if missing_assets:
        raise ValueError(f"asset returns are missing columns: {missing_assets}")
    returns = asset_returns.loc[:, base.columns].sort_index()
    rows = []
    for date, base_row, exec_row in zip(base.index, base.to_numpy(), executed.to_numpy()):
        history = returns.loc[:date].tail(int(lookback)).to_numpy(dtype="float64")
        if len(history) < 2:
            rows.append(
                {
                    "base_ex_ante_vol": np.nan,
                    "exec_ex_ante_vol": np.nan,
                    "delta_ex_ante_vol": np.nan,
                    "base_downside_vol": np.nan,
                    "exec_downside_vol": np.nan,
                    "delta_downside_vol": np.nan,
                }
            )
            continue
        covariance = estimate_covariance(history)
        downside_covariance = estimate_covariance(np.minimum(history, 0.0))
        base_vol = _portfolio_volatility(base_row, covariance)
        exec_vol = _portfolio_volatility(exec_row, covariance)
        base_down = _portfolio_volatility(base_row, downside_covariance)
        exec_down = _portfolio_volatility(exec_row, downside_covariance)
        rows.append(
            {
                "base_ex_ante_vol": base_vol,
                "exec_ex_ante_vol": exec_vol,
                "delta_ex_ante_vol": exec_vol - base_vol,
                "base_downside_vol": base_down,
                "exec_downside_vol": exec_down,
                "delta_downside_vol": exec_down - base_down,
            }
        )
    return pd.DataFrame(rows, index=base.index)


def attach_market_volatility_regime(
    daily: pd.DataFrame,
    equal_weight_market_return: pd.Series,
    *,
    lookback: int = 20,
) -> tuple[pd.DataFrame, dict[str, float]]:
    market_return = equal_weight_market_return.sort_index().astype("float64")
    realized_volatility = market_return.rolling(int(lookback), min_periods=int(lookback)).std(ddof=1)
    aligned = realized_volatility.reindex(daily.index)
    finite = aligned[np.isfinite(aligned)]
    if finite.empty:
        cuts = {"lower": np.nan, "upper": np.nan}
        labels = pd.Series(pd.NA, index=daily.index, dtype="object")
    else:
        lower, upper = finite.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
        cuts = {"lower": float(lower), "upper": float(upper)}
        labels = pd.Series(
            np.select(
                [aligned <= lower, aligned <= upper, aligned > upper],
                ["low", "mid", "high"],
                default=None,
            ),
            index=daily.index,
            dtype="object",
        )
    result = daily.copy()
    result["market_realized_vol_20d"] = aligned
    result["volatility_regime"] = pd.Categorical(
        labels,
        categories=["low", "mid", "high"],
        ordered=True,
    )
    return result, cuts


def drift_weights(previous_target: np.ndarray, gross_ratio: np.ndarray) -> np.ndarray:
    drifted = np.asarray(previous_target, dtype="float64") * np.asarray(
        gross_ratio, dtype="float64"
    )
    total = float(drifted.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("drifted weights have a non-positive total")
    return drifted / total


def frozen_path_direct_effect(
    base: pd.DataFrame,
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    transaction_cost_pct: float,
) -> pd.DataFrame:
    _require_aligned_weights(base, executed)
    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative")
    missing_assets = sorted(set(base.columns).difference(prices.columns))
    if missing_assets:
        raise ValueError(f"prices are missing columns: {missing_assets}")
    price_panel = prices.loc[:, base.columns].sort_index().astype("float64")
    rows = []
    for position in range(1, len(base)):
        previous_date = pd.Timestamp(base.index[position - 1])
        date = pd.Timestamp(base.index[position])
        later_dates = price_panel.index[price_panel.index > date]
        if not len(later_dates):
            continue
        next_date = pd.Timestamp(later_dates[0])
        if previous_date not in price_panel.index or date not in price_panel.index:
            raise ValueError(f"prices do not contain decision dates {previous_date} and {date}")
        previous_to_today = (
            price_panel.loc[date].to_numpy() / price_panel.loc[previous_date].to_numpy()
        )
        today_to_next = (
            price_panel.loc[next_date].to_numpy() / price_panel.loc[date].to_numpy()
        )
        base_drift = drift_weights(base.iloc[position - 1].to_numpy(), previous_to_today)
        exec_drift = drift_weights(executed.iloc[position - 1].to_numpy(), previous_to_today)
        base_target = base.iloc[position].to_numpy(dtype="float64")
        exec_target = executed.iloc[position].to_numpy(dtype="float64")
        base_turnover = float(np.abs(base_target - base_drift).sum())
        exec_turnover = float(np.abs(exec_target - exec_drift).sum())
        base_cost_rate = float(transaction_cost_pct * base_turnover)
        exec_cost_rate = float(transaction_cost_pct * exec_turnover)
        base_growth = float((1.0 - base_cost_rate) * np.dot(base_target, today_to_next))
        exec_growth = float((1.0 - exec_cost_rate) * np.dot(exec_target, today_to_next))
        if base_growth <= 0 or exec_growth <= 0:
            raise ValueError("net portfolio growth must be positive")
        base_net_log_return = float(np.log(base_growth))
        exec_net_log_return = float(np.log(exec_growth))
        rows.append(
            {
                "date": date,
                "next_date": next_date,
                "exec_net_log_return": exec_net_log_return,
                "base_net_log_return": base_net_log_return,
                "delta_net_log_return": exec_net_log_return - base_net_log_return,
                "exec_turnover": exec_turnover,
                "base_turnover": base_turnover,
                "exec_cost_rate": exec_cost_rate,
                "base_cost_rate": base_cost_rate,
                "market_simple_return": float(np.mean(today_to_next - 1.0)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "next_date",
                "exec_net_log_return",
                "base_net_log_return",
                "delta_net_log_return",
                "exec_turnover",
                "base_turnover",
                "exec_cost_rate",
                "base_cost_rate",
                "market_simple_return",
            ]
        )
    return pd.DataFrame(rows).set_index("date")


def newey_west_mean_test(
    values: Sequence[float],
    *,
    maxlags: int = 5,
) -> dict[str, float]:
    import statsmodels.api as sm

    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": np.nan, "se": np.nan, "t_stat": np.nan, "p_value": np.nan}
    if np.max(np.abs(array - array.mean())) < 1e-15:
        mean = float(array.mean())
        if abs(mean) < 1e-15:
            return {"mean": mean, "se": 0.0, "t_stat": 0.0, "p_value": 1.0}
        return {
            "mean": mean,
            "se": 0.0,
            "t_stat": float(np.sign(mean) * np.inf),
            "p_value": 0.0,
        }
    fit = sm.OLS(array, np.ones((len(array), 1))).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": min(int(maxlags), max(len(array) - 1, 0)),
            "use_correction": True,
        },
    )
    return {
        "mean": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
    }


def circular_block_bootstrap(
    arrays: Sequence[np.ndarray],
    statistic: Callable[..., np.ndarray | float],
    *,
    block_length: int,
    reps: int,
    seed: int,
) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one array is required")
    prepared = [np.asarray(array) for array in arrays]
    sample_size = len(prepared[0])
    if any(len(array) != sample_size for array in prepared):
        raise ValueError("paired bootstrap arrays must have equal lengths")
    if sample_size == 0 or block_length < 1 or reps < 1:
        raise ValueError("sample size, block length, and reps must be positive")
    rng = np.random.default_rng(int(seed))
    block_count = int(np.ceil(sample_size / int(block_length)))
    offsets = np.arange(int(block_length))
    results = []
    for _ in range(int(reps)):
        starts = rng.integers(0, sample_size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % sample_size).reshape(-1)
        indices = indices[:sample_size]
        results.append(statistic(*(array[indices] for array in prepared)))
    return np.asarray(results)


def _expected_shortfall(values: np.ndarray, probability: float = 0.05) -> float:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    count = max(1, int(np.ceil(len(values) * probability)))
    return float(np.sort(values)[:count].mean())


def summarize_frozen_path(
    direct: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float | int]:
    required = {
        "exec_net_log_return",
        "base_net_log_return",
        "delta_net_log_return",
        "exec_turnover",
        "base_turnover",
        "exec_cost_rate",
        "base_cost_rate",
        "market_simple_return",
    }
    missing = sorted(required.difference(direct.columns))
    if missing:
        raise ValueError(f"direct effect table is missing columns: {missing}")
    clean = direct.dropna(subset=list(required))
    delta = clean["delta_net_log_return"].to_numpy(dtype="float64")
    hac = newey_west_mean_test(delta, maxlags=5)
    boot = circular_block_bootstrap(
        [delta],
        lambda values: float(np.mean(values)),
        block_length=block_length,
        reps=bootstrap_reps,
        seed=seed,
    )
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    std = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
    market_cut = float(clean["market_simple_return"].quantile(0.05))
    worst_market = clean["market_simple_return"] <= market_cut
    return {
        "days": int(len(clean)),
        "mean_net_alpha_bp_day": float(hac["mean"] * 10000.0),
        "nw_se_bp_day": float(hac["se"] * 10000.0),
        "nw_t_stat": float(hac["t_stat"]),
        "nw_p_value": float(hac["p_value"]),
        "block_ci_low_bp_day": float(ci_low * 10000.0),
        "block_ci_high_bp_day": float(ci_high * 10000.0),
        "positive_alpha_ratio": float(np.mean(delta > 0)),
        "annualized_alpha_sharpe": (
            float(np.mean(delta) / std * np.sqrt(252.0))
            if np.isfinite(std) and std > 0
            else np.nan
        ),
        "cumulative_net_alpha": float(np.exp(np.sum(delta)) - 1.0),
        "mean_exec_turnover": float(clean["exec_turnover"].mean()),
        "mean_base_turnover": float(clean["base_turnover"].mean()),
        "mean_incremental_cost_bp": float(
            (clean["exec_cost_rate"] - clean["base_cost_rate"]).mean() * 10000.0
        ),
        "daily_volatility_difference": float(
            clean["exec_net_log_return"].std(ddof=1)
            - clean["base_net_log_return"].std(ddof=1)
        ),
        "expected_shortfall_5_difference": float(
            _expected_shortfall(clean["exec_net_log_return"].to_numpy())
            - _expected_shortfall(clean["base_net_log_return"].to_numpy())
        ),
        "worst_market_5_mean_delta": float(
            clean.loc[worst_market, "delta_net_log_return"].mean()
        ),
    }


def _trace_cache_paths(
    output_dir: Path,
    market: str,
    seed: int,
    scenario: str,
) -> dict[str, Path]:
    trace_dir = Path(output_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{market}_seed{int(seed)}_{scenario}"
    return {
        "portfolio": trace_dir / f"{prefix}_portfolio.csv",
        "actions": trace_dir / f"{prefix}_actions.csv",
        "switch_events": trace_dir / f"{prefix}_switch_events.csv",
    }


def _cached_trace_bundle(paths: dict[str, Path]) -> dict[str, pd.DataFrame] | None:
    if not all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        return None
    bundle = {name: pd.read_csv(path) for name, path in paths.items()}
    if not {"date", "portfolio_value", "daily_log_return"}.issubset(
        bundle["portfolio"].columns
    ):
        return None
    if not {
        "date",
        "base_weights_json",
        "exec_weights_json",
        "inner_tilt_json",
    }.issubset(bundle["actions"].columns):
        return None
    return bundle


def ensure_closed_loop_trace(
    *,
    results_root: Path,
    output_dir: Path,
    market: str,
    seed: int,
    scenario: str,
    device: str,
    force_eval: bool,
) -> dict[str, pd.DataFrame]:
    if scenario not in {"full_controller", "controller_outer"}:
        raise ValueError(f"unsupported closed-loop scenario: {scenario}")
    paths = _trace_cache_paths(Path(output_dir), market, seed, scenario)
    if not force_eval:
        cached = _cached_trace_bundle(paths)
        if cached is not None:
            return cached

    from paper_experiments.eval_end_to_end_explain import (
        build_loaded_trainer,
        collect_eval_trace,
        load_checkpoint_into_trainer,
    )
    from paper_experiments.trace_utils import discover_runs

    runs = discover_runs(
        Path(results_root),
        markets=[market],
        seed_map={market: [int(seed)]},
    )
    if not runs:
        raise RuntimeError(f"no run found for {market}:seed{seed}")
    run = runs[0]
    trainer, _, torch_module = build_loaded_trainer(
        run,
        output_dir=Path(output_dir) / "_runtime",
        device=device,
        results_root=Path(results_root),
    )
    checkpoint = run.checkpoints["best_model"]
    if not load_checkpoint_into_trainer(trainer, torch_module, checkpoint.path):
        raise RuntimeError(f"could not load checkpoint: {checkpoint.path}")
    bundle = collect_eval_trace(
        trainer,
        scenario=scenario,
        fixed_cycle=None,
        disable_inner=scenario == "controller_outer",
    )
    for name, path in paths.items():
        bundle[name].to_csv(path, index=False)
    return bundle


def align_closed_loop_returns(
    full_portfolio: pd.DataFrame,
    no_inner_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    required = {"date", "daily_log_return"}
    for label, frame in [("full", full_portfolio), ("no_inner", no_inner_portfolio)]:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} portfolio is missing columns: {missing}")
    full = full_portfolio.loc[:, ["date", "daily_log_return"]].copy()
    no_inner = no_inner_portfolio.loc[:, ["date", "daily_log_return"]].copy()
    full["date"] = pd.to_datetime(full["date"])
    no_inner["date"] = pd.to_datetime(no_inner["date"])
    full = full.rename(columns={"daily_log_return": "full_log_return"})
    no_inner = no_inner.rename(columns={"daily_log_return": "no_inner_log_return"})
    paired = full.merge(no_inner, on="date", how="inner", validate="one_to_one")
    paired = paired.sort_values("date").set_index("date")
    paired["difference_log_return"] = (
        paired["full_log_return"] - paired["no_inner_log_return"]
    )
    return paired


def portfolio_path_metrics(log_returns: Sequence[float]) -> dict[str, float]:
    log_returns = np.asarray(log_returns, dtype="float64")
    log_returns = log_returns[np.isfinite(log_returns)]
    if not len(log_returns):
        return {
            "mean_daily_return": np.nan,
            "total_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "expected_shortfall_5": np.nan,
        }
    simple_returns = np.expm1(log_returns)
    wealth = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    peaks = np.maximum.accumulate(wealth)
    max_drawdown = float(np.max((peaks - wealth) / np.maximum(peaks, 1e-12)))
    volatility = float(np.std(simple_returns, ddof=1)) if len(simple_returns) > 1 else np.nan
    annual_return = float(np.mean(simple_returns) * 252.0)
    sharpe = (
        float(np.mean(simple_returns) / volatility * np.sqrt(252.0))
        if np.isfinite(volatility) and volatility > 1e-12
        else np.nan
    )
    return {
        "mean_daily_return": float(np.mean(simple_returns)),
        "total_return": float(wealth[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": (
            float(annual_return / max_drawdown) if max_drawdown > 1e-12 else np.nan
        ),
        "expected_shortfall_5": _expected_shortfall(simple_returns),
    }


def summarize_closed_loop(
    paired: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"full_log_return", "no_inner_log_return", "difference_log_return"}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"paired closed-loop table is missing columns: {missing}")
    clean = paired.dropna(subset=list(required))
    full = clean["full_log_return"].to_numpy(dtype="float64")
    no_inner = clean["no_inner_log_return"].to_numpy(dtype="float64")
    metric_names = [
        "mean_daily_return",
        "total_return",
        "sharpe",
        "max_drawdown",
        "calmar",
        "expected_shortfall_5",
    ]

    def metric_differences(full_sample, no_inner_sample):
        full_metrics = portfolio_path_metrics(full_sample)
        no_inner_metrics = portfolio_path_metrics(no_inner_sample)
        return np.asarray(
            [full_metrics[name] - no_inner_metrics[name] for name in metric_names],
            dtype="float64",
        )

    boot_values = circular_block_bootstrap(
        [full, no_inner],
        metric_differences,
        block_length=block_length,
        reps=bootstrap_reps,
        seed=seed,
    )
    bootstrap = pd.DataFrame(boot_values, columns=metric_names)
    full_metrics = portfolio_path_metrics(full)
    no_inner_metrics = portfolio_path_metrics(no_inner)
    rows = []
    for name in metric_names:
        values = bootstrap[name].replace([np.inf, -np.inf], np.nan).dropna()
        ci_low, ci_high = (
            np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)
        )
        rows.append(
            {
                "metric": name,
                "full": full_metrics[name],
                "no_inner": no_inner_metrics[name],
                "difference": full_metrics[name] - no_inner_metrics[name],
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_reps": int(bootstrap_reps),
            }
        )
    return pd.DataFrame(rows), bootstrap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_model_identity(
    results_root: Path,
    market: str,
    seed: int,
) -> dict[str, str]:
    run_dir = Path(results_root) / f"{market}_seed{int(seed)}"
    checkpoint = run_dir / "checkpoints" / "best_model.pth"
    command_json = run_dir / f"seed_{int(seed)}_command.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    if not command_json.exists():
        raise FileNotFoundError(f"missing command json: {command_json}")
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "command_json_path": str(command_json.resolve()),
        "command_json_sha256": sha256_file(command_json),
    }


def permute_tilt_within_support(
    base: pd.DataFrame,
    tilt: pd.DataFrame,
    *,
    seed: int,
    max_attempts: int = 100,
) -> tuple[pd.DataFrame, int]:
    if not base.index.equals(tilt.index) or list(base.columns) != list(tilt.columns):
        raise ValueError("base and tilt must be aligned")
    rng = np.random.default_rng(int(seed))
    result = base.copy().astype("float64")
    invalid_rows = 0
    for date in base.index:
        base_row = base.loc[date].to_numpy(dtype="float64")
        tilt_row = tilt.loc[date].to_numpy(dtype="float64")
        support = np.flatnonzero(base_row > 1e-12)
        if not len(support):
            continue
        accepted = None
        for _ in range(int(max_attempts)):
            candidate = base_row.copy()
            candidate[support] += rng.permutation(tilt_row[support])
            if np.min(candidate) >= -1e-12 and abs(candidate.sum() - 1.0) <= 1e-7:
                candidate = np.clip(candidate, 0.0, None)
                candidate /= candidate.sum()
                accepted = candidate
                break
        if accepted is None:
            invalid_rows += 1
            accepted = base_row + tilt_row
        result.loc[date] = accepted
    return result, invalid_rows


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype="float64")
    adjusted = np.full(values.shape, np.nan, dtype="float64")
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    corrected = ranked * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    restored = np.empty_like(corrected)
    restored[order] = corrected
    adjusted[finite_indices] = restored
    return adjusted


def _parse_seed_specs(specs: Sequence[str]) -> dict[str, int]:
    result = {"nas": 49, "sh": 90}
    for spec in specs:
        market, separator, seed = str(spec).partition(":")
        if not separator:
            raise ValueError(f"invalid seed specification: {spec}")
        result[market] = int(seed)
    return result


def _load_prices(prices_root: Path, market: str, assets: Sequence[str]) -> pd.DataFrame:
    path = Path(prices_root) / market / f"{market}_data.csv"
    frame = pd.read_csv(path, usecols=["date", "tic", "adjclose"])
    frame["date"] = pd.to_datetime(frame["date"])
    prices = frame.pivot(index="date", columns="tic", values="adjclose").sort_index()
    missing = sorted(set(assets).difference(prices.columns))
    if missing:
        raise ValueError(f"{path} is missing assets: {missing}")
    return prices.loc[:, list(assets)].ffill()


def _block_mean_interval(
    values: Sequence[float],
    *,
    block_length: int,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    boot = circular_block_bootstrap(
        [array],
        lambda sample: float(np.mean(sample)),
        block_length=block_length,
        reps=reps,
        seed=seed,
    )
    return tuple(float(value) for value in np.quantile(boot, [0.025, 0.975]))


def summarize_configuration(
    daily: pd.DataFrame,
    *,
    primary_risk_window: int,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float | int]:
    risk_column = f"delta_ex_ante_vol_{int(primary_risk_window)}d"
    downside_column = f"delta_downside_vol_{int(primary_risk_window)}d"
    risk = daily[risk_column].dropna().to_numpy(dtype="float64")
    downside = daily[downside_column].dropna().to_numpy(dtype="float64")
    risk_hac = newey_west_mean_test(risk, maxlags=5)
    downside_hac = newey_west_mean_test(downside, maxlags=5)
    risk_ci = _block_mean_interval(
        risk,
        block_length=block_length,
        reps=bootstrap_reps,
        seed=seed,
    )
    downside_ci = _block_mean_interval(
        downside,
        block_length=block_length,
        reps=bootstrap_reps,
        seed=seed + 1,
    )
    active = daily["active_share"]
    return {
        "days": int(len(daily)),
        "mean_active_share": float(active.mean()),
        "median_active_share": float(active.median()),
        "p90_active_share": float(active.quantile(0.90)),
        "active_share_gt_1pct": float((active > 0.01).mean()),
        "active_share_gt_5pct": float((active > 0.05).mean()),
        "active_share_gt_10pct": float((active > 0.10).mean()),
        "mean_delta_hhi": float(daily["delta_hhi"].mean()),
        "mean_delta_effective_n": float(daily["delta_effective_n"].mean()),
        "mean_delta_ex_ante_vol": float(np.mean(risk)),
        "delta_ex_ante_vol_nw_t": float(risk_hac["t_stat"]),
        "delta_ex_ante_vol_p": float(risk_hac["p_value"]),
        "delta_ex_ante_vol_ci_low": risk_ci[0],
        "delta_ex_ante_vol_ci_high": risk_ci[1],
        "risk_reduction_days": float(np.mean(risk < 0)),
        "mean_delta_downside_vol": float(np.mean(downside)),
        "delta_downside_vol_nw_t": float(downside_hac["t_stat"]),
        "delta_downside_vol_p": float(downside_hac["p_value"]),
        "delta_downside_vol_ci_low": downside_ci[0],
        "delta_downside_vol_ci_high": downside_ci[1],
        "downside_risk_reduction_days": float(np.mean(downside < 0)),
    }


def run_placebo_analysis(
    base: pd.DataFrame,
    tilt: pd.DataFrame,
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    transaction_cost_pct: float,
    risk_lookback: int,
    reps: int,
    seed: int,
    observed: dict[str, float | int],
    observed_mean_delta_exante_vol: float,
) -> dict[str, float | int]:
    observed_cumulative = float(observed["cumulative_net_alpha"])
    prepared = _prepare_frozen_path_arrays(base, prices)
    risk_covariances, base_risk, risk_valid = _prepare_exante_risk_arrays(
        base,
        asset_returns,
        lookback=risk_lookback,
    )
    base_values = base.to_numpy(dtype="float64")
    tilt_values = tilt.to_numpy(dtype="float64")
    rng = np.random.default_rng(int(seed))
    placebo_values = []
    placebo_risk_values = []
    invalid_rows = 0
    positions = prepared["positions"]
    previous_positions = positions - 1
    previous_ratios = prepared["previous_ratios"]
    next_ratios = prepared["next_ratios"]
    base_drift = base_values[previous_positions] * previous_ratios
    base_drift /= base_drift.sum(axis=1, keepdims=True)
    base_targets = base_values[positions]
    base_turnover = np.abs(base_targets - base_drift).sum(axis=1)
    base_growth = (
        1.0 - float(transaction_cost_pct) * base_turnover
    ) * np.sum(base_targets * next_ratios, axis=1)
    batch_size = min(100, int(reps))
    for batch_start in range(0, int(reps), batch_size):
        current_batch_size = min(batch_size, int(reps) - batch_start)
        batch = np.empty(
            (current_batch_size, len(base_values), base_values.shape[1]),
            dtype="float64",
        )
        for batch_index in range(current_batch_size):
            permuted_values, invalid = _permuted_executed_numpy(
                base_values,
                tilt_values,
                rng,
            )
            invalid_rows += invalid
            batch[batch_index] = permuted_values
        exec_drift = batch[:, previous_positions, :] * previous_ratios[None, :, :]
        exec_drift /= exec_drift.sum(axis=2, keepdims=True)
        exec_targets = batch[:, positions, :]
        exec_turnover = np.abs(exec_targets - exec_drift).sum(axis=2)
        exec_growth = (
            1.0 - float(transaction_cost_pct) * exec_turnover
        ) * np.sum(exec_targets * next_ratios[None, :, :], axis=2)
        delta = np.log(np.maximum(exec_growth, 1e-12)) - np.log(
            np.maximum(base_growth[None, :], 1e-12)
        )
        placebo_values.extend(np.expm1(delta.sum(axis=1)).tolist())
        exec_variance = np.einsum(
            "bti,tij,btj->bt",
            batch,
            risk_covariances,
            batch,
            optimize=True,
        )
        exec_risk = np.sqrt(np.clip(exec_variance, 0.0, None)) * np.sqrt(252.0)
        placebo_risk_values.extend(
            np.mean((exec_risk - base_risk[None, :])[:, risk_valid], axis=1).tolist()
        )
    placebo = np.asarray(placebo_values, dtype="float64")
    placebo_risk = np.asarray(placebo_risk_values, dtype="float64")
    positive_p = float(
        (1 + np.sum(placebo >= observed_cumulative)) / (len(placebo) + 1)
    )
    return {
        "placebo_reps": int(reps),
        "observed_cumulative_net_alpha": observed_cumulative,
        "placebo_mean_cumulative_net_alpha": float(placebo.mean()),
        "placebo_ci_low": float(np.quantile(placebo, 0.025)),
        "placebo_ci_high": float(np.quantile(placebo, 0.975)),
        "positive_permutation_p": positive_p,
        "observed_percentile": float(np.mean(placebo <= observed_cumulative)),
        "observed_mean_delta_exante_vol": float(observed_mean_delta_exante_vol),
        "placebo_mean_delta_exante_vol": float(placebo_risk.mean()),
        "placebo_risk_ci_low": float(np.quantile(placebo_risk, 0.025)),
        "placebo_risk_ci_high": float(np.quantile(placebo_risk, 0.975)),
        "negative_risk_permutation_p": float(
            (1 + np.sum(placebo_risk <= observed_mean_delta_exante_vol))
            / (len(placebo_risk) + 1)
        ),
        "risk_observed_percentile": float(
            np.mean(placebo_risk <= observed_mean_delta_exante_vol)
        ),
        "invalid_row_count": int(invalid_rows),
    }


def _prepare_exante_risk_arrays(
    base: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = asset_returns.loc[:, base.columns].sort_index()
    asset_count = len(base.columns)
    covariances = np.zeros((len(base), asset_count, asset_count), dtype="float64")
    valid = np.zeros(len(base), dtype=bool)
    for position, date in enumerate(base.index):
        history = returns.loc[:date].tail(int(lookback)).to_numpy(dtype="float64")
        if len(history) < 2:
            continue
        covariances[position] = estimate_covariance(history)
        valid[position] = True
    base_values = base.to_numpy(dtype="float64")
    base_variance = np.einsum(
        "ti,tij,tj->t",
        base_values,
        covariances,
        base_values,
        optimize=True,
    )
    base_risk = np.sqrt(np.clip(base_variance, 0.0, None)) * np.sqrt(252.0)
    return covariances, base_risk, valid


def _prepare_frozen_path_arrays(
    base: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, np.ndarray]:
    price_panel = prices.loc[:, base.columns].sort_index().astype("float64")
    positions = []
    previous_ratios = []
    next_ratios = []
    for position in range(1, len(base)):
        previous_date = pd.Timestamp(base.index[position - 1])
        date = pd.Timestamp(base.index[position])
        later_dates = price_panel.index[price_panel.index > date]
        if not len(later_dates):
            continue
        next_date = pd.Timestamp(later_dates[0])
        positions.append(position)
        previous_ratios.append(
            price_panel.loc[date].to_numpy() / price_panel.loc[previous_date].to_numpy()
        )
        next_ratios.append(
            price_panel.loc[next_date].to_numpy() / price_panel.loc[date].to_numpy()
        )
    return {
        "positions": np.asarray(positions, dtype=int),
        "previous_ratios": np.asarray(previous_ratios, dtype="float64"),
        "next_ratios": np.asarray(next_ratios, dtype="float64"),
    }


def _cumulative_frozen_path_alpha_from_arrays(
    base_values: np.ndarray,
    executed_values: np.ndarray,
    prepared: dict[str, np.ndarray],
    transaction_cost_pct: float,
) -> float:
    positions = prepared["positions"]
    previous_ratios = prepared["previous_ratios"]
    next_ratios = prepared["next_ratios"]
    previous_positions = positions - 1
    base_drift = base_values[previous_positions] * previous_ratios
    base_drift /= base_drift.sum(axis=1, keepdims=True)
    exec_drift = executed_values[previous_positions] * previous_ratios
    exec_drift /= exec_drift.sum(axis=1, keepdims=True)
    base_targets = base_values[positions]
    exec_targets = executed_values[positions]
    base_turnover = np.abs(base_targets - base_drift).sum(axis=1)
    exec_turnover = np.abs(exec_targets - exec_drift).sum(axis=1)
    base_growth = (
        1.0 - float(transaction_cost_pct) * base_turnover
    ) * np.sum(base_targets * next_ratios, axis=1)
    exec_growth = (
        1.0 - float(transaction_cost_pct) * exec_turnover
    ) * np.sum(exec_targets * next_ratios, axis=1)
    delta = np.log(np.maximum(exec_growth, 1e-12)) - np.log(
        np.maximum(base_growth, 1e-12)
    )
    return float(np.exp(np.sum(delta)) - 1.0)


def cumulative_frozen_path_alpha(
    base: pd.DataFrame,
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    transaction_cost_pct: float,
) -> float:
    _require_aligned_weights(base, executed)
    prepared = _prepare_frozen_path_arrays(base, prices)
    return _cumulative_frozen_path_alpha_from_arrays(
        base.to_numpy(dtype="float64"),
        executed.to_numpy(dtype="float64"),
        prepared,
        transaction_cost_pct,
    )


def _permuted_executed_numpy(
    base_values: np.ndarray,
    tilt_values: np.ndarray,
    rng: np.random.Generator,
    *,
    max_attempts: int = 100,
) -> tuple[np.ndarray, int]:
    executed = base_values.copy()
    supports = base_values > 1e-12
    groups: dict[bytes, list[int]] = {}
    for row_index, support in enumerate(supports):
        groups.setdefault(support.tobytes(), []).append(row_index)
    for row_indices in groups.values():
        rows = np.asarray(row_indices, dtype=int)
        support_indices = np.flatnonzero(supports[rows[0]])
        values = tilt_values[np.ix_(rows, support_indices)]
        keys = rng.random(values.shape)
        order = np.argsort(keys, axis=1)
        executed[np.ix_(rows, support_indices)] += np.take_along_axis(
            values,
            order,
            axis=1,
        )
    invalid_mask = (executed < -1e-12).any(axis=1)
    initially_invalid = np.flatnonzero(invalid_mask)
    unresolved = []
    for row_index in initially_invalid:
        support_indices = np.flatnonzero(supports[row_index])
        accepted = False
        for _ in range(int(max_attempts)):
            candidate = base_values[row_index].copy()
            candidate[support_indices] += rng.permutation(
                tilt_values[row_index, support_indices]
            )
            if np.min(candidate) >= -1e-12:
                executed[row_index] = candidate
                accepted = True
                break
        if not accepted:
            executed[row_index] = base_values[row_index] + tilt_values[row_index]
            unresolved.append(row_index)
    executed = np.clip(executed, 0.0, None)
    executed /= executed.sum(axis=1, keepdims=True)
    return executed, len(unresolved)


def _load_cached_only(
    output_dir: Path,
    market: str,
    seed: int,
    scenario: str,
) -> dict[str, pd.DataFrame]:
    paths = _trace_cache_paths(output_dir, market, seed, scenario)
    bundle = _cached_trace_bundle(paths)
    if bundle is None:
        raise RuntimeError(
            f"--skip_eval requires valid cached trace bundle for {market}:{seed}:{scenario}"
        )
    return bundle


def _plot_market(
    market: str,
    daily: pd.DataFrame,
    paired: pd.DataFrame,
    figures_dir: Path,
    primary_risk_window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    market_label = {"nas": "Nasdaq-100", "sh": "CSI-300"}[market]
    risk_column = f"delta_ex_ante_vol_{primary_risk_window}d"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, column, title in [
        (axes[0], "active_share", "Active Share"),
        (axes[1], "delta_hhi", "Executed − Base HHI"),
        (axes[2], risk_column, f"Executed − Base {primary_risk_window}d ex-ante vol"),
    ]:
        axis.hist(daily[column].dropna(), bins=35, color="#2A9D8F", alpha=0.85)
        axis.axvline(0.0, color="#D1495B", linewidth=1.0)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle(f"{market_label}: Inner configuration refinement")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(
            figures_dir / f"configuration_refinement_{market}{suffix}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)

    full_wealth = np.exp(paired["full_log_return"].cumsum())
    no_inner_wealth = np.exp(paired["no_inner_log_return"].cumsum())
    cumulative_difference = paired["difference_log_return"].cumsum() * 100.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(paired.index, full_wealth, label="Full", color="#2A9D8F")
    axes[0].plot(paired.index, no_inner_wealth, label="No-Inner", color="#D1495B")
    axes[0].set_ylabel("Wealth")
    axes[0].legend()
    axes[1].plot(paired.index, cumulative_difference, color="#264653")
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    axes[1].set_ylabel("Cumulative log-return gap (pp)")
    axes[1].set_xlabel("Date")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"{market_label}: Full vs No-Inner closed loop")
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(
            figures_dir / f"closed_loop_difference_{market}{suffix}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)


def _status_from_interval(low: float, high: float, beneficial_sign: str) -> str:
    if beneficial_sign == "negative" and high < 0:
        return "SUPPORTED"
    if beneficial_sign == "positive" and low > 0:
        return "SUPPORTED"
    return "NOT SUPPORTED"


def _write_report(
    output_dir: Path,
    configuration: pd.DataFrame,
    direct: pd.DataFrame,
    closed: pd.DataFrame,
    placebo: pd.DataFrame,
) -> None:
    lines = [
        "# Inner–Outer Statistical Validation",
        "",
        "## Material Passport",
        "",
        "- Verification Status: ANALYZED",
        "- Scope: paper-selected checkpoint, one checkpoint per market",
        "- Inference: Newey-West HAC and paired circular block bootstrap",
        "- Limitation: intervals describe test-period uncertainty, not training-seed uncertainty",
        "",
        "## Claim-level results",
        "",
    ]
    for _, row in configuration.iterrows():
        risk_status = _status_from_interval(
            row["delta_ex_ante_vol_ci_low"],
            row["delta_ex_ante_vol_ci_high"],
            "negative",
        )
        lines.extend(
            [
                f"### {row['market'].upper()} configuration refinement",
                "",
                f"- Mean Active Share: {row['mean_active_share']:.4%} "
                f"(>1% on {row['active_share_gt_1pct']:.1%} of days).",
                f"- Mean ex-ante volatility change: {row['mean_delta_ex_ante_vol']:.4%}; "
                f"95% CI [{row['delta_ex_ante_vol_ci_low']:.4%}, "
                f"{row['delta_ex_ante_vol_ci_high']:.4%}] — **{risk_status}**.",
                "",
            ]
        )
    for _, row in direct.iterrows():
        alpha_status = _status_from_interval(
            row["block_ci_low_bp_day"],
            row["block_ci_high_bp_day"],
            "positive",
        )
        lines.extend(
            [
                f"### {row['market'].upper()} frozen-path direct effect",
                "",
                f"- Fair net alpha: {row['mean_net_alpha_bp_day']:.3f} bp/day; "
                f"95% CI [{row['block_ci_low_bp_day']:.3f}, "
                f"{row['block_ci_high_bp_day']:.3f}], p={row['nw_p_value']:.4f} "
                f"— **{alpha_status}**.",
                "",
            ]
        )
    for market in closed["market"].unique():
        sub = closed[closed["market"] == market].set_index("metric")
        total = sub.loc["total_return"]
        mdd = sub.loc["max_drawdown"]
        total_status = _status_from_interval(total["ci_low"], total["ci_high"], "positive")
        mdd_status = _status_from_interval(mdd["ci_low"], mdd["ci_high"], "negative")
        lines.extend(
            [
                f"### {market.upper()} closed-loop contribution",
                "",
                f"- Total-return difference: {total['difference']:.2%}; "
                f"95% CI [{total['ci_low']:.2%}, {total['ci_high']:.2%}] "
                f"— **{total_status}**.",
                f"- MDD difference: {mdd['difference']:.2%}; "
                f"95% CI [{mdd['ci_low']:.2%}, {mdd['ci_high']:.2%}] "
                f"— **{mdd_status}**.",
                "",
            ]
        )
    lines.extend(["## Risk-refinement placebo", ""])
    for _, row in placebo.iterrows():
        if int(row.get("placebo_reps", 0)) <= 0:
            continue
        lines.extend(
            [
                f"### {row['market'].upper()}",
                "",
                f"- Actual mean ex-ante volatility change: "
                f"{row['observed_mean_delta_exante_vol']:.4%}; random-tilt mean: "
                f"{row['placebo_mean_delta_exante_vol']:.4%}.",
                f"- One-sided risk-reduction permutation p="
                f"{row['negative_risk_permutation_p']:.4g}.",
                f"- Alpha-direction permutation p={row['positive_permutation_p']:.4g}; "
                "this is not used to claim standalone alpha.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "A non-significant frozen-path alpha is reported as a null result. "
            "A positive closed-loop difference is interpreted as a system-level "
            "complementary contribution, not as proof of standalone daily alpha.",
            "",
            "Placebo results are available in `tables/placebo_analysis.csv`.",
        ]
    )
    (Path(output_dir) / "INNER_OUTER_STATISTICAL_VALIDATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--full_actions_root", required=True)
    parser.add_argument(
        "--prices_root",
        default=str(Path(__file__).resolve().parents[1] / "DeepAries" / "data"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--markets", nargs="+", default=["nas", "sh"])
    parser.add_argument("--seeds", nargs="+", default=["nas:49", "sh:90"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transaction_cost_pct", type=float, default=0.00005)
    parser.add_argument("--risk_windows", nargs="+", type=int, default=[60, 20])
    parser.add_argument("--block_length", type=int, default=20)
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--placebo_reps", type=int, default=5000)
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = _parse_seed_specs(args.seeds)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    metadata_dir = output_dir / "metadata"
    figures_dir = output_dir / "figures"
    for directory in [output_dir, tables_dir, metadata_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    configuration_rows = []
    direct_rows = []
    closed_frames = []
    regime_frames = []
    placebo_rows = []
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "markets": {},
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest["code_commit"] = commit

    for market_index, market in enumerate(args.markets):
        seed = seeds[market]
        if args.skip_eval:
            full_bundle = _load_cached_only(
                output_dir, market, seed, "full_controller"
            )
            no_inner_bundle = _load_cached_only(
                output_dir, market, seed, "controller_outer"
            )
        else:
            full_bundle = ensure_closed_loop_trace(
                results_root=Path(args.results_root),
                output_dir=output_dir,
                market=market,
                seed=seed,
                scenario="full_controller",
                device=args.device,
                force_eval=args.force_eval,
            )
            no_inner_bundle = ensure_closed_loop_trace(
                results_root=Path(args.results_root),
                output_dir=output_dir,
                market=market,
                seed=seed,
                scenario="controller_outer",
                device=args.device,
                force_eval=args.force_eval,
            )

        action_path = (
            Path(args.full_actions_root)
            / f"{market}_seed{seed}_full_controller_inner_base_actions.csv"
        )
        actions = pd.read_csv(action_path) if action_path.exists() else full_bundle["actions"]
        weight_trace = parse_weight_trace(actions)
        invariants = validate_weight_invariants(weight_trace)
        prices = _load_prices(Path(args.prices_root), market, weight_trace.base.columns)
        asset_returns = prices.pct_change(fill_method=None)
        market_return = asset_returns.mean(axis=1)
        daily = configuration_shape_metrics(weight_trace.base, weight_trace.executed)
        for risk_window in args.risk_windows:
            risk = ex_ante_risk_metrics(
                weight_trace.base,
                weight_trace.executed,
                asset_returns,
                lookback=risk_window,
            ).add_suffix(f"_{risk_window}d")
            daily = daily.join(risk)
        daily, regime_cuts = attach_market_volatility_regime(
            daily,
            market_return,
            lookback=20,
        )
        daily.to_csv(tables_dir / f"{market}_configuration_daily.csv", index_label="date")
        configuration = summarize_configuration(
            daily,
            primary_risk_window=int(args.risk_windows[0]),
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=1000 + seed,
        )
        configuration_rows.append({"market": market, "seed": seed, **configuration})
        regime = (
            daily.groupby("volatility_regime", observed=False)
            .agg(
                days=("active_share", "size"),
                mean_active_share=("active_share", "mean"),
                mean_delta_hhi=("delta_hhi", "mean"),
                mean_delta_ex_ante_vol=(
                    f"delta_ex_ante_vol_{int(args.risk_windows[0])}d",
                    "mean",
                ),
            )
            .reset_index()
        )
        regime.insert(0, "market", market)
        regime_frames.append(regime)

        direct_daily = frozen_path_direct_effect(
            weight_trace.base,
            weight_trace.executed,
            prices,
            transaction_cost_pct=args.transaction_cost_pct,
        )
        direct_daily.to_csv(tables_dir / f"{market}_frozen_path_daily.csv", index_label="date")
        direct = summarize_frozen_path(
            direct_daily,
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=2000 + seed,
        )
        direct_rows.append({"market": market, "seed": seed, **direct})

        paired = align_closed_loop_returns(
            full_bundle["portfolio"],
            no_inner_bundle["portfolio"],
        )
        paired.to_csv(tables_dir / f"{market}_closed_loop_daily.csv", index_label="date")
        closed, bootstrap = summarize_closed_loop(
            paired,
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=3000 + seed,
        )
        closed.insert(0, "seed", seed)
        closed.insert(0, "market", market)
        closed_frames.append(closed)
        bootstrap.to_csv(
            tables_dir / f"{market}_closed_loop_bootstrap.csv",
            index=False,
        )

        if args.placebo_reps > 0:
            placebo = run_placebo_analysis(
                weight_trace.base,
                weight_trace.tilt,
                prices,
                asset_returns,
                transaction_cost_pct=args.transaction_cost_pct,
                risk_lookback=int(args.risk_windows[0]),
                reps=args.placebo_reps,
                seed=4000 + seed,
                observed=direct,
                observed_mean_delta_exante_vol=float(
                    configuration["mean_delta_ex_ante_vol"]
                ),
            )
        else:
            placebo = {"placebo_reps": 0}
        placebo_rows.append({"market": market, "seed": seed, **placebo})
        _plot_market(
            market,
            daily,
            paired,
            figures_dir,
            int(args.risk_windows[0]),
        )
        manifest["markets"][market] = {
            "seed": seed,
            **discover_model_identity(Path(args.results_root), market, seed),
            "action_trace": str(action_path),
            "action_trace_sha256": sha256_file(action_path),
            "price_file_sha256": sha256_file(
                Path(args.prices_root) / market / f"{market}_data.csv"
            ),
            "invariants": invariants,
            "regime_cuts": regime_cuts,
        }

    configuration_table = pd.DataFrame(configuration_rows)
    direct_table = pd.DataFrame(direct_rows)
    closed_table = pd.concat(closed_frames, ignore_index=True)
    regime_table = pd.concat(regime_frames, ignore_index=True)
    placebo_table = pd.DataFrame(placebo_rows)
    configuration_table.to_csv(tables_dir / "configuration_refinement.csv", index=False)
    direct_table.to_csv(tables_dir / "frozen_path_direct_effect.csv", index=False)
    closed_table.to_csv(tables_dir / "closed_loop_effect.csv", index=False)
    regime_table.to_csv(tables_dir / "regime_analysis.csv", index=False)
    placebo_table.to_csv(tables_dir / "placebo_analysis.csv", index=False)
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        output_dir,
        configuration_table,
        direct_table,
        closed_table,
        placebo_table,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
