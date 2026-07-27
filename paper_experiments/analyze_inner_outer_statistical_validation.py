#!/usr/bin/env python3
"""Validate how the Inner actor refines and complements the Outer portfolio."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


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
