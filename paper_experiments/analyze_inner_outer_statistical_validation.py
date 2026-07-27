#!/usr/bin/env python3
"""Validate how the Inner actor refines and complements the Outer portfolio."""

from __future__ import annotations

import json
from dataclasses import dataclass

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
