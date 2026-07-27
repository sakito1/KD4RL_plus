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
