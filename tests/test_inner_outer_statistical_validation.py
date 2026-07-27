import json

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_inner_outer_statistical_validation import (
    parse_weight_trace,
    validate_weight_invariants,
)


def synthetic_actions(base, executed, *, names=None):
    names = names or [chr(ord("A") + i) for i in range(len(base[0]))]
    tilt = (np.asarray(executed, dtype=float) - np.asarray(base, dtype=float)).tolist()
    dates = pd.date_range("2020-01-02", periods=len(base), freq="B")
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "asset_names_json": [json.dumps(names)] * len(base),
            "base_weights_json": [json.dumps(row) for row in base],
            "exec_weights_json": [json.dumps(row) for row in executed],
            "inner_tilt_json": [json.dumps(row) for row in tilt],
        }
    )


def test_parse_weight_trace_and_validate_support():
    actions = synthetic_actions(
        base=[[0.6, 0.4, 0.0], [0.5, 0.5, 0.0]],
        executed=[[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]],
    )

    parsed = parse_weight_trace(actions)
    validation = validate_weight_invariants(parsed)

    assert list(parsed.base.columns) == ["A", "B", "C"]
    assert validation["max_abs_tilt_identity_error"] < 1e-12
    assert validation["max_abs_weight_sum_error"] < 1e-12
    assert validation["support_violation_count"] == 0


def test_validate_weight_invariants_detects_support_violation():
    actions = synthetic_actions(
        base=[[0.6, 0.4, 0.0]],
        executed=[[0.5, 0.4, 0.1]],
    )

    validation = validate_weight_invariants(parse_weight_trace(actions))

    assert validation["support_violation_count"] == 1


def test_parse_weight_trace_rejects_duplicate_dates():
    actions = synthetic_actions(
        base=[[0.6, 0.4], [0.5, 0.5]],
        executed=[[0.5, 0.5], [0.4, 0.6]],
    )
    actions.loc[1, "date"] = actions.loc[0, "date"]

    with pytest.raises(ValueError, match="duplicate"):
        parse_weight_trace(actions)
