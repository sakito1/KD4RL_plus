import json

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_inner_outer_statistical_validation import (
    attach_market_volatility_regime,
    configuration_shape_metrics,
    ex_ante_risk_metrics,
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


def test_configuration_metrics_match_hand_calculation():
    index = pd.DatetimeIndex(["2020-01-02"])
    base = pd.DataFrame([[0.6, 0.4]], index=index, columns=["A", "B"])
    executed = pd.DataFrame([[0.5, 0.5]], index=index, columns=["A", "B"])

    result = configuration_shape_metrics(base, executed)

    assert result.loc[index[0], "active_share"] == pytest.approx(0.1)
    assert result.loc[index[0], "base_hhi"] == pytest.approx(0.52)
    assert result.loc[index[0], "exec_hhi"] == pytest.approx(0.50)
    assert result.loc[index[0], "delta_effective_n"] > 0


def test_ex_ante_risk_uses_only_returns_available_by_decision_date():
    dates = pd.date_range("2020-01-02", periods=7, freq="B")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, 0.01, -0.02, 0.03, 0.02],
            "B": [-0.01, 0.02, 0.01, -0.01, 0.01, -0.02, 0.01],
        },
        index=dates,
    )
    base = pd.DataFrame([[0.7, 0.3]] * 2, index=dates[4:6], columns=returns.columns)
    executed = pd.DataFrame([[0.4, 0.6]] * 2, index=dates[4:6], columns=returns.columns)

    first = ex_ante_risk_metrics(base, executed, returns, lookback=3)
    changed = returns.copy()
    changed.loc[dates[5]:, :] = 99.0
    second = ex_ante_risk_metrics(base, executed, changed, lookback=3)

    assert first.loc[dates[4], "delta_ex_ante_vol"] == pytest.approx(
        second.loc[dates[4], "delta_ex_ante_vol"]
    )
    assert np.isfinite(first.loc[dates[4], "delta_ex_ante_vol"])


def test_ex_ante_risk_constant_returns_are_finite_zero():
    dates = pd.date_range("2020-01-02", periods=6, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    base = pd.DataFrame([[0.6, 0.4]], index=[dates[-1]], columns=returns.columns)
    executed = pd.DataFrame([[0.5, 0.5]], index=[dates[-1]], columns=returns.columns)

    result = ex_ante_risk_metrics(base, executed, returns, lookback=3)

    assert result.loc[dates[-1], "base_ex_ante_vol"] == pytest.approx(0.0)
    assert result.loc[dates[-1], "exec_ex_ante_vol"] == pytest.approx(0.0)
    assert result.loc[dates[-1], "delta_downside_vol"] == pytest.approx(0.0)


def test_market_volatility_regime_has_fixed_three_labels():
    dates = pd.date_range("2020-01-02", periods=12, freq="B")
    daily = pd.DataFrame(index=dates[3:])
    market_return = pd.Series(
        [0.001, -0.001, 0.002, -0.005, 0.006, -0.01, 0.012, -0.02, 0.021, -0.03, 0.031, -0.04],
        index=dates,
    )

    result, cuts = attach_market_volatility_regime(
        daily,
        market_return,
        lookback=3,
    )

    assert set(result["volatility_regime"].dropna().astype(str)) == {"low", "mid", "high"}
    assert cuts["lower"] < cuts["upper"]
