import json

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_inner_outer_statistical_validation import (
    align_closed_loop_returns,
    attach_market_volatility_regime,
    circular_block_bootstrap,
    configuration_shape_metrics,
    ensure_closed_loop_trace,
    ex_ante_risk_metrics,
    frozen_path_direct_effect,
    newey_west_mean_test,
    parse_weight_trace,
    portfolio_path_metrics,
    summarize_closed_loop,
    summarize_frozen_path,
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


def test_frozen_path_direct_effect_uses_fair_path_specific_costs():
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 121.0], "B": [100.0, 100.0, 100.0, 110.0]},
        index=dates,
    )
    base = pd.DataFrame([[0.5, 0.5]] * 3, index=dates[:3], columns=prices.columns)
    executed = pd.DataFrame(
        [[0.5, 0.5], [0.7, 0.3], [0.7, 0.3]],
        index=dates[:3],
        columns=prices.columns,
    )
    cost = 0.001

    result = frozen_path_direct_effect(
        base,
        executed,
        prices,
        transaction_cost_pct=cost,
    )

    date = dates[1]
    prior_ratio = np.array([1.1, 1.0])
    prior_drift = np.array([0.5, 0.5]) * prior_ratio
    prior_drift /= prior_drift.sum()
    expected_exec_turnover = np.abs(np.array([0.7, 0.3]) - prior_drift).sum()
    expected_base_turnover = np.abs(np.array([0.5, 0.5]) - prior_drift).sum()
    next_ratio = np.array([1.1, 1.0])
    expected_exec_net = np.log(
        (1 - cost * expected_exec_turnover) * np.dot([0.7, 0.3], next_ratio)
    )
    expected_base_net = np.log(
        (1 - cost * expected_base_turnover) * np.dot([0.5, 0.5], next_ratio)
    )

    assert result.loc[date, "exec_turnover"] == pytest.approx(expected_exec_turnover)
    assert result.loc[date, "base_turnover"] == pytest.approx(expected_base_turnover)
    assert result.loc[date, "delta_net_log_return"] == pytest.approx(
        expected_exec_net - expected_base_net
    )


def test_frozen_path_equal_targets_have_zero_paired_effect():
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 101, 102, 103, 104], "B": [100, 99, 100, 98, 101]},
        index=dates,
        dtype=float,
    )
    weights = pd.DataFrame([[0.6, 0.4]] * 4, index=dates[:4], columns=prices.columns)

    result = frozen_path_direct_effect(
        weights,
        weights,
        prices,
        transaction_cost_pct=0.001,
    )

    assert np.max(np.abs(result["delta_net_log_return"])) < 1e-12
    assert np.max(np.abs(result["exec_turnover"] - result["base_turnover"])) < 1e-12


def test_circular_block_bootstrap_is_deterministic_and_paired():
    x = np.arange(12, dtype=float)
    y = x + 10.0

    first = circular_block_bootstrap(
        [x, y],
        lambda a, b: np.mean(b - a),
        block_length=3,
        reps=100,
        seed=7,
    )
    second = circular_block_bootstrap(
        [x, y],
        lambda a, b: np.mean(b - a),
        block_length=3,
        reps=100,
        seed=7,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, 10.0)


def test_newey_west_and_frozen_summary_report_effect_size():
    direct = pd.DataFrame(
        {
            "exec_net_log_return": [0.01, -0.005, 0.008, 0.002, -0.001],
            "base_net_log_return": [0.008, -0.006, 0.006, 0.001, -0.002],
            "delta_net_log_return": [0.002, 0.001, 0.002, 0.001, 0.001],
            "exec_turnover": [0.1] * 5,
            "base_turnover": [0.05] * 5,
            "exec_cost_rate": [0.0001] * 5,
            "base_cost_rate": [0.00005] * 5,
            "market_simple_return": [0.01, -0.02, 0.015, -0.03, 0.005],
        }
    )

    test = newey_west_mean_test(direct["delta_net_log_return"], maxlags=2)
    summary = summarize_frozen_path(
        direct,
        block_length=2,
        bootstrap_reps=100,
        seed=9,
    )

    assert test["mean"] == pytest.approx(0.0014)
    assert summary["mean_net_alpha_bp_day"] == pytest.approx(14.0)
    assert summary["positive_alpha_ratio"] == pytest.approx(1.0)
    assert summary["block_ci_low_bp_day"] > 0


def test_ensure_closed_loop_trace_reuses_valid_cache(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    prefix = "nas_seed49_full_controller"
    portfolio = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "portfolio_value": [100.0, 101.0],
            "daily_log_return": [0.0, np.log(1.01)],
        }
    )
    actions = synthetic_actions(
        base=[[0.6, 0.4], [0.6, 0.4]],
        executed=[[0.5, 0.5], [0.5, 0.5]],
    )
    switches = pd.DataFrame({"date": ["2020-01-02"], "is_switch": [1]})
    portfolio.to_csv(traces / f"{prefix}_portfolio.csv", index=False)
    actions.to_csv(traces / f"{prefix}_actions.csv", index=False)
    switches.to_csv(traces / f"{prefix}_switch_events.csv", index=False)

    result = ensure_closed_loop_trace(
        results_root=tmp_path / "unused",
        output_dir=tmp_path,
        market="nas",
        seed=49,
        scenario="full_controller",
        device="cpu",
        force_eval=False,
    )

    pd.testing.assert_frame_equal(result["portfolio"], portfolio)
    assert len(result["actions"]) == 2


def test_align_closed_loop_returns_inner_joins_dates():
    full = pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-03"], "daily_log_return": [0.01, 0.02]}
    )
    no_inner = pd.DataFrame(
        {"date": ["2020-01-03", "2020-01-06"], "daily_log_return": [0.005, 0.01]}
    )

    result = align_closed_loop_returns(full, no_inner)

    assert list(result.index) == [pd.Timestamp("2020-01-03")]
    assert result.iloc[0]["difference_log_return"] == pytest.approx(0.015)


def test_portfolio_path_metrics_and_closed_loop_summary():
    full = np.array([0.01, 0.005, -0.002, 0.008, 0.003])
    no_inner = np.array([0.005, 0.002, -0.003, 0.004, 0.001])
    paired = pd.DataFrame(
        {
            "full_log_return": full,
            "no_inner_log_return": no_inner,
            "difference_log_return": full - no_inner,
        }
    )

    metrics = portfolio_path_metrics(full)
    summary, bootstrap = summarize_closed_loop(
        paired,
        block_length=2,
        bootstrap_reps=100,
        seed=11,
    )

    assert metrics["total_return"] == pytest.approx(np.exp(full.sum()) - 1)
    total = summary.loc[summary["metric"] == "total_return"].iloc[0]
    assert total["difference"] > 0
    assert total["ci_low"] > 0
    assert len(bootstrap) == 100
