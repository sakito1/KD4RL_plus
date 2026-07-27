import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_inner_outer_statistical_validation import (
    align_closed_loop_returns,
    attach_market_volatility_regime,
    benjamini_hochberg,
    circular_block_bootstrap,
    configuration_shape_metrics,
    cumulative_frozen_path_alpha,
    ensure_closed_loop_trace,
    ex_ante_risk_metrics,
    frozen_path_direct_effect,
    main,
    newey_west_mean_test,
    parse_weight_trace,
    permute_tilt_within_support,
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


def test_permute_tilt_preserves_support_distribution_and_normalization():
    index = pd.date_range("2020-01-02", periods=2, freq="B")
    base = pd.DataFrame(
        [[0.5, 0.3, 0.2, 0.0], [0.4, 0.35, 0.25, 0.0]],
        index=index,
        columns=["A", "B", "C", "D"],
    )
    tilt = pd.DataFrame(
        [[0.05, -0.03, -0.02, 0.0], [-0.02, 0.04, -0.02, 0.0]],
        index=index,
        columns=base.columns,
    )

    first, invalid_first = permute_tilt_within_support(base, tilt, seed=13)
    second, invalid_second = permute_tilt_within_support(base, tilt, seed=13)

    pd.testing.assert_frame_equal(first, second)
    assert invalid_first == invalid_second == 0
    np.testing.assert_allclose(first.sum(axis=1), 1.0)
    np.testing.assert_allclose(first["D"], 0.0)
    for date in index:
        support = base.loc[date] > 0
        original = np.sort(tilt.loc[date, support].to_numpy())
        permuted = np.sort((first.loc[date, support] - base.loc[date, support]).to_numpy())
        np.testing.assert_allclose(original, permuted)


def test_benjamini_hochberg_is_monotone_and_preserves_nan():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])

    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.04)
    assert np.isnan(adjusted[3])


def test_fast_cumulative_frozen_alpha_matches_daily_counterfactual():
    dates = pd.date_range("2020-01-02", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "A": [100, 101, 103, 102, 105, 106],
            "B": [100, 99, 100, 102, 101, 103],
        },
        index=dates,
        dtype=float,
    )
    base = pd.DataFrame([[0.6, 0.4]] * 5, index=dates[:5], columns=prices.columns)
    executed = pd.DataFrame(
        [[0.55, 0.45], [0.65, 0.35], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
        index=dates[:5],
        columns=prices.columns,
    )
    direct = frozen_path_direct_effect(
        base,
        executed,
        prices,
        transaction_cost_pct=0.0005,
    )

    fast = cumulative_frozen_path_alpha(
        base,
        executed,
        prices,
        transaction_cost_pct=0.0005,
    )

    assert fast == pytest.approx(np.exp(direct["delta_net_log_return"].sum()) - 1.0)


def test_cli_skip_eval_writes_tables_and_report(tmp_path):
    dates = pd.date_range("2020-01-02", periods=10, freq="B")
    prices_root = tmp_path / "prices"
    (prices_root / "nas").mkdir(parents=True)
    price_rows = []
    for asset, levels in {
        "A": np.linspace(100.0, 109.0, len(dates)),
        "B": np.linspace(100.0, 104.0, len(dates)),
    }.items():
        price_rows.extend(
            {"date": date, "tic": asset, "adjclose": level}
            for date, level in zip(dates, levels)
        )
    pd.DataFrame(price_rows).to_csv(prices_root / "nas" / "nas_data.csv", index=False)

    actions_root = tmp_path / "actions"
    actions_root.mkdir()
    actions = synthetic_actions(
        base=[[0.6, 0.4]] * 8,
        executed=[[0.55, 0.45]] * 8,
        names=["A", "B"],
    )
    actions["date"] = dates[:8].astype(str)
    actions.to_csv(
        actions_root / "nas_seed49_full_controller_inner_base_actions.csv",
        index=False,
    )

    output = tmp_path / "output"
    traces = output / "traces"
    traces.mkdir(parents=True)
    for scenario, returns, scenario_actions in [
        ("full_controller", np.full(8, 0.002), actions),
        (
            "controller_outer",
            np.full(8, 0.001),
            synthetic_actions(
                base=[[0.6, 0.4]] * 8,
                executed=[[0.6, 0.4]] * 8,
                names=["A", "B"],
            ),
        ),
    ]:
        scenario_actions = scenario_actions.copy()
        scenario_actions["date"] = dates[:8].astype(str)
        prefix = f"nas_seed49_{scenario}"
        pd.DataFrame(
            {
                "date": dates[:8].astype(str),
                "portfolio_value": 1000.0 * np.exp(np.cumsum(returns)),
                "daily_log_return": returns,
            }
        ).to_csv(traces / f"{prefix}_portfolio.csv", index=False)
        scenario_actions.to_csv(traces / f"{prefix}_actions.csv", index=False)
        pd.DataFrame({"date": [str(dates[0].date())], "is_switch": [1]}).to_csv(
            traces / f"{prefix}_switch_events.csv",
            index=False,
        )

    exit_code = main(
        [
            "--results_root",
            str(tmp_path / "unused"),
            "--full_actions_root",
            str(actions_root),
            "--prices_root",
            str(prices_root),
            "--output_dir",
            str(output),
            "--markets",
            "nas",
            "--seeds",
            "nas:49",
            "--risk_windows",
            "3",
            "--block_length",
            "2",
            "--bootstrap_reps",
            "20",
            "--placebo_reps",
            "5",
            "--skip_eval",
        ]
    )

    assert exit_code == 0
    assert (output / "tables" / "configuration_refinement.csv").exists()
    assert (output / "tables" / "frozen_path_direct_effect.csv").exists()
    assert (output / "tables" / "closed_loop_effect.csv").exists()
    assert (output / "INNER_OUTER_STATISTICAL_VALIDATION.md").exists()
    placebo = pd.read_csv(output / "tables" / "placebo_analysis.csv")
    assert "negative_risk_permutation_p" in placebo.columns


def test_script_entrypoint_adds_project_root_to_sys_path():
    script = (
        Path(__file__).resolve().parents[1]
        / "paper_experiments"
        / "analyze_inner_outer_statistical_validation.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy,sys;"
                f"runpy.run_path({str(script)!r}, run_name='not_main');"
                f"assert {str(script.parents[1])!r} in sys.path"
            ),
        ],
        cwd=script.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
