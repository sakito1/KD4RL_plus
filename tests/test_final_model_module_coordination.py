import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_final_model_module_coordination import (
    build_base_transition_events,
    build_controller_events,
    build_daily_events,
    circular_block_bootstrap_ci,
    parse_vector,
    summarize_controller_decisions,
    summarize_holding_age,
    summarize_probability_bins,
    summarize_volatility_quartiles,
    write_outputs,
)


def _actions(n=12):
    tilts = [[0.10, -0.10]] + [[0.02, -0.02] for _ in range(n - 1)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "step": np.arange(n),
            "decision_type": ["forced_switch"] + ["free_decision"] * (n - 1),
            "is_switch": [1] + [0] * (n - 1),
            "is_free_switch": [0] * n,
            "is_forced_switch": [1] + [0] * (n - 1),
            "exit_prob": [np.nan] + [0.25] * (n - 1),
            "duration_before_decision": np.arange(n),
            "controller_switch_advantage": [np.nan] + [0.01] * (n - 1),
            "hold_future_return_20": [np.nan] + [0.01] * (n - 1),
            "switch_future_return_20": [np.nan] + [0.02] * (n - 1),
            "switch_advantage_20": [np.nan] + [0.01] * (n - 1),
            "base_log_return": np.arange(n, dtype=float) / 100.0,
            "inner_tilt_json": [json.dumps(value) for value in tilts],
            "base_weights_json": [json.dumps([0.5, 0.5])] * n,
        }
    )


def _portfolio(n=12):
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "holding_duration": np.arange(n),
        }
    )


def test_parse_vector_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        parse_vector("[0.1, NaN]")


def test_build_daily_events_computes_refinement_l1_and_age_bucket():
    events = build_daily_events(
        _actions(), _portfolio(), market="Nasdaq-100", seed=49
    )

    assert events.loc[0, "refinement_l1"] == pytest.approx(0.20)
    assert events.loc[0, "holding_age_bucket"] == "0-1"
    assert events.loc[5, "holding_age_bucket"] == "2-5"
    assert events.loc[10, "holding_age_bucket"] == "6-10"


def test_recent_volatility_uses_only_ten_returns_before_current_day():
    actions = _actions()
    events = build_daily_events(
        actions, _portfolio(), market="Nasdaq-100", seed=49
    )

    assert events.loc[:9, "recent_volatility_10"].isna().all()
    expected = actions.loc[:9, "base_log_return"].std(ddof=1)
    assert events.loc[10, "recent_volatility_10"] == pytest.approx(expected)

    changed = actions.copy()
    changed.loc[10, "base_log_return"] = 999.0
    changed_events = build_daily_events(
        changed, _portfolio(), market="Nasdaq-100", seed=49
    )
    assert changed_events.loc[10, "recent_volatility_10"] == pytest.approx(expected)


def test_controller_events_orient_advantage_to_the_chosen_action():
    actions = _actions(3)
    actions.loc[1:, "is_free_switch"] = [1, 0]
    actions.loc[1:, "is_switch"] = [1, 0]
    actions.loc[1:, "switch_advantage_20"] = [0.03, 0.03]

    events = build_controller_events(actions, market="Nasdaq-100", seed=49)

    assert events.loc[0, "chosen_action_advantage_20"] == pytest.approx(0.03)
    assert events.loc[1, "chosen_action_advantage_20"] == pytest.approx(-0.03)


def test_base_transition_events_measure_support_replacement():
    actions = _actions(2)
    actions.loc[0, "base_weights_json"] = json.dumps([0.4, 0.3, 0.3, 0.0])
    actions.loc[1, "base_weights_json"] = json.dumps([0.0, 0.3, 0.3, 0.4])
    actions.loc[1, "is_free_switch"] = 1
    actions.loc[1, "is_switch"] = 1

    events = build_base_transition_events(
        actions, market="Nasdaq-100", seed=49
    )

    row = events.iloc[0]
    assert row["retained_assets"] == 2
    assert row["added_assets"] == 1
    assert row["removed_assets"] == 1
    assert row["support_jaccard"] == pytest.approx(0.5)
    assert row["weight_overlap"] == pytest.approx(0.6)
    assert row["weight_l1_distance"] == pytest.approx(0.8)


def test_holding_age_summary_preserves_fixed_bucket_order():
    daily = pd.DataFrame(
        {
            "market": ["Nasdaq-100"] * 5,
            "seed": [49] * 5,
            "holding_age_bucket": ["21+", "0-1", "11-20", "6-10", "2-5"],
            "refinement_l1": [0.5, 0.1, 0.4, 0.3, 0.2],
        }
    )

    summary = summarize_holding_age(daily)

    assert summary["holding_age_bucket"].tolist() == [
        "0-1",
        "2-5",
        "6-10",
        "11-20",
        "21+",
    ]
    assert summary["mean_refinement_l1"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )


def test_volatility_summary_assigns_four_rank_based_quartiles():
    daily = pd.DataFrame(
        {
            "market": ["CSI-300"] * 8,
            "seed": [90] * 8,
            "recent_volatility_10": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
            "refinement_l1": np.arange(1, 9, dtype=float),
        }
    )

    summary = summarize_volatility_quartiles(daily)

    assert summary["volatility_quartile"].tolist() == [
        "vol_q1",
        "vol_q2",
        "vol_q3",
        "vol_q4",
    ]
    assert summary["count"].tolist() == [2, 2, 2, 2]


def test_probability_summary_assigns_stable_quintiles_with_ties():
    events = pd.DataFrame(
        {
            "market": ["Nasdaq-100"] * 10,
            "seed": [49] * 10,
            "exit_prob": [0.2] * 5 + [0.8] * 5,
            "is_switch": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            "switch_advantage_20": np.linspace(-0.05, 0.05, 10),
        }
    )

    summary = summarize_probability_bins(events)

    assert summary["probability_quintile"].tolist() == [
        "prob_q1",
        "prob_q2",
        "prob_q3",
        "prob_q4",
        "prob_q5",
    ]
    assert summary["count"].sum() == 10


def test_circular_block_bootstrap_is_deterministic():
    values = np.arange(1.0, 9.0)

    result_a = circular_block_bootstrap_ci(
        values,
        lambda sample: float(np.mean(sample)),
        block_length=3,
        reps=200,
        seed=7,
    )
    result_b = circular_block_bootstrap_ci(
        values,
        lambda sample: float(np.mean(sample)),
        block_length=3,
        reps=200,
        seed=7,
    )

    assert result_a == result_b
    assert result_a["ci_low"] <= result_a["estimate"] <= result_a["ci_high"]


def test_write_outputs_creates_eight_csvs_and_chinese_report(tmp_path: Path):
    tables = {
        "daily_module_events": pd.DataFrame(
            {"market": ["Nasdaq-100"], "seed": [49], "refinement_l1": [0.02]}
        ),
        "controller_decision_summary": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "seed": [49],
                "total_days": [100],
                "free_decisions": [90],
                "free_switches": [10],
                "free_switch_ratio": [0.1],
                "forced_switches": [2],
                "chosen_advantage_mean": [0.001],
                "chosen_advantage_ci_low": [-0.001],
                "chosen_advantage_ci_high": [0.003],
                "chosen_advantage_positive_ratio": [0.52],
            }
        ),
        "controller_probability_bins": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "probability_quintile": ["prob_q1"],
                "mean_exit_prob": [0.1],
                "free_switch_ratio": [0.0],
                "mean_switch_advantage_20": [0.001],
                "positive_switch_advantage_ratio": [0.5],
            }
        ),
        "base_transition_events": pd.DataFrame(
            {"market": ["Nasdaq-100"], "added_assets": [2]}
        ),
        "base_transition_summary": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "free_switches": [10],
                "added_assets_mean": [2.0],
                "support_jaccard_mean": [0.67],
                "weight_overlap_mean": [0.75],
                "weight_l1_distance_mean": [0.5],
            }
        ),
        "trader_holding_age_summary": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "holding_age_bucket": ["0-1"],
                "count": [10],
                "mean_refinement_l1": [0.01],
                "median_refinement_l1": [0.01],
            }
        ),
        "trader_volatility_summary": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "volatility_quartile": ["vol_q1"],
                "count": [10],
                "mean_refinement_l1": [0.01],
                "median_refinement_l1": [0.01],
            }
        ),
        "trader_correlation_summary": pd.DataFrame(
            {
                "market": ["Nasdaq-100"],
                "age_spearman": [0.5],
                "age_spearman_ci_low": [0.3],
                "age_spearman_ci_high": [0.7],
                "volatility_spearman": [0.2],
                "volatility_spearman_ci_low": [0.0],
                "volatility_spearman_ci_high": [0.4],
                "high_minus_low_vol_refinement": [0.01],
                "high_minus_low_vol_refinement_ci_low": [0.0],
                "high_minus_low_vol_refinement_ci_high": [0.02],
            }
        ),
    }

    write_outputs(tmp_path, tables)

    expected = {f"{name}.csv" for name in tables}
    expected.add("最终模型模块协同统计.md")
    assert {path.name for path in tmp_path.iterdir()} == expected
    report = (tmp_path / "最终模型模块协同统计.md").read_text(encoding="utf-8")
    assert "# 最终模型模块协同统计" in report
    assert "## Controller 决策行为" in report
    assert "## Manager--Controller 底仓转换" in report
    assert "## Trader 跨时间尺度修正" in report
    assert "## 可安全用于论文的解释" in report
    assert "## 局限" in report


def test_controller_bootstrap_preserves_full_trading_calendar():
    actions = _actions(12)
    actions["decision_type"] = "forced_switch"
    actions["is_switch"] = 0
    actions["is_free_switch"] = 0
    actions.loc[[1, 10], "decision_type"] = "free_decision"
    actions.loc[[1, 10], "switch_advantage_20"] = [0.10, -0.05]
    controller = build_controller_events(actions, market="Nasdaq-100", seed=49)
    dense = np.full(len(actions), np.nan)
    dense[[1, 10]] = [-0.10, 0.05]
    expected = circular_block_bootstrap_ci(
        dense,
        lambda sample: (
            float(np.mean(sample[np.isfinite(sample)]))
            if np.isfinite(sample).any()
            else float("nan")
        ),
        block_length=3,
        reps=300,
        seed=11,
    )

    summary = summarize_controller_decisions(
        actions,
        controller,
        block_length=3,
        bootstrap_reps=300,
        bootstrap_seed=11,
    ).iloc[0]

    assert summary["chosen_advantage_ci_low"] == pytest.approx(expected["ci_low"])
    assert summary["chosen_advantage_ci_high"] == pytest.approx(expected["ci_high"])
