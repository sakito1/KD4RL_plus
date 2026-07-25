import json

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_population_level_mechanisms import (
    _scaled_interval,
    build_controller_population_events,
    build_holding_segments,
    build_inner_daily_statistics,
    build_tilt_quantile_events,
)


def test_scaled_interval_uses_requested_paper_units():
    low, estimate, high = _scaled_interval(-0.001, 0.002, 0.003, scale=100)

    assert (low, estimate, high) == pytest.approx((-0.1, 0.2, 0.3))


def test_controller_events_require_full_horizon_and_orient_mdd_improvement():
    actions = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "decision_type": ["free_decision"] * 3,
            "exit_prob": [0.1, 0.5, 0.9],
            "hold_curve_20": [json.dumps([1.0] * 21), json.dumps([1.0] * 20), json.dumps([1.0] * 21)],
            "switch_curve_20": [json.dumps([1.0] * 21)] * 3,
            "hold_future_return_20": [0.01, 0.02, 0.03],
            "switch_future_return_20": [0.03, 0.05, 0.02],
            "hold_future_mdd_20": [0.08, 0.07, 0.04],
            "switch_future_mdd_20": [0.05, 0.02, 0.06],
        }
    )

    result = build_controller_population_events(actions, market="Nasdaq-100", seed=49)

    assert result["date"].tolist() == ["2020-01-01", "2020-01-03"]
    assert result["return_uplift_20"].tolist() == pytest.approx([0.02, -0.01])
    assert result["drawdown_improvement_20"].tolist() == pytest.approx([0.03, -0.02])


def _inner_actions():
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D").astype(str),
            "asset_names_json": [json.dumps(["A", "B", "C", "D"])] * 3,
            "base_weights_json": [json.dumps([0.25, 0.25, 0.25, 0.25])] * 3,
            "inner_tilt_json": [json.dumps([-0.2, -0.1, 0.1, 0.2])] * 3,
        }
    )


def test_inner_daily_statistics_use_t_to_t_plus_h_active_support_returns():
    actions = _inner_actions()
    prices = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [1.0, 2.0, 2.0, 2.0],
            "C": [1.0, 3.0, 3.0, 3.0],
            "D": [1.0, 4.0, 4.0, 4.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = build_inner_daily_statistics(actions, prices, market="X", seed=1, horizons=(1,))

    first = result.iloc[0]
    assert first["date"] == "2020-01-01"
    assert first["asset_count"] == 4
    assert first["pearson_ic"] > 0.9
    assert first["rank_ic"] == pytest.approx(1.0)
    assert len(result) == 1


def test_tilt_quantiles_are_equal_count_and_preserve_date_balance():
    actions = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "asset_names_json": [json.dumps([f"A{i}" for i in range(10)])],
            "base_weights_json": [json.dumps([0.1] * 10)],
            "inner_tilt_json": [json.dumps(np.linspace(-0.09, 0.09, 10).tolist())],
        }
    )
    prices = pd.DataFrame(
        [np.ones(10), np.exp(np.linspace(-0.1, 0.1, 10))],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        columns=[f"A{i}" for i in range(10)],
    )

    result = build_tilt_quantile_events(actions, prices, market="X", seed=1, horizons=(1,))

    assert result.groupby("tilt_quantile").size().tolist() == [2, 2, 2, 2, 2]
    means = result.groupby("tilt_quantile")["future_relative_return"].mean()
    assert means.loc["tilt_q5"] > means.loc["tilt_q1"]


def test_holding_segments_use_next_revision_as_termination_and_drop_censored_from_completed():
    actions = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D").astype(str),
            "step": np.arange(8),
            "is_switch": [1, 0, 0, 1, 0, 0, 1, 0],
            "is_free_switch": [0, 0, 0, 1, 0, 0, 0, 0],
            "is_forced_switch": [1, 0, 0, 0, 0, 0, 1, 0],
        }
    )

    result = build_holding_segments(actions, market="X", seed=1)

    assert result["duration"].tolist() == [3, 3, 2]
    assert result["termination_type"].tolist() == ["free", "forced", "censored"]
    assert result["is_completed"].tolist() == [1, 1, 0]
