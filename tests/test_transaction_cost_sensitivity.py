import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_transaction_cost_sensitivity import (
    cost_label,
    replay_executed_weight_path,
    replay_recorded_trace,
    summarize_replay,
    write_outputs,
)


def two_asset_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=["A", "B"],
    )
    prices = pd.DataFrame(
        [[100.0, 100.0], [110.0, 100.0], [121.0, 100.0]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        columns=["A", "B"],
    )
    return weights, prices


def test_replay_uses_price_drifted_previous_weights() -> None:
    weights, prices = two_asset_fixture()

    replay = replay_executed_weight_path(weights, prices, [0.0, 0.001])

    expected = abs(0.5 - 110.0 / 210.0) + abs(0.5 - 100.0 / 210.0)
    assert replay.index.tolist() == [pd.Timestamp("2020-01-02")]
    assert replay.loc[pd.Timestamp("2020-01-02"), "turnover"] == pytest.approx(
        expected
    )
    assert replay.loc[pd.Timestamp("2020-01-02"), "gross_growth"] == pytest.approx(
        0.5 * 121.0 / 110.0 + 0.5
    )


def test_higher_cost_cannot_improve_same_fixed_path() -> None:
    weights, prices = two_asset_fixture()

    replay = replay_executed_weight_path(weights, prices, [0.00005, 0.0005])

    assert (
        replay[f"net_growth_{cost_label(0.0005)}"]
        <= replay[f"net_growth_{cost_label(0.00005)}"] + 1e-15
    ).all()


def test_zero_turnover_has_identical_growth_at_every_cost() -> None:
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=["A", "B"],
    )
    prices = pd.DataFrame(
        [[100.0, 100.0], [110.0, 110.0], [121.0, 121.0]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        columns=["A", "B"],
    )

    replay = replay_executed_weight_path(weights, prices, [0.0, 0.001])

    assert replay["turnover"].iloc[0] == pytest.approx(0.0)
    assert replay[f"net_growth_{cost_label(0.0)}"].iloc[0] == pytest.approx(
        replay[f"net_growth_{cost_label(0.001)}"].iloc[0]
    )


def test_replay_rejects_misaligned_assets_and_negative_rates() -> None:
    weights, prices = two_asset_fixture()

    with pytest.raises(ValueError, match="same ordered assets"):
        replay_executed_weight_path(weights, prices[["B", "A"]], [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        replay_executed_weight_path(weights, prices, [-0.001])


def test_recorded_trace_replay_exactly_recovers_reference_path() -> None:
    baseline_rate = 0.00005
    turnover = np.array([1.2, 0.1, 0.4])
    gross_growth = np.array([1.01, 0.99, 1.02])
    recorded = pd.DataFrame(
        {
            "exec_log_return": np.log(
                gross_growth * (1.0 - baseline_rate * turnover)
            ),
            "turnover": turnover,
            "cost_rate": baseline_rate * turnover,
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )

    replay = replay_recorded_trace(
        recorded,
        cost_rates=[baseline_rate, 0.0005],
        reference_rate=baseline_rate,
    )

    reference_column = f"net_log_return_{cost_label(baseline_rate)}"
    assert replay.index.tolist() == recorded.index.tolist()
    assert replay[reference_column].to_numpy() == pytest.approx(
        recorded["exec_log_return"].to_numpy()
    )
    assert replay["gross_growth"].to_numpy() == pytest.approx(gross_growth)
    assert (
        replay[f"net_growth_{cost_label(0.0005)}"]
        <= replay[f"net_growth_{cost_label(baseline_rate)}"]
    ).all()


def test_summary_reports_reference_deltas_and_monotonic_return() -> None:
    weights, prices = two_asset_fixture()
    replay = replay_executed_weight_path(
        weights,
        prices,
        [0.00005, 0.0001, 0.0005],
    )

    summary = summarize_replay(
        replay,
        market="nas",
        seed=49,
        cost_rates=[0.00005, 0.0001, 0.0005],
        reference_rate=0.00005,
    )

    assert summary["transaction_cost_pct"].tolist() == pytest.approx(
        [0.005, 0.01, 0.05]
    )
    assert summary["total_return"].is_monotonic_decreasing
    assert summary.loc[0, "delta_total_return_pp"] == pytest.approx(0.0)
    assert np.isfinite(
        summary[["total_return", "sharpe", "max_drawdown", "calmar"]]
    ).to_numpy().all()


def test_write_outputs_creates_auditable_artifacts(tmp_path: Path) -> None:
    weights, prices = two_asset_fixture()
    replay = replay_executed_weight_path(weights, prices, [0.00005, 0.0001])
    summary = summarize_replay(
        replay,
        market="nas",
        seed=49,
        cost_rates=[0.00005, 0.0001],
        reference_rate=0.00005,
    )

    write_outputs(
        output_dir=tmp_path,
        replays={"nas": replay},
        summaries=[summary],
        manifest={"scope": "unit-test", "markets": {"nas": {"seed": 49}}},
        reference_rate=0.00005,
    )

    assert (tmp_path / "tables/transaction_cost_sensitivity.csv").is_file()
    assert (tmp_path / "tables/nas_daily_replay.csv").is_file()
    assert (tmp_path / "TRANSACTION_COST_SENSITIVITY.md").is_file()
    manifest = json.loads(
        (tmp_path / "metadata/run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scope"] == "unit-test"
