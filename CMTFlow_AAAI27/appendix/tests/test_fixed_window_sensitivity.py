import csv
import math
import sys
from pathlib import Path

import pytest


APPENDIX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPENDIX_ROOT / "code"))

from analyze_fixed_window_sensitivity import (  # noqa: E402
    path_metrics,
    reprice_growth,
)


def test_reprice_growth_recovers_gross_path_before_new_fee():
    original_growth = 1.01
    turnover = 0.4
    original_fee = 0.00005
    new_fee = 0.00010
    expected = original_growth / (1.0 - turnover * original_fee)
    expected *= 1.0 - turnover * new_fee

    assert reprice_growth(
        original_growth,
        turnover,
        original_fee,
        new_fee,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("growth", "turnover", "original_fee", "new_fee"),
    [
        (0.0, 0.1, 0.00005, 0.00010),
        (-1.0, 0.1, 0.00005, 0.00010),
        (1.0, -0.1, 0.00005, 0.00010),
        (1.0, 10.0, 0.1, 0.00010),
        (1.0, 10.0, 0.00005, 0.1),
    ],
)
def test_reprice_growth_rejects_invalid_path_values(
    growth,
    turnover,
    original_fee,
    new_fee,
):
    with pytest.raises(ValueError):
        reprice_growth(growth, turnover, original_fee, new_fee)


def test_path_metrics_uses_paper_daily_metric_definitions():
    growth = [1.02, 0.99, 1.01, 1.03]
    metrics = path_metrics(growth)
    returns = [value - 1.0 for value in growth]
    expected_wealth = math.prod(growth)

    assert metrics["total_return"] == pytest.approx(expected_wealth - 1.0)
    assert metrics["sharpe"] > 0.0
    assert metrics["max_drawdown"] == pytest.approx(0.01)
    assert metrics["calmar"] > 0.0


def test_public_daily_replay_columns_are_minimal_when_inputs_exist():
    input_dir = APPENDIX_ROOT / "inputs/fixed_window"
    for market in ("nasdaq100", "csi300"):
        path = input_dir / f"daily_replay_{market}.csv"
        if not path.exists():
            pytest.skip("public dense replay has not been generated yet")
        with path.open(newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames
        assert fieldnames is not None
        assert fieldnames[0] == "date"
        assert fieldnames[1:] == [
            column
            for window in range(1, 61)
            for column in (
                f"net_growth_w{window:02d}",
                f"turnover_w{window:02d}",
            )
        ]
