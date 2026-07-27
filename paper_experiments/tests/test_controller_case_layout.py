import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paper_experiments import run_paper_experiments_final as figures


def test_controller_case_uses_two_equal_side_by_side_panels_and_shared_legend(
    monkeypatch, tmp_path
) -> None:
    curve = [1.0 + 0.001 * day for day in range(31)]
    case = pd.Series(
        {
            "step": 20,
            "date": "2024-01-02",
            "exit_prob": 0.72,
            "hold_curve_30": json.dumps(curve),
            "switch_curve_30": json.dumps([value + 0.0005 * day for day, value in enumerate(curve)]),
        }
    )
    actions = pd.DataFrame(
        {
            "step": range(10, 31),
            "exit_prob": [0.72] * 21,
            "switch_advantage_20": [0.01] * 21,
            "switch_advantage_30": [0.02] * 21,
            "is_switch": [0] * 10 + [1] + [0] * 10,
            "is_free_switch": [0] * 10 + [1] + [0] * 10,
        }
    )
    captured = {}
    monkeypatch.setattr(figures, "save_figure", lambda fig, path: captured.setdefault("fig", fig))

    figures.plot_controller_case(
        "nas",
        1,
        case,
        pd.DataFrame({"step": range(61)}),
        actions,
        tmp_path,
    )

    fig = captured["fig"]
    try:
        assert len(fig.axes) == 2
        left, right = fig.axes
        assert left.get_position().x0 < right.get_position().x0
        assert left.get_position().y0 == pytest.approx(right.get_position().y0)
        assert left.get_position().width == pytest.approx(right.get_position().width)
        assert len(fig.legends) == 1
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert labels == [
            "No-controller keep",
            "Controller switch",
            "Switch advantage area",
            "Avoided drawdown",
        ]
    finally:
        plt.close(fig)
