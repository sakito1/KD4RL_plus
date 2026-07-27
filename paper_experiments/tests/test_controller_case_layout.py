import json

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paper_experiments import run_paper_experiments_final as figures


def controller_case_fixture(date: str, slope: float = 0.001):
    curve = [1.0 + slope * day for day in range(31)]
    case = pd.Series(
        {
            "step": 20,
            "date": date,
            "exit_prob": 0.72,
            "hold_curve_30": json.dumps(curve),
            "switch_curve_30": json.dumps(
                [value + 0.0005 * day for day, value in enumerate(curve)]
            ),
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
    portfolio = pd.DataFrame(
        {
            "step": range(61),
            "date": pd.bdate_range(pd.Timestamp(date) - pd.offsets.BDay(20), periods=61),
        }
    )
    return case, portfolio, actions


def test_controller_case_uses_two_equal_side_by_side_panels_and_shared_legend(
    monkeypatch, tmp_path
) -> None:
    curve = [1.0 + 0.001 * day for day in range(31)]
    case = pd.Series(
        {
            "step": 20,
            "date": "2021-07-07",
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
    portfolio = pd.DataFrame(
        {
            "step": range(61),
            "date": pd.bdate_range("2021-06-09", periods=61),
        }
    )

    figures.plot_controller_case(
        "nas",
        1,
        case,
        portfolio,
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
            "Controller reconstruct",
            "Switch advantage area",
            "Avoided drawdown",
        ]
        assert len(fig.texts) == 0
        expected_panel_titles = [
            "A. Future return after the switch decision",
            "B. Future drawdown under the same frozen window",
        ]
        for axis, panel_title in zip(fig.axes, expected_panel_titles):
            assert axis.get_title() == ""
            assert axis.get_xlim() == pytest.approx((1.0, 30.0))
            assert list(axis.get_xticks()) == pytest.approx([1, 5, 10, 15, 20, 25, 30])
            assert axis.get_xlabel() == "2021-07-08—2021-08-18"
            assert panel_title in [text.get_text() for text in axis.texts]
    finally:
        plt.close(fig)


def test_controller_case_combinations_are_cartesian_product() -> None:
    sh_cases = [(1, "sh-1"), (2, "sh-2")]
    nas_cases = [(1, "nas-1"), (2, "nas-2")]

    pairs = figures.controller_case_combinations(sh_cases, nas_cases)

    assert [(sh_id, nas_id) for sh_id, _, nas_id, _ in pairs] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_combined_controller_case_uses_two_by_two_layout(
    monkeypatch, tmp_path
) -> None:
    sh_case, sh_portfolio, sh_actions = controller_case_fixture("2021-07-07")
    nas_case, nas_portfolio, nas_actions = controller_case_fixture(
        "2021-08-02", slope=0.0008
    )
    captured = {}

    def capture(fig, path):
        captured["fig"] = fig
        captured["path"] = path

    monkeypatch.setattr(figures, "save_figure", capture)

    figures.plot_combined_controller_case(
        sh_case_id=1,
        sh_case=sh_case,
        sh_portfolio=sh_portfolio,
        sh_actions=sh_actions,
        nas_case_id=2,
        nas_case=nas_case,
        nas_portfolio=nas_portfolio,
        nas_actions=nas_actions,
        out_dir=tmp_path,
    )

    fig = captured["fig"]
    try:
        assert len(fig.axes) == 4
        top_left, top_right, bottom_left, bottom_right = fig.axes
        assert top_left.get_position().width == pytest.approx(
            top_right.get_position().width
        )
        assert top_left.get_position().width == pytest.approx(
            bottom_left.get_position().width
        )
        assert top_left.get_position().height == pytest.approx(
            bottom_left.get_position().height
        )
        assert top_left.get_position().y0 > bottom_left.get_position().y0
        horizontal_gap = top_right.get_position().x0 - top_left.get_position().x1
        vertical_gap = top_left.get_position().y0 - bottom_left.get_position().y1
        assert horizontal_gap < 0.15 * top_left.get_position().width
        assert vertical_gap < 0.25 * top_left.get_position().height
        figure_labels = [text.get_text() for text in fig.texts]
        assert figure_labels.count("CSI-300") == 1
        assert figure_labels.count("Nasdaq-100") == 1
        assert figure_labels.count(
            "A. Future return after the switch decision"
        ) == 1
        assert figure_labels.count(
            "B. Future drawdown under the same frozen window"
        ) == 1
        assert len(fig.legends) == 1
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend_labels == [
            "No-controller keep",
            "Controller reconstruct",
            "Switch advantage area",
            "Avoided drawdown",
        ]
        assert captured["path"].name == "controller_case_combined_sh01_nas02"
        row_labels = {
            text.get_text(): text
            for text in fig.texts
            if text.get_text() in {"CSI-300", "Nasdaq-100"}
        }
        assert row_labels["CSI-300"].get_position()[1] == pytest.approx(
            (top_left.get_position().y0 + top_left.get_position().y1) / 2 + 0.015
        )
        assert row_labels["Nasdaq-100"].get_position()[1] == pytest.approx(
            (bottom_left.get_position().y0 + bottom_left.get_position().y1) / 2
            + 0.015
        )
        for axis in fig.axes:
            assert list(axis.get_xticks()) == pytest.approx(
                [1, 5, 10, 15, 20, 25, 30]
            )
    finally:
        plt.close(fig)
