import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from paper_experiments import plot_inner_actor_base_adjustment as trader


def panel_data(scale: float, periods: int = 30) -> dict:
    dates = pd.bdate_range("2021-01-04", periods=periods)
    assets = ["AAA.O", "BBB.O", "CCC.O"]
    return {
        "assets": assets,
        "idx": dates,
        "fut_pct": pd.DataFrame(
            np.arange(3 * periods, dtype=float).reshape(3, periods) * scale,
            index=assets,
            columns=dates,
        ),
        "tilt_pct": pd.DataFrame(
            np.linspace(-1, 1, 3 * periods).reshape(3, periods) * scale,
            index=assets,
            columns=dates,
        ),
        "xticks": np.array([0, 14, periods - 1]),
        "xticklabels": [
            dates[0].strftime("%Y-%m-%d"),
            dates[14].strftime("%Y-%m-%d"),
            dates[-1].strftime("%Y-%m-%d"),
        ],
    }


def test_trader_refinement_matches_controller_style_layout(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def capture(fig, path, **kwargs):
        captured["fig"] = fig
        captured["path"] = path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(trader, "save_figure", capture)

    trader.plot_combined_market_heatmaps(
        {"nas": panel_data(0.1), "sh": panel_data(0.2)},
        tmp_path,
        future_horizon=7,
    )

    fig = captured["fig"]
    try:
        image_axes = [axis for axis in fig.axes if axis.images]
        assert len(image_axes) == 4
        top_left, top_right, bottom_left, bottom_right = image_axes
        assert top_left.get_position().x0 < top_right.get_position().x0
        assert top_left.get_position().y0 > bottom_left.get_position().y0
        assert bottom_left.get_position().x0 < bottom_right.get_position().x0
        assert not any(axis.get_title() for axis in image_axes)
        np.testing.assert_allclose(
            np.asarray(top_left.images[0].get_array()),
            panel_data(0.2)["fut_pct"].to_numpy(),
        )
        np.testing.assert_allclose(
            np.asarray(top_right.images[0].get_array()),
            panel_data(0.2)["tilt_pct"].to_numpy(),
        )
        np.testing.assert_allclose(
            np.asarray(bottom_left.images[0].get_array()),
            panel_data(0.1)["fut_pct"].to_numpy(),
        )
        np.testing.assert_allclose(
            np.asarray(bottom_right.images[0].get_array()),
            panel_data(0.1)["tilt_pct"].to_numpy(),
        )
        figure_text = [text.get_text() for text in fig.texts]
        assert figure_text.count("A. Future 7-day relative return") == 1
        assert figure_text.count("B. Refinement tilt") == 1
        assert "CSI-300" not in figure_text
        assert "Nasdaq-100" not in figure_text
        figure_width, figure_height = fig.get_size_inches()
        assert figure_width <= 14.0
        assert figure_height <= 5.8
        assert top_left.get_position().x0 <= 0.10
        assert all(
            axis.get_yticklabels()[0].get_fontsize() >= 13
            for axis in image_axes
        )
        assert all(
            axis.get_xticklabels()[0].get_fontsize() >= 12
            for axis in image_axes
        )
        assert all(axis.xaxis.label.get_fontsize() >= 13 for axis in image_axes)
        assert [label.get_text() for label in top_left.get_xticklabels()] == [
            "1",
            "5",
            "10",
            "15",
            "20",
            "25",
            "30",
        ]
        assert all(
            axis.get_xlabel() == "2021-01-04—2021-02-12"
            for axis in image_axes
        )
        bottom_titles = [
            text
            for text in fig.texts
            if text.get_text().startswith(("A.", "B."))
        ]
        assert all(text.get_fontsize() >= 17 for text in bottom_titles)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert all(
            label.get_window_extent(renderer).x0 >= fig.bbox.x0
            for axis in image_axes
            for label in axis.get_yticklabels()
        )
        assert len(fig.axes) == 6
        colorbar_axes = [axis for axis in fig.axes if not axis.images]
        assert all(
            axis.xaxis.label.get_fontsize() >= 13 for axis in colorbar_axes
        )
        assert all(
            axis.get_xticklabels()[0].get_fontsize() >= 12
            for axis in colorbar_axes
        )
        assert colorbar_axes[0].get_position().y0 > top_left.get_position().y1
        assert colorbar_axes[1].get_position().y0 > top_right.get_position().y1
        assert colorbar_axes[0].get_position().x0 == pytest.approx(
            top_left.get_position().x0
        )
        assert colorbar_axes[1].get_position().x0 == pytest.approx(
            top_right.get_position().x0
        )
        assert top_left.images[0].get_clim() == bottom_left.images[0].get_clim()
        assert top_right.images[0].get_clim() == bottom_right.images[0].get_clim()
        assert any(label.get_text() for label in top_left.get_xticklabels())
        assert any(label.get_text() for label in bottom_left.get_xticklabels())
        assert captured["path"].name == "trader_refinement_two_markets"
        assert captured["kwargs"] == {"pad_inches": 0.02}
    finally:
        plt.close(fig)


def test_trader_refinement_caps_combined_heatmaps_at_30_days(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    monkeypatch.setattr(
        trader,
        "save_figure",
        lambda fig, path, **kwargs: captured.setdefault("fig", fig),
    )

    trader.plot_combined_market_heatmaps(
        {"nas": panel_data(0.1, periods=40), "sh": panel_data(0.2, periods=40)},
        tmp_path,
        future_horizon=5,
    )

    fig = captured["fig"]
    try:
        image_axes = [axis for axis in fig.axes if axis.images]
        assert all(axis.images[0].get_array().shape[1] == 30 for axis in image_axes)
        assert all(
            axis.get_xticklabels()[-1].get_text() == "30" for axis in image_axes
        )
        expected_future_limit = max(
            1.0,
            *[
                float(
                    np.nanpercentile(
                        np.abs(
                            panel_data(scale, periods=40)["fut_pct"]
                            .iloc[:, :30]
                            .to_numpy()
                        ),
                        94,
                    )
                )
                for scale in (0.1, 0.2)
            ],
        )
        assert image_axes[0].images[0].get_clim() == pytest.approx(
            (-expected_future_limit, expected_future_limit)
        )
    finally:
        plt.close(fig)


def test_select_window_accepts_exactly_30_valid_days() -> None:
    index = pd.bdate_range("2021-01-04", periods=30)
    tilt = pd.DataFrame(
        {
            "AAA.O": np.linspace(-0.02, 0.03, 30),
            "BBB.O": np.linspace(0.01, -0.01, 30),
        },
        index=index,
    )
    future = pd.DataFrame(
        {
            "AAA.O": np.linspace(-0.01, 0.02, 30),
            "BBB.O": np.linspace(0.02, -0.02, 30),
        },
        index=index,
    )

    selected = trader.select_window(tilt, future, windows=(30,))

    assert selected["start"] == 0
    assert selected["end"] == 29
    assert selected["window"] == 30


def test_trader_figure_is_saved_at_paper_resolution(monkeypatch, tmp_path) -> None:
    fig = plt.figure()
    calls = []
    monkeypatch.setattr(fig, "savefig", lambda *args, **kwargs: calls.append(kwargs))

    trader.save_figure(fig, tmp_path / "trader", pad_inches=0.02)

    assert len(calls) == 2
    assert all(call["dpi"] == 240 for call in calls)
