import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from paper_experiments import plot_inner_actor_base_adjustment as trader


def panel_data(scale: float) -> dict:
    dates = pd.bdate_range("2021-01-04", periods=8)
    assets = ["AAA.O", "BBB.O", "CCC.O"]
    return {
        "assets": assets,
        "idx": dates,
        "fut_pct": pd.DataFrame(
            np.arange(24, dtype=float).reshape(3, 8) * scale,
            index=assets,
            columns=dates,
        ),
        "tilt_pct": pd.DataFrame(
            np.linspace(-1, 1, 24).reshape(3, 8) * scale,
            index=assets,
            columns=dates,
        ),
        "xticks": np.array([0, 3, 7]),
        "xticklabels": ["2021-01-04", "2021-01-07", "2021-01-13"],
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
        assert figure_text.count("CSI-300") == 1
        assert figure_text.count("Nasdaq-100") == 1
        assert top_left.get_yticklabels()[0].get_fontsize() >= 11
        assert len(fig.axes) == 6
        colorbar_axes = [axis for axis in fig.axes if not axis.images]
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


def test_trader_figure_is_saved_at_paper_resolution(monkeypatch, tmp_path) -> None:
    fig = plt.figure()
    calls = []
    monkeypatch.setattr(fig, "savefig", lambda *args, **kwargs: calls.append(kwargs))

    trader.save_figure(fig, tmp_path / "trader", pad_inches=0.02)

    assert len(calls) == 2
    assert all(call["dpi"] == 240 for call in calls)
