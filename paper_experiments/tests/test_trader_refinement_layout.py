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


def test_trader_refinement_combines_markets_side_by_side_with_large_text(
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
        assert bottom_left.get_position().x0 < bottom_right.get_position().x0
        assert top_left.get_title() == "Future 7-day relative return"
        assert bottom_left.get_title() == "Refinement tilt"
        figure_text = [text.get_text() for text in fig.texts]
        assert "Nasdaq-100 Trader Refinement" in figure_text
        assert "CSI-300 Trader Refinement" in figure_text
        assert top_left.title.get_fontsize() >= 14
        assert top_left.get_yticklabels()[0].get_fontsize() >= 11
        assert len(fig.axes) == 6
        colorbar_axes = [axis for axis in fig.axes if not axis.images]
        assert colorbar_axes[0].get_position().x0 > top_right.get_position().x1
        assert colorbar_axes[1].get_position().x0 > bottom_right.get_position().x1
        assert top_left.images[0].get_clim() == top_right.images[0].get_clim()
        assert bottom_left.images[0].get_clim() == bottom_right.images[0].get_clim()
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
