import json
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_experiments import plot_inner_actor_base_adjustment as inner_plot
from paper_experiments import run_paper_experiments_final as final_plot


def visible_font_sizes(fig):
    fig.canvas.draw()
    return [
        float(text.get_fontsize())
        for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text))
        if text.get_visible() and text.get_text().strip()
    ]


class PaperFigureReadabilityTests(unittest.TestCase):
    @patch.object(plt.Figure, "savefig")
    def test_final_figure_export_reserves_large_font_padding(self, savefig):
        fig = plt.figure()

        final_plot.save_figure(fig, Path("unused"))

        self.assertEqual(savefig.call_count, 2)
        self.assertTrue(
            all(call.kwargs.get("pad_inches", 0.0) >= 0.25 for call in savefig.call_args_list)
        )

    def load_composites(self):
        path = ROOT / "paper_experiments" / "plot_paper_figure_composites.py"
        self.assertTrue(path.exists(), "composite plotting module is not implemented")
        spec = importlib.util.spec_from_file_location("paper_figure_composites", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def assert_minimum_figure_font(self, fig):
        sizes = visible_font_sizes(fig)
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 21.0)

    @patch.object(final_plot, "save_figure")
    @patch.object(final_plot, "read_curve")
    def test_main_equity_identifies_normalized_portfolio_value(self, read_curve, save_figure):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        read_curve.return_value = pd.DataFrame(
            {"date": dates, "wealth": [1.0, 1.1, 1.05, 1.2, 1.3]}
        )
        manifest = pd.DataFrame(
            [
                {
                    "market": "nas",
                    "curve_status": "available",
                    "method": "buy_hold",
                    "curve_path": "dummy.csv",
                }
            ]
        )

        final_plot.plot_main_equity(manifest, Path("unused"), "nas", 49, Path("unused"))

        fig = save_figure.call_args.args[0]
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Nasdaq-100 Portfolio Value")
        self.assertEqual(ax.get_ylabel(), "Portfolio value\n(initial = 1.0)")
        self.assertEqual(tuple(fig.get_size_inches()), (12.5, 6.2))
        self.assertGreaterEqual(ax.title.get_fontsize(), 30.0)
        self.assertGreaterEqual(ax.yaxis.label.get_fontsize(), 25.0)
        tick_labels = [*ax.get_xticklabels(), *ax.get_yticklabels()]
        self.assertTrue(all(text.get_fontsize() >= 21.0 for text in tick_labels))
        self.assertTrue(all(text.get_fontsize() >= 20.0 for text in ax.get_legend().get_texts()))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_box = ax.yaxis.label.get_window_extent(renderer)
        figure_box = fig.bbox
        self.assertGreaterEqual(label_box.y0, figure_box.y0)
        self.assertLessEqual(label_box.y1, figure_box.y1)
        plt.close(fig)

    @patch.object(final_plot, "save_figure")
    def test_controller_case_has_two_rows_and_trading_day_axis(self, save_figure):
        hold = np.linspace(1.0, 0.96, 31)
        switch = np.linspace(1.0, 1.04, 31)
        case = pd.Series(
            {
                "step": 20,
                "date": "2021-04-19",
                "exit_prob": 0.63,
                "hold_curve_30": json.dumps(hold.tolist()),
                "switch_curve_30": json.dumps(switch.tolist()),
            }
        )
        actions = pd.DataFrame({"step": [20], "is_switch": [1], "is_free_switch": [1]})

        final_plot.plot_controller_case(
            "nas", 1, case, pd.DataFrame({"step": [0, 40]}), actions, Path("unused")
        )

        fig = save_figure.call_args.args[0]
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig._suptitle.get_text(), "Nasdaq-100 switch on 2021-04-19 (p = 0.63)")
        self.assertEqual(fig.axes[0].get_title(loc="left"), "A. Frozen portfolio return")
        self.assertEqual(fig.axes[1].get_title(loc="left"), "B. Frozen portfolio drawdown")
        self.assertEqual(fig.axes[1].get_xlabel(), "Trading days after switch")
        all_text = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
        self.assertNotIn("Day-0 decision evidence", all_text)
        self.assertEqual(tuple(fig.get_size_inches()), (7.2, 5.6))
        self.assertEqual(fig._suptitle.get_fontsize(), 18.0)
        self.assertEqual(fig._suptitle.get_fontweight(), "semibold")
        self.assertTrue(all(ax._left_title.get_fontsize() == 13.0 for ax in fig.axes))
        self.assertTrue(all(ax._left_title.get_fontweight() == "semibold" for ax in fig.axes))
        self.assertTrue(all(ax.yaxis.label.get_fontsize() == 11.0 for ax in fig.axes))
        self.assertEqual(fig.axes[1].xaxis.label.get_fontsize(), 11.0)
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0 and text.get_fontweight() == "normal"
                for ax in fig.axes
                for text in [*ax.get_xticklabels(), *ax.get_yticklabels()]
            )
        )
        self.assertEqual(len(fig.legends), 1)
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0 and text.get_fontweight() == "normal"
                for text in fig.legends[0].get_texts()
            )
        )
        metric_text = [
            text
            for ax in fig.axes
            for text in ax.texts
            if text.get_text().startswith(("Return gap", "MDD reduction"))
        ]
        self.assertEqual(len(metric_text), 2)
        self.assertTrue(
            all(
                text.get_fontsize() == 11.0 and text.get_fontweight() == "semibold"
                for text in metric_text
            )
        )
        self.assertGreater(fig.axes[0].get_position().y1, 0.78)
        panel_gap = fig.axes[0].get_position().y0 - fig.axes[1].get_position().y1
        self.assertLess(panel_gap, 0.18)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = fig.legends[0].get_window_extent(renderer)
        self.assertTrue(all(not legend_box.overlaps(ax.get_window_extent(renderer)) for ax in fig.axes))
        panel_title_box = fig.axes[0]._left_title.get_window_extent(renderer)
        self.assertFalse(legend_box.overlaps(panel_title_box))
        endpoint_labels = [
            text
            for text in fig.axes[0].texts
            if text.get_text().startswith(("Hold ", "Switch "))
        ]
        self.assertEqual(len(endpoint_labels), 2)
        endpoint_offsets = {
            text.get_text().split()[0]: tuple(text.xyann) for text in endpoint_labels
        }
        self.assertEqual(endpoint_offsets, {"Hold": (-8, -22), "Switch": (-8, 12)})
        self.assertTrue(
            all(
                text.get_fontsize() == 11.0 and text.get_fontweight() == "semibold"
                for text in endpoint_labels
            )
        )
        axes_box = fig.axes[0].get_window_extent(renderer)
        for text in endpoint_labels:
            box = text.get_window_extent(renderer)
            self.assertGreaterEqual(box.x0, axes_box.x0)
            self.assertLessEqual(box.x1, axes_box.x1)
            self.assertGreaterEqual(box.y0, axes_box.y0)
            self.assertLessEqual(box.y1, axes_box.y1)
        plt.close(fig)

    @patch.object(inner_plot, "save_figure")
    @patch.object(inner_plot, "select_window")
    @patch.object(inner_plot, "future_relative_return")
    @patch.object(inner_plot, "load_prices")
    @patch.object(inner_plot, "parse_matrix")
    def test_inner_case_keeps_four_rows_with_compact_titles(
        self, parse_matrix, load_prices, future_relative_return, select_window, save_figure
    ):
        dates = pd.bdate_range("2024-01-02", periods=35)
        assets = [f"A{i}" for i in range(6)]
        base = pd.DataFrame(1.0 / 6.0, index=dates, columns=assets)
        tilt = pd.DataFrame(
            np.linspace(-0.005, 0.005, len(dates) * len(assets)).reshape(len(dates), len(assets)),
            index=dates,
            columns=assets,
        )
        executed = base + tilt
        parse_matrix.side_effect = lambda _actions, column: {
            "base_weights_json": base,
            "exec_weights_json": executed,
            "inner_tilt_json": tilt,
        }[column]
        load_prices.return_value = base
        future_relative_return.return_value = tilt * 4.0
        select_window.return_value = {"start": 0, "end": 29, "assets": assets}

        inner_plot.plot_market("nas", pd.DataFrame(), Path("unused"), future_horizon=5)

        fig = save_figure.call_args.args[0]
        panels = fig.axes[:4]
        self.assertEqual(fig._suptitle.get_text(), "Nasdaq-100 Inner-Actor Refinement")
        self.assertEqual(
            [ax.get_title(loc="left") for ax in panels],
            [
                "Future 5-day relative return",
                "Inner tilt",
                "Executed weights",
                "Tilt-return alignment",
            ],
        )
        self.assertEqual(len(panels), 4)
        verbose_text = " ".join(ax.get_title(loc="left") for ax in panels)
        self.assertNotIn("green =", verbose_text)
        self.assertNotIn("positive bars mean", verbose_text)
        width, height = fig.get_size_inches()
        self.assertEqual((width, height), (11.5, 8.8))
        panel_gaps = [
            panels[index].get_position().y0 - panels[index + 1].get_position().y1
            for index in range(len(panels) - 1)
        ]
        self.assertLess(max(panel_gaps), 0.10)
        self.assertEqual(fig._suptitle.get_fontsize(), 18.0)
        self.assertEqual(fig._suptitle.get_fontweight(), "semibold")
        self.assertTrue(all(ax._left_title.get_fontsize() == 13.0 for ax in panels))
        self.assertTrue(all(ax._left_title.get_fontweight() == "semibold" for ax in panels))
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0 and text.get_fontweight() == "normal"
                for ax in panels[:3]
                for text in ax.get_yticklabels()
            )
        )
        self.assertEqual(panels[3].xaxis.label.get_fontsize(), 11.0)
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0 and text.get_fontweight() == "normal"
                for text in [*panels[3].get_xticklabels(), *panels[3].get_yticklabels()]
            )
        )
        self.assertTrue(
            all(
                text.get_fontsize() == 11.0 and text.get_fontweight() == "semibold"
                for text in panels[3].texts
            )
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        summary = next(text for text in panels[3].texts if text.get_text().startswith("Mean r"))
        self.assertFalse(summary.get_window_extent(renderer).overlaps(panels[3].get_window_extent(renderer)))
        colorbars = fig.axes[4:]
        self.assertEqual(len(colorbars), 3)
        self.assertTrue(all(ax.yaxis.label.get_fontsize() == 11.0 for ax in colorbars))
        self.assertTrue(
            all(
                text.get_fontsize() == 10.0 and text.get_fontweight() == "normal"
                for ax in colorbars
                for text in ax.get_yticklabels()
            )
        )
        plt.close(fig)

    def test_main_composite_uses_shared_labels_and_21pt_minimum(self):
        composites = self.load_composites()
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        curves = {
            market: [
                {
                    "date": dates,
                    "wealth": np.linspace(1.0, 1.3 if market == "nas" else 1.2, len(dates)),
                    "label": "Ours",
                    "color": "#B83349",
                    "linewidth": 3.0,
                    "zorder": 5,
                }
            ]
            for market in ("nas", "sh")
        }

        fig = composites.render_main_composite(curves)

        self.assertEqual(len(fig.axes), 2)
        self.assertAlmostEqual(fig.get_size_inches()[0], 7.0)
        self.assertIn(
            "Portfolio value (initial = 1.0)",
            [text.get_text() for text in fig.texts],
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = fig.legends[0].get_window_extent(renderer)
        ylabel = next(text for text in fig.texts if "Portfolio value" in text.get_text())
        self.assertFalse(legend_box.overlaps(ylabel.get_window_extent(renderer)))
        for ax in fig.axes:
            for tick in ax.get_xticklabels():
                if tick.get_visible() and tick.get_text():
                    self.assertFalse(legend_box.overlaps(tick.get_window_extent(renderer)))
        self.assert_minimum_figure_font(fig)
        plt.close(fig)

    def test_controller_composite_has_four_panels_and_21pt_minimum(self):
        composites = self.load_composites()
        days = np.arange(31)
        hold = np.linspace(0.0, -4.0, len(days))
        switch = np.linspace(0.0, 3.0, len(days))
        case = {
            "date": "2021-04-19",
            "probability": 0.63,
            "days": days,
            "hold_return_path": hold,
            "switch_return_path": switch,
            "hold_drawdown_path": np.maximum.accumulate(-hold),
            "switch_drawdown_path": np.zeros_like(switch),
            "hold_return": -0.04,
            "switch_return": 0.03,
            "return_gap": 0.07,
            "mdd_reduction": 0.04,
        }

        fig = composites.render_controller_composite({"nas": case, "sh": case})

        self.assertEqual(len(fig.axes), 4)
        self.assertAlmostEqual(fig.get_size_inches()[0], 7.0)
        self.assertIn(
            "Trading days after switch",
            [text.get_text() for text in fig.texts],
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes[:2]:
            self.assertLessEqual(
                ax.title.get_window_extent(renderer).width,
                ax.get_window_extent(renderer).width,
            )
        for col, market in enumerate(("nas", "sh")):
            case_data = {"nas": case, "sh": case}[market]
            data_min = min(case_data["hold_return_path"].min(), case_data["switch_return_path"].min())
            data_max = max(case_data["hold_return_path"].max(), case_data["switch_return_path"].max())
            self.assertLess(fig.axes[col].get_ylim()[0], data_min - 0.15 * (data_max - data_min))
            drawdown_max = max(case_data["hold_drawdown_path"].max(), case_data["switch_drawdown_path"].max())
            self.assertGreater(fig.axes[2 + col].get_ylim()[1], drawdown_max * 1.15)
        self.assert_minimum_figure_font(fig)
        plt.close(fig)

    def test_inner_composite_has_eight_panels_and_21pt_minimum(self):
        composites = self.load_composites()
        assets = [f"A{i}" for i in range(6)]
        matrix = np.linspace(-1.0, 1.0, 6 * 12).reshape(6, 12)
        case = {
            "assets": assets,
            "future_return": matrix * 4.0,
            "tilt": matrix * 0.4,
            "executed_weight": np.abs(matrix) * 8.0,
            "asset_alignment": pd.Series(np.linspace(-5.0, 20.0, 6), index=assets),
            "asset_hit_rate": pd.Series(np.linspace(0.3, 0.8, 6), index=assets),
            "correlation": 0.45,
            "positive_ratio": 0.73,
        }

        fig = composites.render_inner_composite({"nas": case, "sh": case})

        self.assertEqual(len(fig.axes[:8]), 8)
        self.assertAlmostEqual(fig.get_size_inches()[0], 7.0)
        self.assertIn("Inner-Actor Refinement", [text.get_text() for text in fig.texts])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        heading_boxes = [
            text.get_window_extent(renderer)
            for text in fig.texts
            if text.get_text() in {"Inner-Actor Refinement", "Nasdaq-100", "CSI-300"}
        ]
        self.assertEqual(len(heading_boxes), 3)
        self.assertFalse(heading_boxes[0].overlaps(heading_boxes[1]))
        self.assertFalse(heading_boxes[0].overlaps(heading_boxes[2]))
        for ax in (fig.axes[6], fig.axes[7]):
            self.assertIn("r=", ax.get_xlabel())
            self.assertEqual(len(ax.texts), 0)
            xlabel_box = ax.xaxis.label.get_window_extent(renderer)
            self.assertLessEqual(xlabel_box.width, ax.get_window_extent(renderer).width)
            self.assertGreaterEqual(xlabel_box.y0, fig.bbox.y0)
        self.assert_minimum_figure_font(fig)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
