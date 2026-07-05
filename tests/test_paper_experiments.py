import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_experiments.metrics import compute_financial_metrics, summarize_inner_alpha
from paper_experiments.trace_utils import discover_runs, parse_seed_specs
from paper_experiments.eval_end_to_end_explain import ABLATION_SCENARIOS, SCENARIOS, STAGE_SPECS
from paper_experiments.plot_end_to_end_explain import (
    COLORS,
    EXPORT_DPI,
    MAIN_COMPARISON_METRICS,
    _annotate_endpoint,
    _figure_group_for_stem,
    _market_label,
    _paper_title,
    _prepare_main_comparison_rows,
    _prepare_ablation_rows,
    _save,
)
from paper_experiments.table_end_to_end_explain import METHOD_NAMES, _filter_display_rows


class PaperExperimentHelperTests(unittest.TestCase):
    def test_financial_metrics_handle_drawdown_and_sortino(self):
        df = pd.DataFrame({"portfolio_value": [1.0, 1.1, 0.99, 1.20]})
        metrics = compute_financial_metrics(df)

        self.assertAlmostEqual(metrics["total_return"], 0.20)
        self.assertGreater(metrics["annualized_volatility"], 0.0)
        self.assertGreater(metrics["max_drawdown"], 0.0)
        self.assertIn("sortino", metrics)
        self.assertIn("calmar", metrics)

    def test_financial_metrics_include_recorded_first_daily_return(self):
        df = pd.DataFrame(
            {
                "portfolio_value_before": [1000.0, 990.0],
                "portfolio_value": [990.0, 1089.0],
                "daily_simple_return": [-0.01, 0.10],
            }
        )
        metrics = compute_financial_metrics(df)

        self.assertAlmostEqual(metrics["total_return"], 0.089)
        self.assertAlmostEqual(metrics["annualized_return"], 0.09 / 2 * 252)
        self.assertAlmostEqual(metrics["daily_win_rate"], 0.5)

    def test_inner_alpha_summary_ignores_nan(self):
        df = pd.DataFrame(
            {
                "inner_alpha": [0.01, np.nan, -0.005, 0.002],
                "turnover": [0.2, 0.1, 0.2, 0.1],
            }
        )
        summary = summarize_inner_alpha(df)

        self.assertAlmostEqual(summary["cumulative_inner_alpha"], 0.007)
        self.assertAlmostEqual(summary["positive_inner_alpha_ratio"], 2 / 3)
        self.assertGreater(summary["inner_alpha_per_turnover"], 0.0)

    def test_discover_runs_reports_missing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "sh_seed90"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            (run_dir / "seed_90_command.json").write_text(
                json.dumps({"command": ["python", "run_hrl_training.py", "--markets", "sh"]}),
                encoding="utf-8",
            )
            (ckpt_dir / "best_model.pth").write_text("placeholder", encoding="utf-8")

            runs = discover_runs(root, markets=["sh"], seed_map={"sh": [90]})

        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].checkpoints["best_model"].exists)
        self.assertFalse(runs[0].checkpoints["controller_best"].exists)

    def test_parse_seed_specs_supports_market_mapping(self):
        parsed = parse_seed_specs(["sh:90", "nas:49"])

        self.assertEqual(parsed["sh"], [90])
        self.assertEqual(parsed["nas"], [49])

    def test_controller_outer_ablation_is_controller_with_inner_disabled(self):
        self.assertIn("controller_outer", SCENARIOS)
        scenario = SCENARIOS["controller_outer"]

        self.assertIsNone(scenario["fixed_cycle"])
        self.assertTrue(scenario["use_controller"])
        self.assertTrue(scenario["disable_inner"])
        self.assertIn("controller_outer", ABLATION_SCENARIOS)

    def test_plot_ablation_rows_exclude_stage_duplicates(self):
        rows = pd.DataFrame(
            [
                {"scenario": "fixed_hrl", "stage": "Fixed HRL checkpoint", "total_return": 0.1},
                {"scenario": "full_controller", "stage": "Controller-PG checkpoint", "total_return": 0.4},
                {"scenario": "fixed_hrl_no_inner", "stage": np.nan, "total_return": 0.0},
                {"scenario": "fixed_hrl", "stage": np.nan, "total_return": 0.2},
                {"scenario": "controller_outer", "stage": np.nan, "total_return": 0.3},
                {"scenario": "full_controller", "stage": np.nan, "total_return": 0.5},
            ]
        )

        selected = _prepare_ablation_rows(rows)

        self.assertEqual(
            selected["scenario"].astype(str).tolist(),
            ["fixed_hrl_no_inner", "fixed_hrl", "controller_outer", "full_controller"],
        )
        self.assertEqual(selected["total_return"].tolist(), [0.0, 0.2, 0.3, 0.5])

    def test_main_comparison_rows_keep_only_full_controller_and_fixed_hrl(self):
        rows = pd.DataFrame(
            [
                {"scenario": "fixed_hrl_no_inner", "stage": np.nan, "total_return": 0.0},
                {"scenario": "fixed_hrl", "stage": np.nan, "total_return": 0.2},
                {"scenario": "controller_outer", "stage": np.nan, "total_return": 0.5},
                {"scenario": "full_controller", "stage": np.nan, "total_return": 0.4},
            ]
        )

        selected = _prepare_main_comparison_rows(rows)

        self.assertEqual(selected["scenario"].astype(str).tolist(), ["fixed_hrl", "full_controller"])
        self.assertNotIn("controller_outer", selected["scenario"].astype(str).tolist())
        self.assertNotIn("fixed_hrl_no_inner", selected["scenario"].astype(str).tolist())

    def test_main_comparison_metrics_focus_on_performance_not_risk_tradeoff(self):
        self.assertEqual(MAIN_COMPARISON_METRICS, ["total_return", "sharpe"])

    def test_stage_progression_omits_final_e2e_display(self):
        labels = [stage for stage, _checkpoint, _scenario in STAGE_SPECS]

        self.assertEqual(labels, ["Fixed HRL checkpoint", "Controller-PG checkpoint"])
        self.assertNotIn("Final E2E checkpoint", labels)

    def test_tables_label_full_controller_plainly_and_filter_final_e2e_rows(self):
        rows = pd.DataFrame(
            [
                {"scenario": "full_controller", "stage": "Final E2E checkpoint"},
                {"scenario": "full_controller", "stage": "Controller-PG checkpoint"},
            ]
        )

        filtered = _filter_display_rows(rows)

        self.assertEqual(METHOD_NAMES["full_controller"], "Full Controller")
        self.assertEqual(filtered["stage"].tolist(), ["Controller-PG checkpoint"])

    def test_figure_stems_are_grouped_by_experiment(self):
        self.assertEqual(_figure_group_for_stem("fig03c_inference_ablation_bar_sh_seed90"), "02_inference_ablation")
        self.assertEqual(_figure_group_for_stem("fig07_random_switch_comparison_nas_seed49"), "06_random_switch")
        self.assertEqual(_figure_group_for_stem("fig10_case_window_sh_2021_07_large_avoidance"), "07_case_windows")

    def test_pdf_fonttype_uses_embedded_truetype(self):
        import matplotlib

        self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["ps.fonttype"], 42)

    def test_aaai_style_uses_high_dpi_and_colorblind_safe_palette(self):
        self.assertGreaterEqual(EXPORT_DPI, 420)
        self.assertEqual(COLORS["full_controller"], "#D55E00")
        self.assertEqual(COLORS["controller_outer"], "#E69F00")
        self.assertEqual(COLORS["fixed_hrl_no_inner"], "#0072B2")
        self.assertEqual(COLORS["switch"], "#CC79A7")

    def test_endpoint_helper_adds_readable_label_without_metric_box(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(2.5, 1.8))
        line = ax.plot([0, 1, 2], [1.0, 1.2, 1.5], color=COLORS["full_controller"])[0]

        _annotate_endpoint(ax, line, "Full controller")

        texts = [text.get_text() for text in ax.texts]
        self.assertIn("Full Controller", texts)
        self.assertFalse(any("Return" in text or "Switches" in text for text in texts))
        plt.close(fig)

    def test_main_pdf_is_flattened_and_editable_pdf_is_preserved(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(figsize=(2.0, 1.4))
            ax.plot([0, 1], [0, 1], label="Full controller")
            ax.set_title("Readable PDF")
            ax.legend(frameon=False)

            _save(fig, Path(tmp), "fig03_test_pdf_export")

            target_dir = Path(tmp) / "02_inference_ablation"
            main_pdf = target_dir / "fig03_test_pdf_export.pdf"
            editable_pdf = target_dir / "fig03_test_pdf_export_editable.pdf"

            self.assertTrue((target_dir / "fig03_test_pdf_export.png").exists())
            self.assertTrue(main_pdf.exists())
            self.assertTrue(editable_pdf.exists())
            self.assertIn(b"/Image", main_pdf.read_bytes())
            self.assertNotIn(b"/FontFile", main_pdf.read_bytes())
            self.assertIn(b"/Font", editable_pdf.read_bytes())

    def test_paper_titles_hide_seed_and_hyperparameters(self):
        titles = [
            _paper_title("Inference Ablation", "sh"),
            _paper_title("Random Switch Matched-Count Comparison", "nas"),
            _paper_title("Switch Event Study", "nas"),
        ]

        self.assertEqual(_market_label("sh"), "SH Market")
        self.assertEqual(_market_label("nas"), "NASDAQ Market")
        for title in titles:
            lower = title.lower()
            self.assertNotIn("seed", lower)
            self.assertNotIn("epoch", lower)
            self.assertNotIn("lr", lower)


if __name__ == "__main__":
    unittest.main()
