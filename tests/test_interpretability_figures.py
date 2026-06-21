import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_interpretability_figures import (
    ARCHIVED_METRICS,
    build_switch_event_study,
    build_switch_narrative_cases,
    build_multiswitch_windows,
    compute_financial_metrics,
    fixed_weight_future_curve,
    safe_corr,
    select_multiswitch_window_case,
    select_paper_switch_cases,
    select_switch_cases,
)


class InterpretabilityFigureHelperTests(unittest.TestCase):
    def test_archived_metrics_include_confirmed_final_returns(self):
        sh_s3 = ARCHIVED_METRICS[("sh", "Controller+HRL")]
        nas_s3 = ARCHIVED_METRICS[("nas", "Controller+HRL")]

        self.assertEqual(round(sh_s3["total_return"] * 100.0, 2), 204.99)
        self.assertEqual(round(nas_s3["total_return"] * 100.0, 2), 265.53)
        self.assertEqual(sh_s3["free_switches"], 102)
        self.assertEqual(nas_s3["free_switches"], 231)

    def test_compute_financial_metrics_matches_simple_curve(self):
        values = [1.0, 1.1, 1.21, 1.089]
        metrics = compute_financial_metrics(values)

        self.assertEqual(round(metrics["total_return"], 3), 0.089)
        self.assertEqual(round(metrics["max_drawdown"], 3), 0.100)
        self.assertIn("sharpe", metrics)

    def test_safe_corr_handles_constant_inputs(self):
        self.assertEqual(safe_corr([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]), 0.0)
        self.assertEqual(round(safe_corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6), 1.0)

    def test_build_switch_event_study_aligns_windows(self):
        returns = np.array([0.0, 0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
        event_idx = [3]
        study = build_switch_event_study(returns, event_idx, pre_days=2, post_days=2)

        self.assertEqual(study["offset"].tolist(), [-2, -1, 0, 1, 2])
        self.assertEqual(len(study), 5)
        self.assertEqual(study["event_count"].iloc[0], 1)
        self.assertEqual(study["mean_cum_return"].iloc[2], 0.0)

    def test_fixed_weight_future_curve_includes_transaction_cost(self):
        ratio_matrix = np.array(
            [
                [0.90, 0.90],
                [1.10, 1.10],
            ],
            dtype="float64",
        )
        weights = np.array([1.0, 0.0])
        current_weights = np.array([0.0, 1.0])

        curve = fixed_weight_future_curve(
            ratio_matrix,
            weights,
            current_weights=current_weights,
            transaction_cost=0.01,
        )

        self.assertEqual(curve.tolist()[0], 1.0)
        self.assertAlmostEqual(curve[1], 0.90 * 0.98)
        self.assertAlmostEqual(curve[2], 0.81 * 0.98)

    def test_select_switch_cases_prioritizes_avoided_downside(self):
        import pandas as pd

        trace = pd.DataFrame(
            [
                {
                    "date": "2020-01-01",
                    "is_free_switch": 1,
                    "hold_future_return_20": -0.12,
                    "switch_future_return_20": 0.03,
                    "avoided_loss_20": 0.15,
                },
                {
                    "date": "2020-01-02",
                    "is_free_switch": 1,
                    "hold_future_return_20": -0.02,
                    "switch_future_return_20": 0.10,
                    "avoided_loss_20": 0.12,
                },
                {
                    "date": "2020-01-03",
                    "is_free_switch": 0,
                    "hold_future_return_20": -0.30,
                    "switch_future_return_20": 0.30,
                    "avoided_loss_20": 0.60,
                },
            ]
        )

        cases = select_switch_cases(trace, top_n=2)

        self.assertEqual(cases["date"].tolist(), ["2020-01-01", "2020-01-02"])
        self.assertEqual(cases["case_rank"].tolist(), [1, 2])

    def test_build_switch_narrative_cases_reconstructs_full_holding_period(self):
        import json
        import pandas as pd

        trace = pd.DataFrame(
            [
                {
                    "date": "2020-01-01",
                    "step": 0,
                    "portfolio_value": 1.00,
                    "portfolio_value_before": 1.00,
                    "is_free_switch": 0,
                    "hold_duration": 0,
                    "hold_future_return_20": np.nan,
                    "switch_future_return_20": np.nan,
                    "avoided_loss_20": np.nan,
                    "hold_curve_20": "",
                    "switch_curve_20": "",
                },
                {
                    "date": "2020-01-02",
                    "step": 1,
                    "portfolio_value": 1.08,
                    "portfolio_value_before": 1.00,
                    "is_free_switch": 0,
                    "hold_duration": 1,
                    "hold_future_return_20": np.nan,
                    "switch_future_return_20": np.nan,
                    "avoided_loss_20": np.nan,
                    "hold_curve_20": "",
                    "switch_curve_20": "",
                },
                {
                    "date": "2020-01-03",
                    "step": 2,
                    "portfolio_value": 1.07,
                    "portfolio_value_before": 1.03,
                    "is_free_switch": 1,
                    "hold_duration": 2,
                    "hold_future_return_20": -0.10,
                    "switch_future_return_20": 0.04,
                    "avoided_loss_20": 0.14,
                    "hold_curve_20": json.dumps([1.0, 0.96, 0.90]),
                    "switch_curve_20": json.dumps([1.0, 1.02, 1.04]),
                },
            ]
        )

        cases = build_switch_narrative_cases(trace, market="sh", top_n=1, horizons=(2,))

        self.assertEqual(cases["start_date"].tolist(), ["2020-01-01"])
        self.assertEqual(cases["switch_date"].tolist(), ["2020-01-03"])
        self.assertEqual(cases["post_horizon"].tolist(), [2])
        self.assertEqual(cases["pre_switch_return"].round(6).tolist(), [0.03])
        self.assertEqual(cases["post_hold_return"].round(6).tolist(), [-0.10])
        self.assertEqual(cases["post_switch_return"].round(6).tolist(), [0.04])
        self.assertEqual(cases["avoided_deterioration"].round(6).tolist(), [0.14])
        pre_curve = [round(x, 6) for x in json.loads(cases["pre_curve"].iloc[0])]
        self.assertEqual(pre_curve, [0.0, 0.08, 0.03])

    def test_select_paper_switch_cases_keeps_strong_nas_examples(self):
        import pandas as pd

        candidates = pd.DataFrame(
            [
                {
                    "market": "nas",
                    "case_rank": 1,
                    "holding_days": 1,
                    "pre_switch_drawdown": 0.0,
                    "post_hold_return": -0.04,
                    "post_switch_return": -0.03,
                    "avoided_deterioration": 0.01,
                    "post_horizon": 20,
                    "story_score": 0.01,
                },
                {
                    "market": "sh",
                    "case_rank": 1,
                    "holding_days": 8,
                    "pre_switch_drawdown": 0.03,
                    "post_hold_return": -0.10,
                    "post_switch_return": 0.04,
                    "avoided_deterioration": 0.14,
                    "post_horizon": 20,
                    "story_score": 0.15,
                },
                {
                    "market": "sh",
                    "case_rank": 2,
                    "holding_days": 27,
                    "pre_switch_drawdown": 0.04,
                    "post_hold_return": -0.03,
                    "post_switch_return": 0.03,
                    "avoided_deterioration": 0.06,
                    "post_horizon": 20,
                    "story_score": 0.07,
                },
                {
                    "market": "nas",
                    "case_rank": 2,
                    "holding_days": 19,
                    "pre_switch_drawdown": 0.094,
                    "post_hold_return": -0.030,
                    "post_switch_return": -0.002,
                    "avoided_deterioration": 0.028,
                    "post_horizon": 10,
                    "story_score": 0.04,
                },
                {
                    "market": "nas",
                    "case_rank": 3,
                    "holding_days": 1,
                    "pre_switch_drawdown": 0.0,
                    "post_hold_return": -0.038,
                    "post_switch_return": 0.002,
                    "avoided_deterioration": 0.040,
                    "post_horizon": 20,
                    "story_score": 0.04,
                },
            ]
        )

        selected = select_paper_switch_cases(candidates, max_cases=4)

        self.assertEqual(selected["market"].tolist(), ["sh", "sh", "nas", "nas"])
        self.assertEqual(selected["post_horizon"].tolist(), [20, 20, 10, 20])

    def test_build_multiswitch_windows_scores_30_day_risk_control(self):
        import json
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
        controller = pd.DataFrame(
            {
                "date": dates,
                "portfolio_value": [1.00, 1.02, 1.01, 1.04, 1.03, 1.06],
                "is_free_switch": [0, 1, 0, 1, 0, 1],
                "is_switch": [0, 1, 0, 1, 0, 1],
            }
        )
        fixed = pd.DataFrame(
            {
                "date": dates,
                "portfolio_value": [1.00, 0.99, 0.94, 0.96, 0.95, 0.98],
            }
        )

        windows = build_multiswitch_windows(
            controller,
            fixed,
            market="nas",
            window_days=5,
            min_free_switches=2,
        )
        selected = select_multiswitch_window_case(windows)

        self.assertEqual(selected["market"], "nas")
        self.assertEqual(selected["free_switches"], 3)
        self.assertGreater(selected["mdd_reduction"], 0.0)
        self.assertGreater(selected["return_gap"], 0.0)
        self.assertEqual(json.loads(selected["switch_offsets"]), [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
