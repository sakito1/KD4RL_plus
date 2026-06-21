import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_interpretability_figures import (
    ARCHIVED_METRICS,
    build_switch_event_study,
    compute_financial_metrics,
    safe_corr,
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


if __name__ == "__main__":
    unittest.main()
