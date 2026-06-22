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


class PaperExperimentHelperTests(unittest.TestCase):
    def test_financial_metrics_handle_drawdown_and_sortino(self):
        df = pd.DataFrame({"portfolio_value": [1.0, 1.1, 0.99, 1.20]})
        metrics = compute_financial_metrics(df)

        self.assertAlmostEqual(metrics["total_return"], 0.20)
        self.assertGreater(metrics["annualized_volatility"], 0.0)
        self.assertGreater(metrics["max_drawdown"], 0.0)
        self.assertIn("sortino", metrics)
        self.assertIn("calmar", metrics)

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


if __name__ == "__main__":
    unittest.main()
