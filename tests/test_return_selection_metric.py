import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import HRL_Trainer


class ReturnSelectionMetricTests(unittest.TestCase):
    def test_fixed_hrl_default_selection_uses_sharpe(self):
        cfg = SimpleNamespace(model_selection_metric="sharpe")
        metrics = {"sharpe": 0.7, "total_ret": 0.2, "max_dd": 0.4}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="joint")

        self.assertAlmostEqual(score, 0.7)

    def test_fixed_hrl_return_selection_uses_total_return(self):
        cfg = SimpleNamespace(model_selection_metric="return")
        metrics = {"sharpe": 0.7, "total_ret": 0.2, "max_dd": 0.4}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="joint")

        self.assertAlmostEqual(score, 0.2)

    def test_controller_default_selection_keeps_existing_risk_return_score(self):
        cfg = SimpleNamespace(
            controller_selection_metric="risk_return",
            controller_mdd_coef=2.0,
            controller_return_coef=0.5,
        )
        metrics = {"sharpe": 0.7, "total_ret": 0.3, "max_dd": 0.2}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="controller")

        self.assertAlmostEqual(score, -2.0 * 0.2 + 0.5 * 0.3)

    def test_controller_return_selection_uses_total_return(self):
        cfg = SimpleNamespace(controller_selection_metric="return")
        metrics = {"sharpe": 0.7, "total_ret": 0.3, "max_dd": 0.2}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="controller")

        self.assertAlmostEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
