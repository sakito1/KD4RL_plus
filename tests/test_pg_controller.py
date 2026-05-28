import unittest

import torch

from pg_controller import PGControllerNet, RunningObjectiveBaseline
from pg_controller_experiment import (
    compute_metrics,
    episode_objective,
    violates_max_hold_after_hold,
    violation_penalty,
)


class PGControllerTests(unittest.TestCase):
    def test_policy_only_shapes(self):
        net = PGControllerNet()
        n = 5
        obs = {
            "ssm": {
                "z": torch.randn(1, n, 16),
                "h": torch.randn(1, n, 16),
            },
            "weights_drift": torch.softmax(torch.randn(1, n), dim=1),
            "base_drift": torch.softmax(torch.randn(1, n), dim=1),
            "candidate_switch_base": torch.softmax(torch.randn(1, n), dim=1),
            "port_state": torch.randn(1, 6),
            "held_p": torch.randn(1),
            "candidate_costs": torch.randn(1, 3),
        }
        logits = net(obs)
        self.assertEqual(logits.shape, (1, 2))
        action, log_prob, entropy = net.act(obs, deterministic=False)
        self.assertEqual(action.shape, (1,))
        loss = -log_prob.sum() - 0.01 * entropy.mean()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))

    def test_episode_objective_uses_sharpe_minus_violation_counts(self):
        history = [100.0, 101.0, 100.5, 102.0]
        metrics = compute_metrics(history)
        objective, returned_metrics, penalty = episode_objective(
            history, early_count=2, long_count=1, lambda_min=1.0, lambda_max=3.0
        )
        self.assertAlmostEqual(metrics["sharpe"], returned_metrics["sharpe"])
        self.assertAlmostEqual(penalty, 5.0)
        self.assertAlmostEqual(objective, metrics["sharpe"] - 5.0)
        self.assertAlmostEqual(violation_penalty(2, 1, 1.0, 3.0), 5.0)
        objective, _, penalty = episode_objective(
            history, early_count=0, long_count=0, lambda_min=1.0, lambda_max=1.0,
            scheduled_switch_rate=1.0, schedule_penalty=0.5,
        )
        self.assertAlmostEqual(penalty, 0.5)
        self.assertAlmostEqual(objective, metrics["sharpe"] - 0.5)

    def test_max_hold_violation_counts_action_that_exceeds_limit(self):
        self.assertFalse(violates_max_hold_after_hold(hold_age=29, max_hold=30))
        self.assertTrue(violates_max_hold_after_hold(hold_age=30, max_hold=30))

    def test_running_baseline_is_scalar_not_learned_value_head(self):
        baseline = RunningObjectiveBaseline(momentum=0.5)
        self.assertEqual(baseline.advantage(2.0), 0.0)
        self.assertEqual(baseline.value, 2.0)
        self.assertAlmostEqual(baseline.advantage(4.0), 2.0)
        self.assertAlmostEqual(baseline.value, 3.0)


if __name__ == "__main__":
    unittest.main()
