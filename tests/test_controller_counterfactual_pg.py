import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.controller_pg import (
    CounterfactualStats,
    controller_pg_loss,
    controller_reward,
    segment_budget_allows_switch,
    max_switch_overflow_penalty,
)


class ControllerCounterfactualPGTests(unittest.TestCase):
    def test_segment_budget_keeps_rollout_within_max_segments(self):
        self.assertTrue(segment_budget_allows_switch(
            day_offset=10,
            rollout_len=400,
            current_segments=1,
            max_hold=40,
            max_segments=20,
        ))
        self.assertFalse(segment_budget_allows_switch(
            day_offset=10,
            rollout_len=400,
            current_segments=19,
            max_hold=40,
            max_segments=20,
        ))

    def test_max_switch_overflow_penalty_only_penalizes_too_many_switches(self):
        self.assertEqual(max_switch_overflow_penalty(20, max_switch_count=25), 0.0)
        self.assertEqual(max_switch_overflow_penalty(25, max_switch_count=25), 0.0)
        self.assertEqual(max_switch_overflow_penalty(28, max_switch_count=25), 9.0)

    def test_controller_reward_uses_return_uplift_without_mdd_reward(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.10,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
        )
        controlled = CounterfactualStats(
            log_return=0.08,
            max_drawdown=0.01,
            turnover=0.55,
            free_switch_count=3,
            segment_count=20,
        )
        reward = controller_reward(
            baseline,
            controlled,
            return_coef=1.0,
            max_switch_count=25,
            max_switch_penalty_coef=0.5,
        )
        expected = 0.08 - 0.10
        self.assertAlmostEqual(reward, expected)

    def test_controller_reward_penalizes_only_actual_switch_overflow(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.25,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
        )
        controlled = CounterfactualStats(
            log_return=0.18,
            max_drawdown=0.35,
            turnover=0.55,
            free_switch_count=3,
            segment_count=28,
        )
        reward = controller_reward(
            baseline,
            controlled,
            return_coef=1.0,
            max_switch_count=25,
            max_switch_penalty_coef=0.5,
        )
        expected = (0.18 - 0.10) - 0.5 * (28 - 25) ** 2
        self.assertAlmostEqual(reward, expected)

    def test_controller_pg_loss_uses_batch_normalized_counterfactual_reward(self):
        log_probs = torch.tensor([0.0, 1.0], requires_grad=True)
        rewards = torch.tensor([0.0, 2.0])
        entropy = torch.tensor([0.0, 0.0])

        loss, diagnostics = controller_pg_loss(log_probs, rewards, entropy, entropy_coef=0.0)
        loss.backward()

        self.assertAlmostEqual(diagnostics["reward_mean"], 1.0)
        self.assertGreater(log_probs.grad[0].item(), 0.0)
        self.assertLess(log_probs.grad[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
