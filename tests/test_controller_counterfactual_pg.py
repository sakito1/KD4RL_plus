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
    segment_count_band_penalty,
    segment_budget_allows_switch,
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

    def test_segment_count_band_penalty_only_applies_outside_target_range(self):
        self.assertEqual(segment_count_band_penalty(20, count_min=15, count_max=25), 0.0)
        self.assertEqual(segment_count_band_penalty(12, count_min=15, count_max=25), 3.0)
        self.assertEqual(segment_count_band_penalty(28, count_min=15, count_max=25), 3.0)

    def test_controller_reward_uses_counterfactual_uplift_without_count_penalty_inside_band(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.25,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
        )
        controlled = CounterfactualStats(
            log_return=0.08,
            max_drawdown=0.15,
            turnover=0.55,
            free_switch_count=3,
            segment_count=20,
        )
        reward = controller_reward(
            baseline,
            controlled,
            mdd_coef=2.0,
            return_coef=0.5,
            count_min=15,
            count_max=25,
            count_penalty_coef=0.5,
            switch_coef=0.0,
            turnover_coef=0.001,
        )
        expected = 2.0 * (0.25 - 0.15) + 0.5 * (0.08 - 0.10) - 0.001 * (0.55 - 0.40)
        self.assertAlmostEqual(reward, expected)

    def test_controller_reward_penalizes_segment_count_outside_band(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.25,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
        )
        controlled = CounterfactualStats(
            log_return=0.08,
            max_drawdown=0.15,
            turnover=0.55,
            free_switch_count=3,
            segment_count=28,
        )
        reward = controller_reward(
            baseline,
            controlled,
            mdd_coef=2.0,
            return_coef=0.5,
            count_min=15,
            count_max=25,
            count_penalty_coef=0.5,
            switch_coef=0.0,
            turnover_coef=0.001,
        )
        expected = (
            2.0 * (0.25 - 0.15)
            + 0.5 * (0.08 - 0.10)
            - 0.5 * (28 - 25)
            - 0.001 * (0.55 - 0.40)
        )
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
