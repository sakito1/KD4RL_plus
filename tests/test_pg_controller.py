import unittest

import torch

from pg_controller import PGControllerNet, RunningObjectiveBaseline
from pg_controller_experiment import (
    counterfactual_advantage_loss,
    compute_metrics,
    compute_reward_to_go,
    episode_objective,
    estimate_candidate_step_returns,
    mask_controller_hold_age,
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
            near_max_switch_rate=1.0, near_max_penalty=0.25,
            min_boundary_switch_rate=1.0, min_boundary_penalty=0.125,
        )
        self.assertAlmostEqual(penalty, 0.875)
        self.assertAlmostEqual(objective, metrics["sharpe"] - 0.875)

    def test_max_hold_violation_counts_action_that_exceeds_limit(self):
        self.assertFalse(violates_max_hold_after_hold(hold_age=29, max_hold=30))
        self.assertTrue(violates_max_hold_after_hold(hold_age=30, max_hold=30))

    def test_reward_to_go_matches_step_count(self):
        returns = compute_reward_to_go(
            [100.0, 101.0, 102.0, 101.0], gamma=1.0, standardize=False
        )
        self.assertEqual(returns.shape, (3,))
        self.assertGreater(returns[0], returns[-1])

    def test_counterfactual_step_return_does_not_mutate_env(self):
        class DummyEnv:
            pass

        env = DummyEnv()
        env.day = 1
        env.transaction_cost_pct = 0.0
        env.ratio = torch.tensor([[1.0, 1.10], [1.0, 0.90]])
        weights = torch.tensor([[0.5, 0.5]])
        hold_exec = torch.tensor([[1.0, 0.0]])
        switch_exec = torch.tensor([[0.0, 1.0]])

        returns, costs = estimate_candidate_step_returns(
            env, weights, hold_exec, switch_exec
        )
        self.assertEqual(returns.shape, (1, 3))
        self.assertEqual(costs.shape, (1, 3))
        self.assertGreater(returns[0, 0].item(), returns[0, 1].item())
        self.assertLess(returns[0, 2].item(), 0.0)
        self.assertEqual(env.day, 1)

    def test_auxiliary_advantage_loss_prefers_positive_switch_margin(self):
        logits = [torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)]
        advantages = [torch.tensor([0.01, -0.01])]
        loss = counterfactual_advantage_loss(logits, advantages)
        self.assertLess(loss.item(), 0.2)
        loss.backward()
        self.assertIsNotNone(logits[0].grad)

    def test_mask_hold_age_feature_only_zeros_first_port_state_column(self):
        obs = {
            "port_state": torch.tensor([[0.7, 0.1, -0.2, 0.3, 0.4, 0.5]]),
            "other": torch.tensor([1.0]),
        }
        masked = mask_controller_hold_age(obs)
        self.assertEqual(masked["port_state"][0, 0].item(), 0.0)
        self.assertAlmostEqual(masked["port_state"][0, 1].item(), 0.1)
        self.assertAlmostEqual(obs["port_state"][0, 0].item(), 0.7)
        self.assertIs(masked["other"], obs["other"])

    def test_running_baseline_is_scalar_not_learned_value_head(self):
        baseline = RunningObjectiveBaseline(momentum=0.5)
        self.assertEqual(baseline.advantage(2.0), 0.0)
        self.assertEqual(baseline.value, 2.0)
        self.assertAlmostEqual(baseline.advantage(4.0), 2.0)
        self.assertAlmostEqual(baseline.value, 3.0)


if __name__ == "__main__":
    unittest.main()
