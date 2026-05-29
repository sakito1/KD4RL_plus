import unittest

import torch

from agent.DQN_monitor import (
    ConditionalUtilityAgent,
    DQNMonitorAgent,
    DQNReplayBuffer,
    EmbMonitorQNet,
    UtilityReplayBuffer,
)
from dqn_monitor_experiment import (
    build_candidates,
    execute_step,
    long_hold_penalty,
    realized_candidate_targets,
)


def make_observation(num_assets=3):
    weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float32)
    return {
        "ssm": {
            "z": torch.randn(1, num_assets, 16),
            "h": torch.randn(1, num_assets, 16),
        },
        "weights_drift": weights,
        "base_drift": torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32),
        "outer_state": torch.zeros(1, num_assets, 2, 1),
        "inner_state": torch.zeros(1, num_assets, 2, 1),
        "port_state": torch.zeros(1, 6),
        "held_p": torch.tensor([0.4], dtype=torch.float32),
    }


class FakeOuter:
    def pi(self, outer_state, weights, deterministic=True):
        switch = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
        return switch, None, None, None, None

    def value(self, outer_state, weights):
        score = (weights * torch.tensor([[0.0, 1.0, 2.0]])).sum(dim=1, keepdim=True)
        return score


class FakeInner:
    def build_inner_action_simple(self, inner_state, base, weights, alpha, deterministic=True):
        return base, None, None, None, None


class FakeModel:
    outer = FakeOuter()
    inner = FakeInner()


class FakeEnv:
    transaction_cost_pct = 0.01
    max_hold = 5
    t_held = 6
    day = 0
    ratio = torch.tensor([[1.01], [1.02], [0.99]], dtype=torch.float32)

    def __init__(self):
        self.step_calls = 0

    def step(self, weights, base, outer_action=None, is_switch=False,
             calculate_outer_reward=True):
        self.step_calls += 1
        info = {"rewards": {"turnover": 0.0, "transaction_cost": 0.0},
                "portfolio_value": 1000.0}
        return make_observation(), 0.0, False, info


class ControllerStateTests(unittest.TestCase):
    def test_extended_q_state_has_expected_shape_and_updates(self):
        obs = make_observation()
        obs["candidate_switch_base"] = torch.tensor([[0.1, 0.2, 0.7]])
        obs["candidate_costs"] = torch.zeros(1, 3)
        net = EmbMonitorQNet()
        encoded = net.encode_state(
            obs["ssm"]["z"], obs["ssm"]["h"], obs["weights_drift"],
            obs["base_drift"], obs["candidate_switch_base"], obs["port_state"],
            obs["held_p"], obs["candidate_costs"],
        )
        self.assertEqual(encoded.shape, (1, 138))
        self.assertEqual(net(
            obs["ssm"]["z"], obs["ssm"]["h"], obs["weights_drift"],
            obs["base_drift"], obs["candidate_switch_base"], obs["port_state"],
            obs["held_p"], obs["candidate_costs"],
        ).shape, (1, 2))
        replay = DQNReplayBuffer(4)
        replay.store(obs, 0, 0.1, obs, False)
        replay.store(obs, 1, -0.1, obs, True)
        agent = DQNMonitorAgent(torch.device("cpu"))
        self.assertIsInstance(agent.update(replay, 2), float)

    def test_long_hold_penalty_is_soft_after_target_only(self):
        self.assertEqual(long_hold_penalty(5, 5), 0.0)
        self.assertEqual(long_hold_penalty(4, 5), 0.0)
        self.assertAlmostEqual(long_hold_penalty(10, 5), 1.0)

    def test_candidate_build_is_non_mutating_and_reward_is_utility_difference(self):
        env = FakeEnv()
        obs, candidate = build_candidates(
            env, FakeModel(), make_observation(), lambda_cost=2.0, lambda_long=0.5
        )
        self.assertEqual(env.step_calls, 0)
        self.assertIn("candidate_switch_base", obs)
        self.assertEqual(obs["candidate_costs"].shape, (1, 3))
        _, reward, _, _, switch_advantage = execute_step(env, candidate, action=1)
        self.assertEqual(env.step_calls, 1)
        self.assertAlmostEqual(
            reward, candidate["utility_switch"] - candidate["utility_hold"]
        )
        self.assertAlmostEqual(reward, switch_advantage)

    def test_conditional_targets_are_net_returns_and_critic_updates(self):
        env = FakeEnv()
        obs, candidate = build_candidates(env, FakeModel(), make_observation())
        targets = realized_candidate_targets(env, candidate, lambda_long=0.5)
        self.assertEqual(targets.shape, (1, 2))
        gross_hold = 0.7 * 1.01 + 0.2 * 1.02 + 0.1 * 0.99
        expected_hold = torch.log(torch.tensor(gross_hold * (1 - candidate["cost_hold"]))).item()
        expected_hold -= 0.5 * candidate["long_penalty_hold"]
        self.assertAlmostEqual(targets[0, 0].item(), expected_hold, places=6)
        replay = UtilityReplayBuffer(4)
        replay.store(obs, targets)
        replay.store(obs, targets)
        utility = ConditionalUtilityAgent(torch.device("cpu"))
        self.assertIsInstance(utility.update(replay, 2), float)


if __name__ == "__main__":
    unittest.main()
