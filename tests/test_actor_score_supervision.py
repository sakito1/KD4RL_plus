import os
import sys
import unittest
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.PPO_agent import HRL_PPO_Agent, actor_score_smooth_l1_loss
from env.PPO_env import PPO_Env


class _FixedPredHead(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32).view(1, -1, 1))
        self.called = False

    def forward(self, feat):
        self.called = True
        return self.values.expand(feat.shape[0], -1, -1)


class _FakeOuter(nn.Module):
    def __init__(self, num_assets=3):
        super().__init__()
        self.score_param = nn.Parameter(torch.tensor([[0.1], [0.2], [-0.1]], dtype=torch.float32))
        self.market_query = nn.Parameter(torch.ones(1, 1))
        self.w_proj = nn.Linear(num_assets, 1)
        self.v_head = nn.Linear(2, 1)
        self.pred_head = _FixedPredHead([0.03, -0.01, 0.02])

    def encode(self, state_subset, weights_drift):
        batch = weights_drift.shape[0]
        return self.score_param.unsqueeze(0).expand(batch, -1, -1)

    def get_dist(self, feat):
        mu = feat.squeeze(-1)
        std = torch.ones_like(mu)
        return Normal(mu, std)


class _FakeInner(nn.Module):
    def __init__(self, num_assets=3):
        super().__init__()
        self.score_param = nn.Parameter(torch.tensor([[0.1], [0.2], [-0.1]], dtype=torch.float32))
        self.alpha_query = nn.Linear(1, 1)
        self.value_head = nn.Linear(2, 1)
        self.pred_head = _FixedPredHead([0.01, -0.03, 0.04])

    def encode(self, inner_state, base_used, weights_drift):
        batch = base_used.shape[0]
        return self.score_param.unsqueeze(0).expand(batch, -1, -1)

    def get_dist(self, feat):
        mu = feat.squeeze(-1)
        std = torch.ones_like(mu)
        return Normal(mu, std)


class _FakeMonitor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))
        self.value = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.return_pred = nn.Parameter(torch.tensor([0.03], dtype=torch.float32))
        self.mdd_pred = nn.Parameter(torch.tensor([0.04], dtype=torch.float32))
        self.called = False

    def decision_stats(self, weights_drift, port_state,
                       switch_action=None, asset_state=None):
        self.called = True
        batch = weights_drift.shape[0]
        return {
            "policy_logit": self.logit.expand(batch),
            "value": self.value.expand(batch, 1),
            "hold_return_pred": self.return_pred.expand(batch),
            "hold_risk_pred": self.mdd_pred.expand(batch),
        }


def _agent_with_module(module_name, module):
    agent = HRL_PPO_Agent.__new__(HRL_PPO_Agent)
    agent.device = torch.device("cpu")
    agent.cfg = SimpleNamespace(
        outer_update_batch_size=1,
        inner_batch_size=1,
        outer_pred_coef=1.0,
        inner_pred_coef=1.0,
        controller_aux_return_coef=1.0,
        controller_aux_mdd_coef=1.0,
        clear_cuda_cache_on_update=False,
    )
    agent.net = SimpleNamespace(**{module_name: module})
    agent.clip_range = 0.2
    agent.vf_coef = 0.1
    agent.ent_coef = 0.0
    agent.max_grad_norm = None
    agent.min_clip = -5
    agent.max_clip = 5
    agent.mse_loss = nn.MSELoss()
    if module_name == "outer":
        agent.opt_out = torch.optim.Adam(module.parameters(), lr=1e-3)
    elif module_name == "inner":
        agent.opt_inn = torch.optim.Adam(module.parameters(), lr=1e-3)
    else:
        agent.opt_mon = torch.optim.Adam(module.parameters(), lr=1e-3)
    return agent


class ActorScoreSupervisionTests(unittest.TestCase):
    def test_relative_market_risk_tracks_drawdown_of_hold_to_pool_wealth(self):
        env = PPO_Env.__new__(PPO_Env)
        env.device = torch.device("cpu")
        env.stop_step = 3
        env.total_days = 4
        env.ratio = torch.tensor(
            [
                [1.10, 0.90, 1.00],
                [1.20, 1.10, 1.00],
            ],
            dtype=torch.float32,
        )

        _, relative_risk = env._future_portfolio_return_and_relative_market_drawdown(
            torch.tensor([1.0, 0.0]),
            start_day=0,
            horizon=2,
        )

        held_path = torch.tensor([1.10, 0.99])
        market_path = torch.tensor([(1.10 + 1.20) / 2, (0.99 + 1.32) / 2])
        relative_path = held_path / market_path
        expected = 1.0 - relative_path[1]
        torch.testing.assert_close(relative_risk, expected)

    def test_outer_supervision_uses_sampled_full_score_before_topk(self):
        mu = torch.tensor([[0.0, 0.4, -0.2]], requires_grad=True)
        std = torch.full_like(mu, 0.25)
        target = torch.tensor([[0.05, -0.03, 0.02]])

        torch.manual_seed(123)
        loss = actor_score_smooth_l1_loss(Normal(mu, std), target, squash=True)

        torch.manual_seed(123)
        expected_score = torch.tanh(Normal(mu, std).rsample())
        expected_loss = F.smooth_l1_loss(expected_score, target)
        mean_loss = F.smooth_l1_loss(torch.tanh(mu), target)

        torch.testing.assert_close(loss, expected_loss)
        self.assertGreater(abs(loss.item() - mean_loss.item()), 1e-4)

    def test_inner_supervision_uses_sampled_full_score(self):
        mu = torch.tensor([[0.0, 0.4, -0.2]], requires_grad=True)
        std = torch.full_like(mu, 0.25)
        target = torch.tensor([[0.01, -0.02, 0.03]])

        torch.manual_seed(321)
        loss = actor_score_smooth_l1_loss(Normal(mu, std), target, squash=False)

        torch.manual_seed(321)
        expected_score = Normal(mu, std).rsample()
        expected_loss = F.smooth_l1_loss(expected_score, target)
        mean_loss = F.smooth_l1_loss(mu, target)

        torch.testing.assert_close(loss, expected_loss)
        self.assertGreater(abs(loss.item() - mean_loss.item()), 1e-4)

    def test_outer_update_uses_return_pred_head_not_actor_score_for_supervision(self):
        outer = _FakeOuter()
        agent = _agent_with_module("outer", outer)
        data = {
            "adv_out": torch.tensor([1.0]),
            "ret_out": torch.tensor([0.2]),
            "outer_state": [torch.zeros(3, 2, 2)],
            "weights_drift": torch.tensor([[0.4, 0.3, 0.3]]),
            "act_out_raw": torch.tensor([[0.0, 0.1, -0.1]]),
            "logp_out": torch.tensor([0.0]),
            "outer_stock_return_target": torch.tensor([[0.03, 0.05, -0.02]]),
        }

        losses = agent._update_outer(data, torch.tensor([True]))

        expected = F.smooth_l1_loss(
            torch.tensor([[0.03, -0.01, 0.02]]),
            data["outer_stock_return_target"],
        )
        self.assertTrue(outer.pred_head.called)
        self.assertAlmostEqual(losses["pred"], expected.item(), places=6)

    def test_inner_update_uses_return_pred_head_not_actor_score_for_supervision(self):
        inner = _FakeInner()
        agent = _agent_with_module("inner", inner)
        data = {
            "adv_inn": torch.tensor([1.0]),
            "ret_inn": torch.tensor([0.2]),
            "inner_state": torch.zeros(1, 3, 2, 2),
            "inner_base_used": torch.tensor([[0.4, 0.3, 0.3]]),
            "inner_weights_drift": torch.tensor([[0.4, 0.3, 0.3]]),
            "base_used": torch.tensor([[0.4, 0.3, 0.3]]),
            "weights_drift": torch.tensor([[0.4, 0.3, 0.3]]),
            "act_inn_raw": torch.tensor([[0.0, 0.1, -0.1]]),
            "logp_inn": torch.tensor([0.0]),
            "inner_stock_return_target": torch.tensor([[0.01, -0.02, 0.03]]),
        }

        losses = agent._update_inner(data)

        expected = F.smooth_l1_loss(
            torch.tensor([[0.01, -0.03, 0.04]]),
            data["inner_stock_return_target"],
        )
        self.assertTrue(inner.pred_head.called)
        self.assertAlmostEqual(losses["pred"], expected.item(), places=6)

    def test_inner_update_scales_supervision_target(self):
        inner = _FakeInner()
        agent = _agent_with_module("inner", inner)
        agent.cfg.inner_pred_target_scale = 10.0
        data = {
            "adv_inn": torch.tensor([1.0]),
            "ret_inn": torch.tensor([0.2]),
            "inner_state": torch.zeros(1, 3, 2, 2),
            "inner_base_used": torch.tensor([[0.4, 0.3, 0.3]]),
            "inner_weights_drift": torch.tensor([[0.4, 0.3, 0.3]]),
            "base_used": torch.tensor([[0.4, 0.3, 0.3]]),
            "weights_drift": torch.tensor([[0.4, 0.3, 0.3]]),
            "act_inn_raw": torch.tensor([[0.0, 0.1, -0.1]]),
            "logp_inn": torch.tensor([0.0]),
            "inner_stock_return_target": torch.tensor([[0.01, -0.02, 0.03]]),
        }

        torch.manual_seed(11)
        losses = agent._update_inner(data)

        torch.manual_seed(11)
        expected = actor_score_smooth_l1_loss(
            torch.tensor([[0.01, -0.03, 0.04]]),
            data["inner_stock_return_target"] * 10.0,
            squash=False,
        )
        self.assertAlmostEqual(losses["pred"], expected.item(), places=6)

    def test_env_inner_supervision_target_is_next_day_log_return(self):
        env = PPO_Env.__new__(PPO_Env)
        env.device = torch.device("cpu")
        env.day = 1
        env.stop_step = 5
        env.total_days = 6
        env.num_stocks = 3
        env.max_hold = 4
        env.min_hold = 2
        env.transaction_cost_pct = 0.0
        env.risk_gamma = 5.0
        env.reward_scale_portfolio = 1.0
        env.reward_scale_base = 1.0
        env.reward_scale_inner = 1.0
        env.reward_scale_controller = 1.0
        env.controller_sup_enabled = False
        env.portfolio_value = torch.tensor(1.0)
        env.prev_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32)
        env.prev_base_weight = env.prev_weights.clone()
        env.t_held = 0
        env.peak_value = 1.0
        env.segment_init_value = 1.0
        env.cumulative_alpha = 0.0
        env.cumulative_risk = 0.0
        env.all_dates = pd.date_range("2020-01-01", periods=8)
        env.ratio = torch.tensor(
            [
                [1.00, 1.10, 1.50, 1.20, 1.00],
                [1.00, 0.90, 0.80, 1.10, 1.00],
                [1.00, 1.05, 1.10, 1.00, 1.00],
            ],
            dtype=torch.float32,
        )
        env._get_observation = lambda: {}

        _, _, _, info = env.step(
            torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
            torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32),
            is_switch=True,
        )

        expected = torch.log(env.ratio[:, 1].clamp_min(1e-8))
        torch.testing.assert_close(info["inner_stock_return_target"], expected)

    def test_env_controller_aux_targets_cover_remaining_holding_period(self):
        env = PPO_Env.__new__(PPO_Env)
        env.device = torch.device("cpu")
        env.day = 1
        env.stop_step = 5
        env.total_days = 6
        env.num_stocks = 3
        env.max_hold = 4
        env.min_hold = 2
        env.transaction_cost_pct = 0.0
        env.risk_gamma = 5.0
        env.reward_scale_portfolio = 1.0
        env.reward_scale_base = 1.0
        env.reward_scale_inner = 1.0
        env.reward_scale_controller = 1.0
        env.controller_sup_enabled = False
        env.controller_switch_advantage_enabled = True
        env.portfolio_value = torch.tensor(1.0)
        env.prev_weights = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        env.prev_base_weight = env.prev_weights.clone()
        env.t_held = 2
        env.peak_value = 1.0
        env.segment_init_value = 1.0
        env.cumulative_alpha = 0.0
        env.cumulative_risk = 0.0
        env.all_dates = pd.date_range("2020-01-01", periods=8)
        env.ratio = torch.tensor(
            [
                [1.00, 0.80, 1.25, 1.00, 1.00],
                [1.00, 1.10, 0.90, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ],
            dtype=torch.float32,
        )
        env._get_observation = lambda: {}

        _, _, _, info = env.step(
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32),
            outer_action=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            is_switch=False,
        )

        expected_return = torch.log(torch.tensor(0.995, dtype=torch.float32))
        expected_mdd = torch.tensor(1.0 - 0.95 / ((0.80 + 1.10 + 1.00) / 3.0))
        expected_switch_advantage = torch.log(torch.tensor(1.0, dtype=torch.float32)) - expected_return
        torch.testing.assert_close(info["controller_hold_return_target"], expected_return)
        torch.testing.assert_close(info["controller_hold_mdd_target"], expected_mdd)
        torch.testing.assert_close(info["controller_switch_advantage"], expected_switch_advantage)

    def test_env_controller_aux_targets_use_actual_drifted_holdings(self):
        env = PPO_Env.__new__(PPO_Env)
        env.device = torch.device("cpu")
        env.day = 1
        env.stop_step = 5
        env.total_days = 6
        env.num_stocks = 3
        env.max_hold = 4
        env.min_hold = 2
        env.transaction_cost_pct = 0.0
        env.risk_gamma = 5.0
        env.reward_scale_portfolio = 1.0
        env.reward_scale_base = 1.0
        env.reward_scale_inner = 1.0
        env.reward_scale_controller = 1.0
        env.controller_sup_enabled = False
        env.controller_switch_advantage_enabled = False
        env.portfolio_value = torch.tensor(1.0)
        env.prev_weights = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        env.prev_base_weight = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        env.t_held = 2
        env.peak_value = 1.0
        env.segment_init_value = 1.0
        env.cumulative_alpha = 0.0
        env.cumulative_risk = 0.0
        env.all_dates = pd.date_range("2020-01-01", periods=8)
        env.ratio = torch.tensor(
            [
                [1.00, 1.20, 1.10, 1.00, 1.00],
                [1.00, 0.80, 0.90, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ],
            dtype=torch.float32,
        )
        env._get_observation = lambda: {}

        day_before_step = env.day
        held_before_step = env.t_held
        r_past = env.ratio[:, day_before_step - 1]
        actual_drift = env._normalize(env.prev_weights * r_past)
        base_drift = env._normalize(env.prev_base_weight * r_past)
        horizon = env.max_hold - held_before_step
        expected_return, expected_mdd = (
            env._future_portfolio_return_and_relative_market_drawdown(
                actual_drift,
                day_before_step,
                horizon,
            )
        )
        base_return, base_mdd = (
            env._future_portfolio_return_and_relative_market_drawdown(
                base_drift,
                day_before_step,
                horizon,
            )
        )

        _, _, _, info = env.step(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            outer_action=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            is_switch=False,
        )

        torch.testing.assert_close(info["controller_hold_return_target"], expected_return)
        torch.testing.assert_close(info["controller_hold_mdd_target"], expected_mdd)
        self.assertFalse(torch.isclose(expected_return, base_return))
        self.assertFalse(torch.isclose(expected_mdd, base_mdd))

    def test_env_skips_switch_advantage_by_default_for_fast_hrl_steps(self):
        env = PPO_Env.__new__(PPO_Env)
        env.device = torch.device("cpu")
        env.day = 1
        env.stop_step = 5
        env.total_days = 6
        env.num_stocks = 3
        env.max_hold = 4
        env.min_hold = 2
        env.transaction_cost_pct = 0.0
        env.risk_gamma = 5.0
        env.reward_scale_portfolio = 1.0
        env.reward_scale_base = 1.0
        env.reward_scale_inner = 1.0
        env.reward_scale_controller = 1.0
        env.controller_sup_enabled = False
        env.portfolio_value = torch.tensor(1.0)
        env.prev_weights = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32)
        env.prev_base_weight = env.prev_weights.clone()
        env.t_held = 2
        env.peak_value = 1.0
        env.segment_init_value = 1.0
        env.cumulative_alpha = 0.0
        env.cumulative_risk = 0.0
        env.all_dates = pd.date_range("2020-01-01", periods=8)
        env.ratio = torch.tensor(
            [
                [1.00, 0.80, 1.25, 1.00, 1.00],
                [1.00, 1.10, 0.90, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ],
            dtype=torch.float32,
        )
        env._get_observation = lambda: {}

        _, _, _, info = env.step(
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32),
            torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32),
            outer_action=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
            is_switch=False,
        )

        expected_return = torch.log(torch.tensor(0.995, dtype=torch.float32))
        expected_mdd = torch.tensor(1.0 - 0.95 / ((0.80 + 1.10 + 1.00) / 3.0))
        torch.testing.assert_close(info["controller_hold_return_target"], expected_return)
        torch.testing.assert_close(info["controller_hold_mdd_target"], expected_mdd)
        torch.testing.assert_close(info["controller_switch_advantage"], torch.tensor(0.0))

    def test_monitor_update_trains_remaining_hold_return_and_mdd_heads(self):
        monitor = _FakeMonitor()
        agent = _agent_with_module("mon", monitor)
        data = {
            "ssm": {
                "z": torch.zeros(1, 3, 4),
                "h": torch.zeros(1, 3, 4),
                "p": torch.zeros(1, 3),
                "q_bear": torch.zeros(1, 3),
                "q_bull": torch.zeros(1, 3),
            },
            "outer_state": [torch.zeros(3, 15, 7)],
            "weights_drift": torch.tensor([[0.5, 0.5, 0.0]]),
            "port_state": torch.zeros(1, 6),
            "act_out": torch.tensor([[0.4, 0.4, 0.2]]),
            "act_mon": torch.tensor([1]),
            "logp_mon": torch.tensor([0.0]),
            "adv_mon": torch.tensor([1.0]),
            "ret_mon": torch.tensor([0.2]),
            "controller_hold_return_target": torch.tensor([-0.02]),
            "controller_hold_mdd_target": torch.tensor([0.10]),
        }

        losses = agent._update_monitor(data, torch.tensor([True]))

        expected_aux_return = F.smooth_l1_loss(torch.tensor([0.03]), data["controller_hold_return_target"])
        expected_aux_mdd = F.smooth_l1_loss(torch.tensor([0.04]), data["controller_hold_mdd_target"])
        self.assertTrue(monitor.called)
        self.assertAlmostEqual(losses["aux_return"], expected_aux_return.item(), places=6)
        self.assertAlmostEqual(losses["aux_mdd"], expected_aux_mdd.item(), places=6)


if __name__ == "__main__":
    unittest.main()
