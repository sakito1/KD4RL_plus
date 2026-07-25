import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch.distributions import Categorical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.controller_pg import (
    CounterfactualStats,
    controller_pg_loss,
    controller_reward,
    overflow_switch_policy_loss,
    segment_budget_allows_switch,
    max_switch_overflow_penalty,
)
from Train.PPO_train import HRL_Trainer, PhaseSpec


class _FakeControllerMonitor:
    def __init__(self, policy_logit=0.25):
        self.policy_logit = float(policy_logit)

    def train(self):
        return self

    def decision_stats(self, weights_drift, port_state,
                       switch_action=None, asset_state=None):
        return {
            "policy_logit": torch.tensor([self.policy_logit], requires_grad=True),
            "exit_prob": torch.sigmoid(torch.tensor([self.policy_logit], requires_grad=True)),
            "hold_return_pred": torch.tensor([0.03], requires_grad=True),
            "hold_risk_pred": torch.tensor([0.04], requires_grad=True),
            "switch_advantage_pred": torch.tensor([0.02], requires_grad=True),
        }


class _FakeOptimizer:
    def __init__(self):
        self.zero_grad_called = False
        self.step_called = False
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self, set_to_none=True):
        self.zero_grad_called = True
        self.zero_grad_calls += 1

    def step(self):
        self.step_called = True
        self.step_calls += 1


class _FakePoolEnv:
    total_days = 1000
    idx_map = {"train": list(range(100, 900))}

    def _build_fixed_train_pool(self, raw_indices, total_days, episode_len, stride_days, start_offsets=None):
        return list(range(20))


class _FakeTestWindowEnv:
    total_days = 1000
    idx_map = {"test": list(range(700, 900))}


class _FakeSwitchAdvEnv:
    day = 7
    t_held = 10
    max_hold = 30
    transaction_cost_pct = 0.01

    def _normalize(self, weights):
        weights = weights.flatten()
        return weights / weights.sum().clamp_min(1e-8)

    def _future_portfolio_return_and_max_drawdown(self, weights, start_day, horizon):
        del start_day, horizon
        weights = self._normalize(weights)
        return torch.sum(weights * torch.tensor([0.04, -0.02])), torch.tensor(0.0)


class _FakeAlwaysSwitchOuter:
    def eval(self):
        return self

    def pi(self, outer_state, weights_drift, deterministic=True):
        del outer_state, deterministic
        return weights_drift.clone(), None, None, None, None


class _FakeAlwaysSwitchInner:
    def eval(self):
        return self


class _FakeAlwaysSwitchAgent:
    def __init__(self):
        self.net = SimpleNamespace(
            outer=_FakeAlwaysSwitchOuter(),
            inner=_FakeAlwaysSwitchInner(),
            mon=_FakeControllerMonitor(policy_logit=10.0),
        )

    def get_action(self, obs, mode="eval", force_switch=None, force_inner_zero=False, force_locked=True):
        del mode, force_inner_zero, force_locked
        act_out = obs["base_drift"].clone()
        act_mon = torch.tensor([int(force_switch or 0)])
        return {
            "act_out": act_out,
            "act_mon": act_mon,
            "base_used": act_out if bool(act_mon.item()) else obs["base_drift"].clone(),
        }


class _FakeControllerPGWindowEnv:
    transaction_cost_pct = 0.0

    def __init__(self, length=6):
        self.length = int(length)
        self.step_idx = 0
        self.portfolio_value = torch.tensor(1.0)
        self.switches = []

    def reset_at(self, start_idx, stop_idx):
        del start_idx
        self.length = max(1, int(stop_idx))
        self.step_idx = 0
        self.portfolio_value = torch.tensor(1.0)
        self.switches = []
        return self._obs()

    def _obs(self):
        weights = torch.tensor([[0.5, 0.5]])
        return {
            "weights_drift": weights.clone(),
            "base_drift": weights.clone(),
            "outer_state": torch.zeros(1, 2),
            "port_state": torch.zeros(1, 2),
            "ssm": {
                "z": torch.zeros(1, 2),
                "h": torch.zeros(1, 2),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
        }

    def step(self, weights_exec, base_used, outer_action=None, is_switch=False):
        del weights_exec, base_used, outer_action
        self.switches.append(bool(is_switch))
        self.step_idx += 1
        self.portfolio_value = torch.tensor(1.0 + 0.01 * self.step_idx)
        done = self.step_idx >= self.length
        info = {
            "portfolio_value": float(self.portfolio_value.item()),
            "controller_hold_return_target": torch.tensor([0.0]),
            "controller_hold_mdd_target": torch.tensor([0.0]),
            "controller_switch_advantage": torch.tensor([0.0]),
        }
        return self._obs(), torch.tensor(0.0), done, info


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

    def test_controller_reward_return_uplift_mode_preserves_current_behavior(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.10,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
            trading_days=252,
        )
        controlled = CounterfactualStats(
            log_return=0.20,
            max_drawdown=0.50,
            turnover=0.55,
            free_switch_count=3,
            segment_count=10,
            trading_days=252,
        )

        reward = controller_reward(
            baseline,
            controlled,
            reward_mode="return_uplift",
            return_coef=2.0,
            max_switch_count=30,
            max_switch_penalty_coef=0.0,
        )

        self.assertAlmostEqual(reward, 2.0 * (0.20 - 0.10))

    def test_controller_reward_relative_cr_mode_uses_calmar_uplift(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.20,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
            trading_days=252,
        )
        controlled = CounterfactualStats(
            log_return=0.18,
            max_drawdown=0.12,
            turnover=0.55,
            free_switch_count=3,
            segment_count=34,
            trading_days=252,
        )

        reward = controller_reward(
            baseline,
            controlled,
            reward_mode="relative_cr",
            max_switch_count=30,
            max_switch_penalty_coef=0.5,
        )

        expected_cr_uplift = (0.18 / 0.12) - (0.10 / 0.20)
        expected_penalty = 0.5 * ((34 - 30) / 30) ** 2
        self.assertAlmostEqual(reward, expected_cr_uplift - expected_penalty)

    def test_controller_reward_relative_return_mdd_mode_uses_weighted_relative_uplifts(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.20,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
        )
        controlled = CounterfactualStats(
            log_return=0.15,
            max_drawdown=0.16,
            turnover=0.55,
            free_switch_count=3,
            segment_count=34,
        )

        reward = controller_reward(
            baseline,
            controlled,
            reward_mode="relative_return_mdd",
            return_coef=1.0,
            mdd_coef=0.3,
            max_switch_count=30,
            max_switch_penalty_coef=0.5,
        )

        expected_return_uplift = (0.15 - 0.10) / 0.10
        expected_mdd_uplift = (0.20 - 0.16) / 0.20
        expected_penalty = 0.5 * ((34 - 30) / 30) ** 2
        self.assertAlmostEqual(
            reward,
            expected_return_uplift + 0.3 * expected_mdd_uplift - expected_penalty,
        )

    def test_controller_reward_relative_downside_mdd_keeps_return_dominant(self):
        baseline = CounterfactualStats(
            log_return=0.10,
            max_drawdown=0.20,
            turnover=0.40,
            free_switch_count=0,
            segment_count=10,
            downside_loss=0.30,
        )
        controlled = CounterfactualStats(
            log_return=0.16,
            max_drawdown=0.15,
            turnover=0.55,
            free_switch_count=3,
            segment_count=36,
            downside_loss=0.21,
        )

        reward = controller_reward(
            baseline,
            controlled,
            reward_mode="relative_downside_mdd",
            return_coef=1.0,
            downside_coef=0.5,
            mdd_coef=0.3,
            max_switch_count=30,
            max_switch_penalty_coef=0.1,
        )

        expected_return_uplift = (0.16 - 0.10) / 0.10
        expected_downside_uplift = (0.30 - 0.21) / 0.30
        expected_mdd_uplift = (0.20 - 0.15) / 0.20
        expected_penalty = 0.1 * ((36 - 30) / 30) ** 2
        self.assertAlmostEqual(
            reward,
            expected_return_uplift
            + 0.5 * expected_downside_uplift
            + 0.3 * expected_mdd_uplift
            - expected_penalty,
        )

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
        expected = (0.18 - 0.10) - 0.5 * ((28 - 25) / 25) ** 2
        self.assertAlmostEqual(reward, expected)

    def test_controller_reward_uses_return_uplift_and_max_switch_30_overflow(self):
        baseline = CounterfactualStats(
            log_return=0.05,
            max_drawdown=0.20,
            turnover=0.40,
            free_switch_count=0,
            segment_count=20,
        )
        controlled = CounterfactualStats(
            log_return=0.13,
            max_drawdown=0.40,
            turnover=1.20,
            free_switch_count=18,
            segment_count=34,
        )

        reward = controller_reward(
            baseline,
            controlled,
            return_coef=1.0,
            max_switch_count=30,
            max_switch_penalty_coef=0.001,
        )

        self.assertAlmostEqual(reward, (0.13 - 0.05) - 0.001 * ((34 - 30) / 30) ** 2)

    def test_counterfactual_stats_records_cumulative_downside_loss(self):
        stats = HRL_Trainer._counterfactual_stats(
            [100.0, 90.0, 99.0, 79.2],
            turnover_sum=0.2,
            free_switch_count=2,
            segment_count=3,
        )

        expected_downside = -torch.log(torch.tensor(90.0 / 100.0)).item()
        expected_downside += -torch.log(torch.tensor(79.2 / 99.0)).item()
        self.assertAlmostEqual(stats.downside_loss, expected_downside, places=6)

    def test_controller_pg_exec_weights_can_bypass_inner_actor(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        called = {"inner": False}

        def fake_inner_exec(obs, base_used, weights_drift):
            del obs, weights_drift
            called["inner"] = True
            return base_used + 1.0

        trainer._deterministic_inner_exec = fake_inner_exec
        base_used = torch.tensor([[0.25, 0.75]])
        weights_drift = torch.tensor([[0.50, 0.50]])

        disabled = trainer._controller_exec_weights(
            obs={},
            base_used=base_used,
            weights_drift=weights_drift,
            disable_inner=True,
        )
        enabled = trainer._controller_exec_weights(
            obs={},
            base_used=base_used,
            weights_drift=weights_drift,
            disable_inner=False,
        )

        self.assertTrue(torch.equal(disabled, base_used))
        self.assertTrue(called["inner"])
        self.assertTrue(torch.equal(enabled, base_used + 1.0))

    def test_no_hold_constraints_do_not_hard_cap_switches_at_penalty_threshold(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_max_switches=40,
            min_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=500,
            duration=1,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
            current_segments=40,
            rollout_len=600,
        )

        self.assertIsNone(force_switch)
        self.assertFalse(force_locked)

    def test_optional_hard_max_switches_force_hold_after_cap(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_max_switches=40,
            controller_hard_max_switches=40,
            min_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=500,
            duration=1,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
            current_segments=40,
            rollout_len=600,
        )

        self.assertEqual(force_switch, 0)
        self.assertTrue(force_locked)

    def test_controller_pg_window_enforces_hard_max_switches_during_sampling(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_hard_max_switches=2,
            controller_train_max_hold=0,
            controller_rollout_len=6,
            controller_pg_disable_inner=True,
            controller_train_record_max_duration=0,
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_aux_switch_adv_coef=0.0,
            controller_local_adv_coef=0.0,
            controller_compute_switch_advantage=False,
            min_hold=30,
            max_hold=30,
        )
        trainer.device = torch.device("cpu")
        trainer.agent = _FakeAlwaysSwitchAgent()
        trainer._controller_exec_weights = (
            lambda obs, base_used, weights_drift, disable_inner=False: base_used.clone()
        )
        env = _FakeControllerPGWindowEnv(length=6)

        result = trainer._run_controller_pg_window(env, 0, 6, fixed_cycle=30)

        self.assertEqual(result["stats"].segment_count, 2)
        self.assertEqual(result["stats"].free_switch_count, 1)
        self.assertEqual(env.switches, [True, True, False, False, False, False])

    def test_stride_decision_mode_only_allows_decisions_on_stride_days(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="stride",
            controller_decision_stride=15,
            controller_eval_decision_stride=10,
            max_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=14,
            duration=14,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
        )
        self.assertEqual(force_switch, 0)
        self.assertTrue(force_locked)

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=15,
            duration=15,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
        )
        self.assertIsNone(force_switch)
        self.assertFalse(force_locked)

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=30,
            duration=30,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
        )
        self.assertEqual(force_switch, 1)
        self.assertTrue(force_locked)

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=10,
            duration=10,
            is_train=False,
            switch_schedule=None,
            fixed_cycle=None,
        )
        self.assertIsNone(force_switch)
        self.assertFalse(force_locked)

    def test_daily_no_hold_constraints_train_max_hold_zero_disables_forced_switch(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_train_max_hold=0,
            max_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=30,
            duration=30,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
            rollout_len=600,
        )

        self.assertIsNone(force_switch)
        self.assertFalse(force_locked)

    def test_daily_no_hold_constraints_positive_train_max_hold_forces_switch(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_train_max_hold=30,
            max_hold=60,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=30,
            duration=30,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
            rollout_len=600,
        )

        self.assertEqual(force_switch, 1)
        self.assertTrue(force_locked)

    def test_controller_eval_max_hold_can_disable_eval_only_forced_switch(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="daily",
            controller_eval_max_hold=0,
            max_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=30,
            duration=30,
            is_train=False,
            switch_schedule=None,
            fixed_cycle=None,
        )

        self.assertIsNone(force_switch)
        self.assertFalse(force_locked)

        trainer.cfg.controller_eval_max_hold = -1
        force_switch, force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=30,
            duration=30,
            is_train=False,
            switch_schedule=None,
            fixed_cycle=None,
        )
        self.assertEqual(force_switch, 1)
        self.assertTrue(force_locked)

    def test_controller_training_max_hold_can_disable_training_only_forced_switch(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)

        trainer.cfg = SimpleNamespace(max_hold=30)
        self.assertEqual(
            trainer._controller_train_max_hold(fixed_cycle=40, rollout_len=600),
            30,
        )

        trainer.cfg = SimpleNamespace(max_hold=30, controller_train_max_hold=120)
        self.assertEqual(
            trainer._controller_train_max_hold(fixed_cycle=40, rollout_len=600),
            120,
        )

        trainer.cfg = SimpleNamespace(max_hold=30, controller_train_max_hold=0)
        self.assertEqual(
            trainer._controller_train_max_hold(fixed_cycle=40, rollout_len=600),
            601,
        )

    def test_controller_train_record_max_duration_limits_pg_records_to_eval_window(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)

        trainer.cfg = SimpleNamespace(controller_train_record_max_duration=30)
        self.assertTrue(trainer._controller_should_record_train_decision(duration=29))
        self.assertFalse(trainer._controller_should_record_train_decision(duration=30))
        self.assertFalse(trainer._controller_should_record_train_decision(duration=300))

        trainer.cfg = SimpleNamespace(controller_train_record_max_duration=0)
        self.assertTrue(trainer._controller_should_record_train_decision(duration=300))

    def test_eval_decision_mode_can_override_training_fixed_window(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            controller_no_hold_constraints=True,
            controller_decision_mode="fixed_window",
            controller_eval_decision_mode="daily",
            controller_fixed_decision_window=5,
            max_hold=30,
        )
        spec = PhaseSpec(
            use_schedule=False,
            inner_always_zero=False,
            monitor_always_forced=False,
            mask_monitor_update=False,
            use_hold_constraints=True,
        )

        train_force_switch, train_force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=10,
            duration=10,
            is_train=True,
            switch_schedule=None,
            fixed_cycle=None,
        )
        eval_force_switch, eval_force_locked = trainer._compute_force_switch_locked(
            spec=spec,
            phase="joint",
            step_idx=10,
            duration=10,
            is_train=False,
            switch_schedule=None,
            fixed_cycle=None,
        )

        self.assertEqual(train_force_switch, 0)
        self.assertTrue(train_force_locked)
        self.assertIsNone(eval_force_switch)
        self.assertFalse(eval_force_locked)

    def test_controller_pg_loss_uses_raw_counterfactual_reward_without_batch_normalization(self):
        log_probs = torch.tensor([0.0, 1.0], requires_grad=True)
        rewards = torch.tensor([0.0, 2.0])
        entropy = torch.tensor([0.0, 0.0])

        loss, diagnostics = controller_pg_loss(log_probs, rewards, entropy, entropy_coef=0.0)
        loss.backward()

        self.assertAlmostEqual(diagnostics["reward_mean"], 1.0)
        self.assertAlmostEqual(diagnostics["reward_std"], 1.0)
        self.assertAlmostEqual(diagnostics["reward_min"], 0.0)
        self.assertAlmostEqual(diagnostics["reward_max"], 2.0)
        self.assertAlmostEqual(diagnostics["reward_abs_mean"], 1.0)
        self.assertAlmostEqual(diagnostics["policy_abs_loss"], 1.0)
        self.assertAlmostEqual(diagnostics["loss_abs"], 1.0)
        torch.testing.assert_close(loss.detach(), torch.tensor(-1.0))
        torch.testing.assert_close(log_probs.grad, torch.tensor([-0.0, -1.0]))

    def test_controller_pg_loss_reports_policy_and_entropy_loss_scales(self):
        log_probs = torch.tensor([0.0, 1.0], requires_grad=True)
        rewards = torch.tensor([0.0, 2.0])
        entropy = torch.tensor([0.5, 0.5])

        _, diagnostics = controller_pg_loss(log_probs, rewards, entropy, entropy_coef=0.01)

        self.assertIn("policy_loss", diagnostics)
        self.assertIn("entropy_loss", diagnostics)
        self.assertAlmostEqual(diagnostics["entropy_loss"], -0.005)

    def test_controller_pg_loss_can_train_value_baseline(self):
        log_probs = torch.tensor([0.0, 1.0], requires_grad=True)
        rewards = torch.tensor([0.0, 2.0])
        values = torch.tensor([1.0, 1.0], requires_grad=True)
        entropy = torch.tensor([0.0, 0.0])

        loss, diagnostics = controller_pg_loss(
            log_probs,
            rewards,
            entropy,
            entropy_coef=0.0,
            values=values,
            value_coef=0.5,
            normalize_value_advantage=False,
        )
        loss.backward()

        expected_policy = -(((rewards - values.detach()) * log_probs).mean())
        expected_value = torch.nn.functional.mse_loss(values, rewards)
        torch.testing.assert_close(loss.detach(), expected_policy + 0.5 * expected_value)
        self.assertAlmostEqual(diagnostics["value_loss"], expected_value.item())
        self.assertGreater(values.grad.abs().sum().item(), 0.0)
        self.assertGreater(log_probs.grad[0].item(), 0.0)
        self.assertLess(log_probs.grad[1].item(), 0.0)

    def test_overflow_switch_policy_loss_pushes_down_excess_switch_logprob(self):
        log_probs = torch.tensor([-0.7, -0.8], requires_grad=True)
        overflow_orders = torch.tensor([1.0, 2.0])

        loss = overflow_switch_policy_loss(log_probs, overflow_orders, penalty_coef=0.01)
        loss.backward()

        self.assertLess(loss.item(), 0.0)
        self.assertGreater(log_probs.grad[0].item(), 0.0)
        self.assertGreater(log_probs.grad[1].item(), log_probs.grad[0].item())

    def test_controller_terms_ignore_switch_supervision_unless_explicitly_enabled(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.1,
            controller_aux_mdd_coef=0.1,
            controller_sup_coef=1.0,
            controller_use_switch_supervision=False,
            controller_max_switches=40,
            controller_max_switch_penalty_coef=0.0,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([1]),
            "free_switch_index": 1,
            "target_return": torch.tensor([0.02]),
            "target_mdd": torch.tensor([0.05]),
            "sup_label": torch.tensor([1.0]),
            "sup_weight": torch.tensor([1.0]),
        }

        _, _, _, aux_return_loss, aux_mdd_loss, _, _, sup_loss, _, _ = trainer._controller_episode_terms([[record]])

        self.assertIsNotNone(aux_return_loss)
        self.assertIsNotNone(aux_mdd_loss)
        self.assertIsNone(sup_loss)

    def test_controller_terms_can_sum_logprob_over_episode_actions(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_pg_logprob_reduction="sum",
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )

        def make_record(action):
            return {
                "ssm": {
                    "z": torch.zeros(1, 1),
                    "h": torch.zeros(1, 1),
                    "p": torch.zeros(1, 1),
                    "q_bear": torch.zeros(1, 1),
                    "q_bull": torch.zeros(1, 1),
                },
                "weights_drift": torch.ones(1, 1),
                "port_state": torch.zeros(1, 6),
                "switch_action": torch.ones(1, 1),
                "asset_state": None,
                "action": torch.tensor([action]),
                "free_switch_index": 0,
            }

        episode_logprob, _, _, _, _, _, _, _, _, _ = trainer._controller_episode_terms([
            [make_record(1), make_record(0)],
            [make_record(1)],
        ])

        logits = torch.tensor([[0.0, 0.25]])
        expected = (
            Categorical(logits=logits).log_prob(torch.tensor([1])).sum()
            + Categorical(logits=logits).log_prob(torch.tensor([0])).sum()
            + Categorical(logits=logits).log_prob(torch.tensor([1])).sum()
        )
        self.assertTrue(torch.allclose(episode_logprob.detach(), expected))

    def test_controller_terms_build_local_switch_advantage_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=1.0,
            controller_local_adv_scale=1.0,
            controller_local_adv_clip=10.0,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([0]),
            "free_switch_index": 0,
            "switch_advantage": torch.tensor([0.2]),
        }

        terms = trainer._controller_episode_terms([[record]])
        local_adv_loss = terms[-2]

        self.assertIsNotNone(local_adv_loss)
        self.assertLess(local_adv_loss.item(), 0.0)

    def test_controller_terms_can_use_weighted_bce_local_advantage_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=1.0,
            controller_local_adv_scale=0.1,
            controller_local_adv_clip=10.0,
            controller_local_adv_loss_type="weighted_bce",
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([0]),
            "free_switch_index": 0,
            "switch_advantage": torch.tensor([0.2]),
        }

        terms = trainer._controller_episode_terms([[record]])
        local_adv_loss = terms[-2]

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.25]),
            torch.tensor([1.0]),
        )
        self.assertIsNotNone(local_adv_loss)
        torch.testing.assert_close(local_adv_loss.detach(), expected)

    def test_weighted_bce_local_advantage_margin_treats_small_positive_advantage_as_hold(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=1.0,
            controller_local_adv_scale=0.1,
            controller_local_adv_clip=10.0,
            controller_local_adv_loss_type="weighted_bce",
            controller_local_adv_margin=0.05,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor(policy_logit=0.25))
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([0]),
            "free_switch_index": 0,
            "switch_advantage": torch.tensor([0.02]),
        }

        terms = trainer._controller_episode_terms([[record]])
        local_adv_loss = terms[-2]

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.25]),
            torch.tensor([0.0]),
        )
        self.assertIsNotNone(local_adv_loss)
        torch.testing.assert_close(local_adv_loss.detach(), expected)

    def test_weighted_bce_local_advantage_weights_records_by_advantage_magnitude(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=1.0,
            controller_local_adv_scale=1.0,
            controller_local_adv_clip=10.0,
            controller_local_adv_loss_type="weighted_bce",
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor(policy_logit=2.0))
        )

        def make_record(switch_advantage):
            return {
                "ssm": {
                    "z": torch.zeros(1, 1),
                    "h": torch.zeros(1, 1),
                    "p": torch.zeros(1, 1),
                    "q_bear": torch.zeros(1, 1),
                    "q_bull": torch.zeros(1, 1),
                },
                "weights_drift": torch.ones(1, 1),
                "port_state": torch.zeros(1, 6),
                "switch_action": torch.ones(1, 1),
                "asset_state": None,
                "action": torch.tensor([0]),
                "free_switch_index": 0,
                "switch_advantage": torch.tensor([switch_advantage]),
            }

        terms = trainer._controller_episode_terms([
            [make_record(2.0), make_record(-0.5)],
        ])
        local_adv_loss = terms[-2]

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0]),
            torch.tensor([1.0]),
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
        )
        expected = (2.0 * pos_loss + 0.5 * neg_loss) / 2.5
        torch.testing.assert_close(local_adv_loss.detach(), expected)

    def test_weighted_bce_local_advantage_can_balance_positive_and_negative_classes(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=1.0,
            controller_local_adv_scale=1.0,
            controller_local_adv_clip=10.0,
            controller_local_adv_loss_type="weighted_bce",
            controller_local_adv_balance_classes=True,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor(policy_logit=2.0))
        )

        def make_record(switch_advantage):
            return {
                "ssm": {
                    "z": torch.zeros(1, 1),
                    "h": torch.zeros(1, 1),
                    "p": torch.zeros(1, 1),
                    "q_bear": torch.zeros(1, 1),
                    "q_bull": torch.zeros(1, 1),
                },
                "weights_drift": torch.ones(1, 1),
                "port_state": torch.zeros(1, 6),
                "switch_action": torch.ones(1, 1),
                "asset_state": None,
                "action": torch.tensor([0]),
                "free_switch_index": 0,
                "switch_advantage": torch.tensor([switch_advantage]),
            }

        terms = trainer._controller_episode_terms([
            [make_record(2.0), make_record(1.0), make_record(-0.5)],
        ])
        local_adv_loss = terms[-2]

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0]),
            torch.tensor([1.0]),
        )
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
        )
        expected = 0.5 * pos_loss + 0.5 * neg_loss
        torch.testing.assert_close(local_adv_loss.detach(), expected)

    def test_controller_terms_build_switch_advantage_aux_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_aux_switch_adv_coef=1.0,
            controller_aux_switch_adv_target_scale=2.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=0.0,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([1]),
            "free_switch_index": 1,
            "switch_advantage": torch.tensor([0.20]),
        }

        terms = trainer._controller_episode_terms([[record]])
        aux_switch_adv_loss = terms[5]

        expected = torch.nn.functional.smooth_l1_loss(
            torch.tensor([0.02]),
            torch.tensor([0.40]),
        )
        self.assertIsNotNone(aux_switch_adv_loss)
        torch.testing.assert_close(aux_switch_adv_loss.detach(), expected)

    def test_controller_terms_can_use_weighted_bce_switch_advantage_aux_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_aux_switch_adv_coef=1.0,
            controller_aux_switch_adv_loss_type="weighted_bce",
            controller_aux_switch_adv_target_scale=1.0,
            controller_local_adv_scale=0.1,
            controller_local_adv_clip=10.0,
            controller_local_adv_margin=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=0,
            controller_local_adv_coef=0.0,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([1]),
            "free_switch_index": 1,
            "switch_advantage": torch.tensor([0.20]),
        }

        terms = trainer._controller_episode_terms([[record]])
        aux_switch_adv_loss = terms[5]

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.02 / 0.1]),
            torch.tensor([1.0]),
        )
        self.assertIsNotNone(aux_switch_adv_loss)
        torch.testing.assert_close(aux_switch_adv_loss.detach(), expected)

    def test_controller_terms_penalize_expected_switch_budget_overflow(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_max_switches=2,
            controller_expected_switch_penalty_coef=0.5,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )

        def make_record():
            return {
                "ssm": {
                    "z": torch.zeros(1, 1),
                    "h": torch.zeros(1, 1),
                    "p": torch.zeros(1, 1),
                    "q_bear": torch.zeros(1, 1),
                    "q_bull": torch.zeros(1, 1),
                },
                "weights_drift": torch.ones(1, 1),
                "port_state": torch.zeros(1, 6),
                "switch_action": torch.ones(1, 1),
                "asset_state": None,
                "action": torch.tensor([0]),
                "free_switch_index": 0,
            }

        terms = trainer._controller_episode_terms([[make_record(), make_record(), make_record()]])
        expected_switch_loss = terms[-1]

        exit_prob = torch.sigmoid(torch.tensor(0.25))
        expected = 0.5 * torch.relu(3 * exit_prob - 1.0).pow(2)
        self.assertIsNotNone(expected_switch_loss)
        torch.testing.assert_close(expected_switch_loss.detach(), expected)

    def test_controller_terms_do_not_add_overflow_action_loss_by_default(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_local_adv_coef=0.0,
            controller_max_switches=2,
            controller_max_switch_penalty_coef=0.001,
        )
        trainer.agent = SimpleNamespace(
            net=SimpleNamespace(mon=_FakeControllerMonitor())
        )
        record = {
            "ssm": {
                "z": torch.zeros(1, 1),
                "h": torch.zeros(1, 1),
                "p": torch.zeros(1, 1),
                "q_bear": torch.zeros(1, 1),
                "q_bull": torch.zeros(1, 1),
            },
            "weights_drift": torch.ones(1, 1),
            "port_state": torch.zeros(1, 6),
            "switch_action": torch.ones(1, 1),
            "asset_state": None,
            "action": torch.tensor([1]),
            "free_switch_index": 3,
        }

        terms = trainer._controller_episode_terms([[record]])
        overflow_action_loss = terms[6]

        self.assertEqual(float(overflow_action_loss.detach().item()), 0.0)

    def test_controller_aux_pretrain_batch_updates_only_return_and_mdd_losses(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        opt = _FakeOptimizer()
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.2,
            controller_aux_mdd_coef=0.3,
        )
        trainer.agent = SimpleNamespace(opt_mon=opt, max_grad_norm=None)

        aux_return = torch.tensor(0.5, requires_grad=True)
        aux_mdd = torch.tensor(0.25, requires_grad=True)

        diagnostics = trainer._update_controller_aux_batch([aux_return], [aux_mdd])

        self.assertTrue(opt.zero_grad_called)
        self.assertTrue(opt.step_called)
        self.assertAlmostEqual(diagnostics["loss"], 0.2 * 0.5 + 0.3 * 0.25)
        self.assertNotIn("sup_loss", diagnostics)

    def test_controller_aux_pretrain_batch_can_update_local_advantage_policy_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        opt = _FakeOptimizer()
        trainer.cfg = SimpleNamespace(
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_aux_switch_adv_coef=0.0,
            controller_local_adv_coef=0.2,
        )
        trainer.agent = SimpleNamespace(opt_mon=opt, max_grad_norm=None)

        local_adv_loss = torch.tensor(0.5, requires_grad=True)

        diagnostics = trainer._update_controller_aux_batch(
            local_adv_losses=[local_adv_loss],
        )

        self.assertTrue(opt.zero_grad_called)
        self.assertTrue(opt.step_called)
        self.assertAlmostEqual(diagnostics["local_adv_loss"], 0.5)
        self.assertAlmostEqual(diagnostics["loss"], 0.1)

    def test_controller_pg_deterministic_rollout_sampling_ignores_global_rng_order(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(
            seed=49,
            controller_deterministic_rollout_sampling=True,
        )
        stats = {"exit_prob": torch.tensor([0.37])}
        logits = torch.log(torch.tensor([[0.63, 0.37]]))

        torch.manual_seed(1)
        action_a, _ = trainer._sample_controller_pg_action(
            stats,
            logits,
            start_idx=1234,
            step_idx=17,
            epoch=2,
        )
        torch.manual_seed(999)
        action_b, _ = trainer._sample_controller_pg_action(
            stats,
            logits,
            start_idx=1234,
            step_idx=17,
            epoch=2,
        )

        self.assertTrue(torch.equal(action_a, action_b))

    def test_controller_aux_replay_recomputes_losses_for_each_replay_epoch(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(controller_aux_replay_epochs=3)
        calls = {"terms": 0, "updates": 0}

        def fake_terms(episode_segments):
            self.assertEqual(episode_segments, ["segment"])
            calls["terms"] += 1
            loss = torch.tensor(float(calls["terms"]), requires_grad=True)
            return None, None, None, None, None, None, None, None, loss, None

        def fake_update(aux_return_losses=None, aux_mdd_losses=None,
                        aux_switch_adv_losses=None, local_adv_losses=None):
            calls["updates"] += 1
            self.assertEqual(len(local_adv_losses), 1)
            return {"loss": float(local_adv_losses[0].detach().item())}

        trainer._controller_episode_terms = fake_terms
        trainer._update_controller_aux_batch = fake_update

        diagnostics = trainer._update_controller_aux_replay_batch([["segment"]])

        self.assertEqual(calls, {"terms": 3, "updates": 3})
        self.assertEqual([diag["loss"] for diag in diagnostics], [1.0, 2.0, 3.0])

    def test_controller_aux_pretrain_windows_can_use_offpolicy_fixed_trajectory(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        windows = [(10, 70)]
        calls = []

        def fixed_runner(windows_arg, fixed_cycle, epoch=None):
            calls.append(("fixed", windows_arg, fixed_cycle, epoch))
            return ["fixed"]

        def controlled_runner(windows_arg, fixed_cycle, epoch=None):
            calls.append(("controlled", windows_arg, fixed_cycle, epoch))
            return ["controlled"]

        trainer._run_controller_aux_fixed_windows = fixed_runner
        trainer._run_controller_pg_controlled_windows = controlled_runner

        trainer.cfg = SimpleNamespace(controller_aux_pretrain_offpolicy=True)
        self.assertEqual(trainer._run_controller_aux_pretrain_windows(windows, 30, epoch=2), ["fixed"])

        trainer.cfg = SimpleNamespace(controller_aux_pretrain_offpolicy=False)
        self.assertEqual(trainer._run_controller_aux_pretrain_windows(windows, 30, epoch=3), ["controlled"])

        self.assertEqual(calls, [
            ("fixed", windows, 30, 2),
            ("controlled", windows, 30, 3),
        ])

    def test_controller_pg_batch_includes_expected_switch_budget_loss(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        opt = _FakeOptimizer()
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_entropy_coef=0.0,
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_local_adv_coef=0.0,
        )
        trainer.agent = SimpleNamespace(opt_mon=opt, max_grad_norm=None, net=SimpleNamespace(mon=torch.nn.Linear(1, 1)))
        log_prob = torch.tensor(0.0, requires_grad=True)
        expected_switch_loss = torch.tensor(0.25, requires_grad=True)

        diagnostics = trainer._update_controller_pg_batch(
            [log_prob],
            [1.0],
            [torch.tensor(0.0)],
            expected_switch_losses=[expected_switch_loss],
        )

        self.assertTrue(opt.zero_grad_called)
        self.assertTrue(opt.step_called)
        self.assertAlmostEqual(diagnostics["expected_switch_loss"], 0.25)
        self.assertAlmostEqual(diagnostics["loss"], 0.25)

    def test_controller_pg_batch_includes_value_loss_when_enabled(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        opt = _FakeOptimizer()
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_entropy_coef=0.0,
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_local_adv_coef=0.0,
            controller_value_coef=0.25,
            controller_value_normalize_advantage=False,
        )
        trainer.agent = SimpleNamespace(opt_mon=opt, max_grad_norm=None, net=SimpleNamespace(mon=torch.nn.Linear(1, 1)))
        log_prob = torch.tensor(0.0, requires_grad=True)
        episode_value = torch.tensor(0.0, requires_grad=True)

        diagnostics = trainer._update_controller_pg_batch(
            [log_prob],
            [2.0],
            [torch.tensor(0.0)],
            episode_values=[episode_value],
        )

        self.assertTrue(opt.zero_grad_called)
        self.assertTrue(opt.step_called)
        self.assertAlmostEqual(diagnostics["value_loss"], 4.0)
        self.assertAlmostEqual(diagnostics["value_weighted_loss"], 1.0)
        self.assertAlmostEqual(diagnostics["loss"], 1.0)

    def test_controller_pg_segments_batch_accumulates_12_episode_batch_with_one_optimizer_step(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        opt = _FakeOptimizer()
        trainer.device = torch.device("cpu")
        trainer.cfg = SimpleNamespace(
            controller_entropy_coef=0.0,
            controller_aux_return_coef=0.0,
            controller_aux_mdd_coef=0.0,
            controller_aux_switch_adv_coef=0.0,
            controller_sup_coef=0.0,
            controller_use_switch_supervision=False,
            controller_local_adv_coef=0.0,
            controller_value_coef=0.25,
            controller_value_normalize_advantage=False,
        )
        trainer.agent = SimpleNamespace(opt_mon=opt, max_grad_norm=None, net=SimpleNamespace(mon=torch.nn.Linear(1, 1)))
        calls = []

        def fake_terms(episode_segments):
            calls.append(episode_segments)
            episode_idx = int(episode_segments[0]["episode_idx"])
            sign = -1.0 if episode_idx % 2 else 1.0
            log_prob = torch.tensor(sign * float(episode_idx) / 10.0, requires_grad=True)
            entropy = torch.tensor(0.0, requires_grad=True)
            episode_value = torch.tensor(0.0, requires_grad=True)
            return log_prob, entropy, episode_value, None, None, None, None, None, None, None

        trainer._controller_episode_terms = fake_terms
        trainer._controller_switch_supervision_enabled = lambda: False
        episode_segments_batch = [[{"episode_idx": i}] for i in range(12)]
        rewards = [float(i) for i in range(12)]

        diagnostics = trainer._update_controller_pg_segments_batch(
            episode_segments_batch,
            rewards,
        )

        self.assertEqual(opt.zero_grad_calls, 1)
        self.assertEqual(opt.step_calls, 1)
        self.assertEqual(len(calls), 24)
        self.assertEqual(calls[:12], episode_segments_batch)
        self.assertEqual(calls[12:], episode_segments_batch)
        self.assertEqual(diagnostics["episode_count"], 12)
        self.assertAlmostEqual(diagnostics["reward_mean"], 5.5)
        self.assertAlmostEqual(diagnostics["reward_std"], 3.4520526, places=5)
        self.assertAlmostEqual(diagnostics["reward_min"], 0.0)
        self.assertAlmostEqual(diagnostics["reward_max"], 11.0)
        self.assertAlmostEqual(diagnostics["reward_abs_mean"], 5.5)
        self.assertGreater(diagnostics["policy_abs_loss"], abs(diagnostics["policy_loss"]))
        self.assertAlmostEqual(diagnostics["value_loss"], sum(i * i for i in range(12)) / 12.0, places=5)
        self.assertAlmostEqual(
            diagnostics["value_weighted_loss"],
            0.25 * sum(i * i for i in range(12)) / 12.0,
            places=5,
        )

    def test_controller_fixed_pool_limit_samples_starts_across_full_pool(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.env = _FakePoolEnv()
        trainer.cfg = SimpleNamespace(
            controller_start_stride_days=1,
            controller_windows_per_epoch=5,
            train_episodes_per_epoch=5,
            controller_fixed_pool_limit=3,
            max_hold=30,
        )

        starts = trainer._controller_train_start_pool(rollout_len=600)

        self.assertEqual(starts, [0, 10, 19])

    def test_test_episode_window_limits_test_days_when_requested(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.env = _FakeTestWindowEnv()
        trainer.cfg = SimpleNamespace(test_max_days=60)

        self.assertEqual(trainer._test_episode_window(), (700, 760))

    def test_test_episode_window_uses_full_test_range_by_default(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.env = _FakeTestWindowEnv()
        trainer.cfg = SimpleNamespace(test_max_days=0)

        self.assertIsNone(trainer._test_episode_window())

    def test_controller_baseline_stats_are_cached_by_window(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(controller_cache_baseline_stats=True)
        trainer.env = SimpleNamespace()
        trainer._controller_baseline_cache = {}
        calls = []
        expected = CounterfactualStats(
            log_return=0.1,
            max_drawdown=0.2,
            turnover=0.3,
            free_switch_count=0,
            segment_count=12,
        )

        def fake_fixed_window(env, start_idx, stop_idx, fixed_cycle, disable_inner=False):
            calls.append((start_idx, stop_idx, fixed_cycle, disable_inner))
            return expected, [1.0]

        trainer._run_fixed_hrl_window = fake_fixed_window

        first = trainer._get_controller_baseline_stats(100, 700, 30)
        second = trainer._get_controller_baseline_stats(100, 700, 30)

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        self.assertEqual(calls, [(100, 700, 30, False)])

        trainer.cfg.controller_pg_disable_inner = True
        third = trainer._get_controller_baseline_stats(100, 700, 30)

        self.assertIs(third, expected)
        self.assertEqual(calls, [(100, 700, 30, False), (100, 700, 30, True)])

    def test_controller_skip_val_disables_final_pg_validation(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        trainer.cfg = SimpleNamespace(controller_skip_val=True)

        self.assertFalse(trainer._should_validate_controller_pg(epoch=1, epochs=2, val_interval=1))
        self.assertFalse(trainer._should_validate_controller_pg(epoch=2, epochs=2, val_interval=999))

    def test_controller_exit_prob_summary_reports_threshold_counts(self):
        summary = HRL_Trainer._controller_exit_prob_summary(
            [0.41, 0.49, 0.501, 0.72],
            thresholds=[0.5, 0.45],
        )

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 0.53025)
        self.assertAlmostEqual(summary["p50"], 0.4955)
        self.assertEqual(summary["gt_0p5"], 2)
        self.assertEqual(summary["gt_0p45"], 3)

    def test_controller_switch_advantage_summary_reports_sign_distribution(self):
        summary = HRL_Trainer._controller_switch_advantage_summary(
            [-0.03, 0.0, 0.02, 0.05]
        )

        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["mean"], 0.01)
        self.assertAlmostEqual(summary["positive_frac"], 0.5)
        self.assertEqual(summary["positive_count"], 2)

    def test_controller_switch_advantage_summary_reports_exit_prob_alignment(self):
        summary = HRL_Trainer._controller_switch_advantage_summary(
            [-0.03, 0.02, 0.05],
            exit_probs=[0.2, 0.6, 0.8],
        )

        self.assertAlmostEqual(summary["positive_exit_prob_mean"], 0.7)
        self.assertAlmostEqual(summary["negative_exit_prob_mean"], 0.2)
        self.assertAlmostEqual(summary["exit_prob_gap"], 0.5)

    def test_inner_adjusted_switch_advantage_compares_inner_executed_weights(self):
        trainer = HRL_Trainer.__new__(HRL_Trainer)
        env = _FakeSwitchAdvEnv()
        obs = {"weights_drift": torch.tensor([[0.50, 0.50]])}
        hold_exec = torch.tensor([[0.60, 0.40]])
        switch_exec = torch.tensor([[0.20, 0.80]])

        advantage = trainer._controller_inner_adjusted_switch_advantage(
            env,
            obs,
            hold_exec,
            switch_exec,
        )

        hold_ret, _ = env._future_portfolio_return_and_max_drawdown(hold_exec.flatten(), env.day, 20)
        switch_ret, _ = env._future_portfolio_return_and_max_drawdown(switch_exec.flatten(), env.day, 20)
        current = env._normalize(obs["weights_drift"])
        hold_turnover = torch.sum(torch.abs(env._normalize(hold_exec) - current))
        switch_turnover = torch.sum(torch.abs(env._normalize(switch_exec) - current))
        expected = switch_ret - hold_ret - (switch_turnover - hold_turnover) * env.transaction_cost_pct
        torch.testing.assert_close(advantage.detach(), expected.detach())


if __name__ == "__main__":
    unittest.main()
