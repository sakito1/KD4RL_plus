import unittest

import torch

from Components.PPO_model import MonitorAC
from agent.PPO_agent import HRL_PPO_Agent


class _PreviewOuter:
    def pi(self, outer_state, weights_drift, deterministic=True):
        del outer_state, weights_drift, deterministic
        candidate = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
        return candidate, torch.zeros_like(candidate), torch.zeros(1), None, None

    def value(self, outer_state, weights_drift):
        del outer_state, weights_drift
        return torch.zeros(1, 1)


class _PreviewInner:
    def build_inner_action_simple(
            self, inner_state, base_used, weight_drift, alpha=1.0, deterministic=True):
        del inner_state, weight_drift, alpha, deterministic
        return base_used, torch.zeros_like(base_used), torch.zeros(1), None, torch.zeros(1, 1)

    def value(self, inner_state, base_used, weight_drift):
        del inner_state, base_used, weight_drift
        return torch.zeros(1, 1)


class _CapturingMonitor:
    def __init__(self):
        self.kwargs = {}

    def __call__(
            self, weights_drift, port_state, switch_action=None,
            deterministic=False, asset_state=None, **kwargs):
        del weights_drift, port_state, switch_action, deterministic, asset_state
        self.kwargs = kwargs
        return (
            torch.tensor([0]),
            torch.zeros(1),
            None,
            None,
            torch.zeros(1, 1),
        )


class ControllerDualBranchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MonitorAC(
            z_dim=6,
            h_dim=8,
            port_state_dim=6,
            hidden_dim=8,
            min_hold=1,
            max_hold=30,
            asset_in_dim=6,
            controller_window=5,
        )
        self.model.eval()
        self.asset_state = torch.randn(2, 4, 5, 6)
        self.hold_weights = torch.tensor(
            [[0.55, 0.25, 0.15, 0.05], [0.10, 0.20, 0.30, 0.40]],
            dtype=torch.float32,
        )
        self.candidate_weights = torch.tensor(
            [[0.05, 0.15, 0.25, 0.55], [0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        self.port_state = torch.tensor(
            [[0.25, 0.08, -0.03, 0.0, 0.0, 0.0],
             [0.60, 0.02, 0.04, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

    def _stats(self, candidate):
        return self.model.decision_stats(
            self.hold_weights,
            self.port_state,
            switch_action=candidate,
            asset_state=self.asset_state,
        )

    def test_outputs_separate_risk_and_advantage_embeddings(self):
        stats = self._stats(self.candidate_weights)

        self.assertEqual(stats["risk_embedding"].shape, (2, 8))
        self.assertEqual(stats["advantage_embedding"].shape, (2, 8))
        self.assertEqual(stats["hold_risk_pred"].shape, (2,))
        self.assertEqual(stats["switch_advantage_pred"].shape, (2,))
        self.assertEqual(stats["policy_logit"].shape, (2,))
        self.assertEqual(stats["exit_prob"].shape, (2,))

    def test_controller_has_no_dropout_during_collection_or_replay(self):
        self.assertFalse(any(
            isinstance(module, torch.nn.Dropout)
            for module in self.model.modules()
        ))

    def test_candidate_changes_only_advantage_branch_input(self):
        hold_stats = self._stats(self.hold_weights)
        switch_stats = self._stats(self.candidate_weights)

        torch.testing.assert_close(
            hold_stats["risk_embedding"],
            switch_stats["risk_embedding"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(
            hold_stats["advantage_embedding"],
            switch_stats["advantage_embedding"],
        ))

    def test_advantage_branch_uses_actual_execution_weights(self):
        base_stats = self._stats(self.candidate_weights)
        aligned_stats = self.model.decision_stats(
            self.hold_weights,
            self.port_state,
            switch_action=self.candidate_weights,
            asset_state=self.asset_state,
            hold_exec_weights=self.candidate_weights,
            switch_exec_weights=self.hold_weights,
            remaining_horizon=torch.tensor([[0.5], [0.5]]),
        )

        torch.testing.assert_close(
            base_stats["risk_embedding"],
            aligned_stats["risk_embedding"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(
            base_stats["advantage_embedding"],
            aligned_stats["advantage_embedding"],
        ))

    def test_remaining_horizon_changes_only_advantage_branch(self):
        short_stats = self.model.decision_stats(
            self.hold_weights,
            self.port_state,
            switch_action=self.candidate_weights,
            asset_state=self.asset_state,
            hold_exec_weights=self.hold_weights,
            switch_exec_weights=self.candidate_weights,
            remaining_horizon=torch.tensor([[0.1], [0.1]]),
        )
        long_stats = self.model.decision_stats(
            self.hold_weights,
            self.port_state,
            switch_action=self.candidate_weights,
            asset_state=self.asset_state,
            hold_exec_weights=self.hold_weights,
            switch_exec_weights=self.candidate_weights,
            remaining_horizon=torch.tensor([[1.0], [1.0]]),
        )

        torch.testing.assert_close(
            short_stats["risk_embedding"],
            long_stats["risk_embedding"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(
            short_stats["advantage_embedding"],
            long_stats["advantage_embedding"],
        ))

    def test_agent_passes_actual_holdings_and_candidate_preview_to_controller(self):
        monitor = _CapturingMonitor()
        agent = HRL_PPO_Agent.__new__(HRL_PPO_Agent)
        agent.device = torch.device("cpu")
        agent.cfg = type("Cfg", (), {
            "inner_use_topk": False,
            "inner_max_boundary": 1.0,
        })()
        agent.net = type("Net", (), {
            "outer": _PreviewOuter(),
            "inner": _PreviewInner(),
            "mon": monitor,
        })()
        obs = {
            "outer_state": torch.zeros(1, 2, 3, 4),
            "inner_state": torch.zeros(1, 2, 3, 4),
            "weights_drift": torch.tensor([[0.6, 0.4]]),
            "base_drift": torch.tensor([[0.7, 0.3]]),
            "port_state": torch.tensor([[0.25, 0.0, 0.0]]),
        }

        agent.get_action(obs, mode="eval", force_switch=None)

        torch.testing.assert_close(
            monitor.kwargs["hold_exec_weights"],
            torch.tensor([[0.6, 0.4]]),
        )
        torch.testing.assert_close(
            monitor.kwargs["switch_exec_weights"],
            torch.tensor([[0.2, 0.8]]),
        )
        torch.testing.assert_close(
            monitor.kwargs["remaining_horizon"],
            torch.tensor([[0.75]]),
        )

    def test_policy_gradient_reaches_both_branches_and_shared_encoder(self):
        self.model.train()
        stats = self._stats(self.candidate_weights)
        stats["policy_logit"].sum().backward()

        for module_name in ("risk_mlp", "advantage_mlp", "asset_lstm"):
            module = getattr(self.model, module_name)
            grad_total = sum(
                parameter.grad.abs().sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(grad_total, 0.0, module_name)

    def test_legacy_fixed_advantage_formula_parameters_do_not_change_policy(self):
        legacy_model = MonitorAC(
            z_dim=6,
            h_dim=8,
            port_state_dim=6,
            hidden_dim=8,
            min_hold=1,
            max_hold=30,
            asset_in_dim=6,
            controller_window=5,
            switch_adv_logit_coef=99.0,
            switch_adv_logit_scale=1e-5,
            switch_adv_logit_detach=True,
        )
        legacy_model.load_state_dict(self.model.state_dict())
        legacy_model.eval()

        expected = self._stats(self.candidate_weights)["policy_logit"]
        actual = legacy_model.decision_stats(
            self.hold_weights,
            self.port_state,
            switch_action=self.candidate_weights,
            asset_state=self.asset_state,
        )["policy_logit"]
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
