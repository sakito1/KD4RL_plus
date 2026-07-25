import unittest

import torch

from Components.PPO_model import MonitorAC


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
