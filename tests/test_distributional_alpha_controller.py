import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Components.PPO_model import MonitorAC


class HoldExitControllerTests(unittest.TestCase):
    def _inputs(self):
        z = torch.tensor(
            [
                [[0.1, 0.2, -0.1, 0.0], [0.3, -0.2, 0.1, 0.4], [-0.2, 0.0, 0.2, 0.1]],
                [[0.0, -0.1, 0.2, 0.2], [0.4, 0.1, -0.3, 0.0], [0.2, 0.3, 0.1, -0.1]],
            ],
            dtype=torch.float32,
        )
        p = torch.zeros(2, 3)
        q = torch.zeros(2, 3)
        weights = torch.tensor([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float32)
        switch = torch.tensor([[0.0, 0.5, 0.5], [0.6, 0.4, 0.0]], dtype=torch.float32)
        port_state = torch.tensor(
            [
                [0.25, 0.05, 0.08, 0.0, 0.0, 0.0],
                [0.80, 0.20, -0.04, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        asset_state = torch.randn(2, 3, 5, 7)
        return z, p, q, weights, switch, port_state, asset_state

    def test_hold_exit_controller_exposes_exit_return_and_risk_outputs(self):
        torch.manual_seed(7)
        controller = MonitorAC(
            z_dim=4,
            h_dim=4,
            port_state_dim=6,
            hidden_dim=8,
            action_dim=3,
            asset_in_dim=7,
        )
        z, p, q, weights, switch, port_state, asset_state = self._inputs()

        stats = controller.decision_stats(
            z, z, p, q, q, weights, port_state, switch_action=switch, asset_state=asset_state
        )

        self.assertEqual(stats["exit_prob"].shape, (2,))
        self.assertEqual(stats["hold_return_pred"].shape, (2,))
        self.assertEqual(stats["hold_risk_pred"].shape, (2,))
        self.assertEqual(stats["policy_logit"].shape, (2,))
        self.assertTrue(torch.all(stats["exit_prob"] > 0.0).item())
        self.assertTrue(torch.all(stats["exit_prob"] < 1.0).item())
        torch.testing.assert_close(stats["pi_switch"], stats["exit_prob"])
        torch.testing.assert_close(
            stats["policy_logit"],
            torch.logit(stats["exit_prob"].clamp(1e-6, 1.0 - 1e-6)),
        )

    def test_deterministic_action_uses_sigmoid_exit_threshold(self):
        torch.manual_seed(11)
        controller = MonitorAC(
            z_dim=4,
            h_dim=4,
            port_state_dim=6,
            hidden_dim=8,
            action_dim=3,
            asset_in_dim=7,
        )
        controller.eval()
        z, p, q, weights, switch, port_state, asset_state = self._inputs()

        stats = controller.decision_stats(
            z, z, p, q, q, weights, port_state, switch_action=switch, asset_state=asset_state
        )
        action, _, _, logits, _ = controller(
            z, z, p, q, q, weights, port_state,
            switch_action=switch,
            deterministic=True,
            asset_state=asset_state,
        )

        torch.testing.assert_close(action, (stats["exit_prob"] > 0.5).long())
        torch.testing.assert_close(logits[:, 1] - logits[:, 0], stats["policy_logit"])

    def test_hold_weights_use_small_floor_for_nonheld_assets(self):
        controller = MonitorAC(
            z_dim=4,
            h_dim=4,
            port_state_dim=6,
            hidden_dim=8,
            action_dim=3,
            asset_in_dim=7,
            weight_floor=1e-6,
        )
        weights = torch.tensor([[0.7, 0.3, 0.0]], dtype=torch.float32)

        w_soft = controller._soft_hold_weights(weights)

        self.assertGreater(w_soft[0, 2].item(), 0.0)
        self.assertLess(w_soft[0, 2].item(), 1e-5)
        self.assertAlmostEqual(float(w_soft.sum().item()), 1.0, places=6)

    def test_controller_uses_last_15_days_of_zscore_outer_state(self):
        controller = MonitorAC(
            z_dim=4,
            h_dim=4,
            port_state_dim=6,
            hidden_dim=8,
            action_dim=3,
            asset_in_dim=7,
            controller_window=15,
        )
        z, p, q, weights, switch, port_state, _ = self._inputs()
        asset_state = torch.randn(2, 3, 60, 7)

        seq = controller._encode_asset_sequence(asset_state, z, z)

        self.assertEqual(seq.shape, (2, 3, 15, 8))


if __name__ == "__main__":
    unittest.main()
