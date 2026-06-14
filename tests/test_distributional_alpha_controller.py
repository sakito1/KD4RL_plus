import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Components.PPO_model import MonitorAC


class DistributionalAlphaControllerTests(unittest.TestCase):
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
        return z, p, q, weights, switch, port_state

    def test_decision_stats_expose_advantage_probability_and_dynamic_threshold(self):
        torch.manual_seed(7)
        controller = MonitorAC(z_dim=4, h_dim=4, port_state_dim=6, hidden_dim=8, min_hold=10, max_hold=40)
        z, p, q, weights, switch, port_state = self._inputs()

        stats = controller.decision_stats(z, z, p, q, q, weights, port_state, switch_action=switch)

        self.assertEqual(stats["p_adv"].shape, (2,))
        self.assertEqual(stats["tau"].shape, (2,))
        self.assertEqual(stats["pi_switch"].shape, (2,))
        self.assertTrue(torch.all(stats["p_adv"] >= 0.0).item())
        self.assertTrue(torch.all(stats["p_adv"] <= 1.0).item())
        self.assertTrue(torch.all(stats["tau"] >= 0.5).item())
        self.assertTrue(torch.all(stats["tau"] <= 0.9).item())
        self.assertTrue(torch.all(stats["pi_switch"] > 0.0).item())
        self.assertTrue(torch.all(stats["pi_switch"] < 1.0).item())

    def test_deterministic_action_uses_p_adv_threshold_rule(self):
        torch.manual_seed(11)
        controller = MonitorAC(z_dim=4, h_dim=4, port_state_dim=6, hidden_dim=8, min_hold=10, max_hold=40)
        controller.eval()
        z, p, q, weights, switch, port_state = self._inputs()

        stats = controller.decision_stats(z, z, p, q, q, weights, port_state, switch_action=switch)
        action, _, _, logits, _ = controller(
            z, z, p, q, q, weights, port_state, switch_action=switch, deterministic=True
        )

        torch.testing.assert_close(action, (stats["p_adv"] > stats["tau"]).long())
        torch.testing.assert_close(logits[:, 1] - logits[:, 0], stats["policy_logit"])

    def test_controller_ignores_old_probability_state_inputs(self):
        torch.manual_seed(13)
        controller = MonitorAC(z_dim=4, h_dim=4, port_state_dim=6, hidden_dim=8, min_hold=10, max_hold=40)
        controller.eval()
        z, p, q, weights, switch, port_state = self._inputs()

        stats_a = controller.decision_stats(z, z, p, q, q, weights, port_state, switch_action=switch)
        stats_b = controller.decision_stats(
            z,
            z,
            p + 100.0,
            q + 50.0,
            q - 50.0,
            weights,
            port_state,
            switch_action=switch,
        )

        torch.testing.assert_close(stats_a["p_adv"], stats_b["p_adv"])
        torch.testing.assert_close(stats_a["tau"], stats_b["tau"])
        torch.testing.assert_close(stats_a["pi_switch"], stats_b["pi_switch"])


if __name__ == "__main__":
    unittest.main()
