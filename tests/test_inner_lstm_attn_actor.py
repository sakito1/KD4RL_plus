import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Components.PPO_model import InnerAC


class InnerLstmAttnActorTests(unittest.TestCase):
    def test_inner_encoder_uses_two_layer_lstm_and_two_temporal_attention_layers(self):
        torch.manual_seed(3)
        model = InnerAC(in_features=7, hidden_dim=16, max_boundary=0.5, dropout=0.0)
        inner_state = torch.randn(2, 5, 10, 7)
        base = torch.full((2, 5), 0.2)
        drift = torch.full((2, 5), 0.2)

        feat = model.encode(inner_state, base, drift)

        self.assertEqual(model.inner_lstm.num_layers, 2)
        self.assertEqual(model.temporal_attn1.embed_dim, 16)
        self.assertEqual(model.temporal_attn2.embed_dim, 16)
        self.assertEqual(model.fusion[0].in_features, 18)
        self.assertEqual(feat.shape, (2, 5, 16))
        self.assertEqual(model.last_temporal_attn1.shape, (2, 5, 10))
        self.assertEqual(model.last_temporal_attn2.shape, (2, 5, 10))
        torch.testing.assert_close(
            model.last_temporal_attn1.sum(dim=-1),
            torch.ones(2, 5),
        )
        torch.testing.assert_close(
            model.last_temporal_attn2.sum(dim=-1),
            torch.ones(2, 5),
        )

    def test_inner_action_shape_and_holding_mask_stay_compatible(self):
        torch.manual_seed(5)
        model = InnerAC(in_features=7, hidden_dim=16, max_boundary=0.5, dropout=0.1)
        inner_state = torch.randn(3, 6, 10, 7)
        base = torch.tensor(
            [
                [0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        drift = base.clone()

        weights, score, logp, entropy, value = model.build_inner_action_simple(
            inner_state,
            base,
            drift,
            alpha=0.5,
            deterministic=True,
        )

        self.assertEqual(weights.shape, base.shape)
        self.assertEqual(score.shape, base.shape)
        self.assertEqual(logp.shape, (3,))
        self.assertEqual(value.shape, (3, 1))
        self.assertEqual(entropy.dim(), 0)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        self.assertTrue(torch.all(weights[base == 0.0] == 0.0).item())

    def test_inner_action_uses_base_weight_as_rebalance_anchor(self):
        torch.manual_seed(17)
        model = InnerAC(in_features=7, hidden_dim=16, max_boundary=0.5, dropout=0.0)
        inner_state = torch.randn(1, 5, 10, 7)
        base = torch.tensor([[0.50, 0.30, 0.20, 0.00, 0.00]], dtype=torch.float32)
        drift = torch.tensor([[0.45, 0.35, 0.20, 0.00, 0.00]], dtype=torch.float32)

        weights, _, _, _, _ = model.build_inner_action_simple(
            inner_state,
            base,
            drift,
            alpha=0.0,
            deterministic=True,
        )

        torch.testing.assert_close(weights, base)


if __name__ == "__main__":
    unittest.main()
