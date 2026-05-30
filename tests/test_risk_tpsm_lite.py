import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from risk_tpsm_lite import (
    RiskTPSMLite,
    build_future_drawdown_risk_labels,
    build_risk_tpsm_features,
    build_soft_regime_labels,
    compute_risk_tpsm_loss,
    compute_rolling_drawdown,
    compute_downside_volatility,
    map_risk_outputs_to_legacy,
)


def make_price_frame(n=120):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    trend = np.linspace(0.0, 0.35, n)
    cycle = 0.05 * np.sin(np.arange(n) / 5.0)
    close = 100.0 * np.exp(trend + cycle)
    return pd.DataFrame(
        {
            "adjopen": close * (1.0 - 0.002),
            "adjhigh": close * (1.0 + 0.01),
            "adjlow": close * (1.0 - 0.01),
            "adjclose": close,
            "amount": 1e6 + 1000 * np.arange(n),
        },
        index=idx,
    )


class RiskTPSMLiteTests(unittest.TestCase):
    def test_feature_builder_is_causal_and_finite(self):
        df = make_price_frame()
        feat = build_risk_tpsm_features(df, window=20)
        self.assertEqual(len(feat), len(df))
        self.assertTrue(np.isfinite(feat.values).all())
        self.assertGreater(feat.shape[1], 10)

        df_changed = df.copy()
        df_changed.iloc[80:, df_changed.columns.get_loc("adjclose")] *= 3.0
        feat_changed = build_risk_tpsm_features(df_changed, window=20)
        np.testing.assert_allclose(feat.iloc[:70].values, feat_changed.iloc[:70].values, atol=1e-6)

    def test_label_shapes_and_masks(self):
        close = make_price_frame(90)["adjclose"].values
        horizons = [5, 10, 20]
        train_mask = np.zeros(len(close), dtype=bool)
        train_mask[:50] = True
        y_risk, dd_norm, dd_raw, mask, thresholds = build_future_drawdown_risk_labels(
            close, horizons=horizons, train_mask=train_mask, threshold_quantile=0.7
        )
        self.assertEqual(y_risk.shape, (90, 3))
        self.assertEqual(dd_norm.shape, (90, 3))
        self.assertEqual(dd_raw.shape, (90, 3))
        self.assertEqual(mask.shape, (90, 3))
        self.assertEqual(thresholds.shape, (3,))
        self.assertFalse(mask[-1].any())
        self.assertTrue(((y_risk >= 0.0) & (y_risk <= 1.0)).all())

        y_regime, rmask = build_soft_regime_labels(close, horizons=horizons)
        self.assertEqual(y_regime.shape, (90, 3, 3))
        self.assertEqual(rmask.shape, (90, 3))
        sums = y_regime[rmask].sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_model_loss_and_gradients(self):
        model = RiskTPSMLite(in_dim=12, emb_dim=16, num_horizons=4)
        batch = {
            "x": torch.randn(8, 63, 12),
            "x_prev": torch.randn(8, 63, 12),
            "prev_valid": torch.ones(8, dtype=torch.bool),
            "y_risk": torch.rand(8, 4),
            "y_regime": torch.softmax(torch.randn(8, 4, 3), dim=-1),
            "dd_norm": torch.rand(8, 4),
            "mask": torch.ones(8, 4, dtype=torch.bool),
        }
        out = model(batch["x"])
        self.assertEqual(out["embedding"].shape, (8, 16))
        self.assertEqual(out["q_risk"].shape, (8, 4))
        self.assertEqual(out["regime_probs"].shape, (8, 4, 3))
        out["q_risk_prev"] = model(batch["x_prev"])["q_risk"].detach()
        loss, parts = compute_risk_tpsm_loss(out, batch, stage=3)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("risk_bce", parts)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in grads))

    def test_asset_conditioning_requires_and_uses_asset_id(self):
        model = RiskTPSMLite(
            in_dim=12,
            emb_dim=16,
            num_horizons=3,
            use_asset_conditioning=True,
            num_assets=4,
            asset_emb_dim=6,
        )
        x = torch.randn(5, 63, 12)
        asset_id = torch.tensor([0, 1, 2, 3, 0])
        out = model(x, asset_id)
        self.assertEqual(out["embedding"].shape, (5, 16))
        self.assertEqual(out["q_risk"].shape, (5, 3))
        with self.assertRaises(ValueError):
            model(x)

    def test_risk_helpers_and_legacy_mapping(self):
        close = np.array([10.0, 11.0, 9.0, 12.0])
        dd = compute_rolling_drawdown(close, 3)
        self.assertGreater(dd[2], 0.0)
        downside = compute_downside_volatility(np.array([0.1, -0.2, -0.1, 0.3]), 3)
        self.assertEqual(downside.shape, (4,))

        q = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        agg, bear, bull = map_risk_outputs_to_legacy(q, [5, 10, 20, 30])
        self.assertAlmostEqual(float(agg[0]), 0.25)
        self.assertAlmostEqual(float(bear[0]), 0.1)
        self.assertAlmostEqual(float(bull[0]), 0.3)


if __name__ == "__main__":
    unittest.main()
