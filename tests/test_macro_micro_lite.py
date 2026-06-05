import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import torch

from SSM_pipeline import (
    MacroMicroAttentionLSTM,
    compute_macro_micro_loss,
    export_macro_micro_outputs,
    _delta_proxy_q,
    _macro_hard_label_from_soft,
)


class _Logger:
    def info(self, msg):
        pass


class MacroMicroLiteTests(unittest.TestCase):
    def test_attention_lstm_dual_head_and_loss(self):
        model = MacroMicroAttentionLSTM(input_dim=25, emb_dim=16, lstm_hidden_dim=32)
        x = torch.randn(8, 63, 25)
        y_macro = torch.rand(8)
        y_macro_cls = torch.randint(0, 2, (8,), dtype=torch.float32)
        y_micro = torch.rand(8)

        out = model(x)
        self.assertEqual(out["embedding"].shape, (8, 16))
        self.assertEqual(out["macro_logits"].shape, (8,))
        self.assertEqual(out["macro_cls_logits"].shape, (8,))
        self.assertEqual(out["micro_logits"].shape, (8,))
        self.assertEqual(out["feature_gate"].shape, (8, 25))

        loss, parts = compute_macro_micro_loss(
            out,
            y_macro,
            y_macro_cls,
            y_micro,
            lambda_macro=2.0,
            lambda_macro_cls=0.25,
            lambda_micro=0.5,
        )
        macro = torch.nn.functional.binary_cross_entropy_with_logits(out["macro_logits"], y_macro)
        macro_cls = torch.nn.functional.binary_cross_entropy_with_logits(out["macro_cls_logits"], y_macro_cls)
        micro = torch.nn.functional.binary_cross_entropy_with_logits(out["micro_logits"], y_micro)
        self.assertTrue(torch.allclose(loss, 2.0 * macro + 0.25 * macro_cls + 0.5 * micro))
        self.assertEqual(set(parts), {"loss", "macro_bce", "macro_cls_bce", "micro_bce"})

    def test_macro_hard_label_hysteresis(self):
        labels = _macro_hard_label_from_soft(
            np.array([0.2, 0.45, 0.55, 0.61, 0.52, 0.39, 0.45], dtype=np.float32),
            high=0.6,
            low=0.4,
        )
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.float32))

    def test_delta_proxy_q(self):
        q_bear, q_bull = _delta_proxy_q(np.array([0.5, 0.7, 0.4, 0.6], dtype=np.float32))
        np.testing.assert_allclose(q_bear, np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(q_bull, np.array([0.0, 0.2, 0.0, 0.2], dtype=np.float32), atol=1e-6)

    def test_export_writes_legacy_columns_and_states(self):
        model = MacroMicroAttentionLSTM(input_dim=25, emb_dim=16, lstm_hidden_dim=12)
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        asset = {
            "code": "AAA",
            "df": pd.DataFrame(
                {
                    "adjclose": np.linspace(10.0, 11.0, 5),
                    "label_ensemble_soft": np.linspace(0.1, 0.9, 5),
                    "label_micro_raw": np.linspace(0.2, 0.8, 5),
                },
                index=dates,
            ),
            "X_all": np.random.randn(5, 63, 25).astype(np.float32),
            "idx_all": dates,
            "y_macro_all": np.linspace(0.1, 0.9, 5).astype(np.float32),
            "y_macro_cls_all": np.array([0, 0, 1, 1, 1], dtype=np.float32),
            "y_micro_all": np.linspace(0.2, 0.8, 5).astype(np.float32),
        }
        args = types.SimpleNamespace(
            window=63,
            normalization="minmax",
            target_feature_count=25,
            lambda_macro=1.0,
            lambda_macro_cls=0.5,
            lambda_micro=1.0,
            macro_cls_high=0.6,
            macro_cls_low=0.4,
            gate_strength=0.5,
            gate_hidden_dim=0,
            infer_batch_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            export_macro_micro_outputs(
                model,
                [asset],
                args,
                tmp,
                "checkpoint.pt",
                [f"f{i}" for i in range(25)],
                torch.device("cpu"),
                _Logger(),
            )
            df = pd.read_csv(f"{tmp}/AAA.csv")
            for col in (
                "ssm3_p",
                "ssm3_q_bear",
                "ssm3_q_bull",
                "ssm3_pred",
                "ssm3_true",
                "ssm3_true_micro",
                "macro_cls_prob",
                "macro_cls_true",
                "macro_soft_true",
            ):
                self.assertIn(col, df.columns)
            state = torch.load(f"{tmp}/AAA_ssm3_states.pt", map_location="cpu")
            self.assertEqual(state["z"].shape, (5, 16))
            self.assertEqual(state["h"].shape, (5, 16))
            self.assertEqual(state["macro_cls_prob"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
