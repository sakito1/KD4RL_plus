import math
import unittest

import torch

from Train.controller_guidance import (
    analyze_guidance_windows,
    balanced_guidance_bce,
    balanced_guidance_weights,
    build_economic_guidance_labels,
    build_topk_guidance_labels,
    render_guidance_report,
)


class ControllerGuidanceLabelTests(unittest.TestCase):
    def test_economic_rule_combines_relative_risk_and_candidate_opportunity(self):
        risk = torch.tensor([0.06, 0.06, 0.01, 0.01])
        advantage = torch.tensor([-0.01, 0.01, 0.06, 0.01])

        result = build_economic_guidance_labels(
            risk,
            advantage,
            risk_threshold=0.05,
            advantage_threshold=0.05,
        )

        self.assertEqual(result.trigger.tolist(), [False, True, True, False])

    def test_economic_rule_supports_stricter_risk_branch_advantage(self):
        risk = torch.tensor([0.11, 0.11, 0.01, 0.01])
        advantage = torch.tensor([0.01, 0.03, 0.09, 0.11])

        result = build_economic_guidance_labels(
            risk,
            advantage,
            risk_threshold=0.10,
            risk_min_advantage_threshold=0.02,
            advantage_threshold=0.10,
        )

        self.assertEqual(result.trigger.tolist(), [False, True, False, True])

    def test_economic_rule_labels_every_triggered_day(self):
        risk = torch.tensor([0.06, 0.07, 0.08, 0.01, 0.01])
        advantage = torch.tensor([0.01, 0.02, 0.03, 0.00, 0.06])

        result = build_economic_guidance_labels(risk, advantage)

        self.assertEqual(result.labels.tolist(), [1.0, 1.0, 1.0, 0.0, 1.0])
        self.assertEqual(result.mask.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_balanced_guidance_weights_give_each_class_equal_total_mass(self):
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])

        weights = balanced_guidance_weights(labels, mask)

        torch.testing.assert_close(weights[labels > 0.5].sum(), torch.tensor(1.5))
        torch.testing.assert_close(
            weights[(labels < 0.5) & (mask > 0.5)].sum(),
            torch.tensor(1.5),
        )
        self.assertEqual(float(weights[-1]), 0.0)

    def test_topk_requires_positive_advantage_and_respects_budget(self):
        risk = torch.tensor([0.90, 0.10, 0.80, 0.20])
        advantage = torch.tensor([-0.50, 0.10, 0.20, 0.30])

        result = build_topk_guidance_labels(risk, advantage, topk=2)

        self.assertEqual(int(result.labels.sum().item()), 2)
        self.assertTrue(torch.all(advantage[result.labels.bool()] > 0))

    def test_topk_does_not_fill_budget_with_negative_advantages(self):
        risk = torch.tensor([0.90, 0.80, 0.70])
        advantage = torch.tensor([-0.10, 0.20, -0.30])

        result = build_topk_guidance_labels(risk, advantage, topk=20)

        self.assertEqual(result.labels.tolist(), [0.0, 1.0, 0.0])

    def test_topk_is_stable_and_reports_advantage_only_comparison(self):
        risk = torch.tensor([0.9, 0.8, 0.1, 0.2])
        advantage = torch.tensor([0.1, 0.2, 0.9, 0.8])

        result = build_topk_guidance_labels(risk, advantage, topk=2)

        self.assertEqual(result.labels.tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(
            result.advantage_only_labels.tolist(),
            [0.0, 0.0, 1.0, 1.0],
        )

    def test_balanced_bce_gives_equal_class_mass_and_correct_gradients(self):
        logits = torch.zeros(5, requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

        loss = balanced_guidance_bce(logits, labels)
        loss.backward()

        torch.testing.assert_close(loss.detach(), torch.tensor(math.log(2.0)))
        self.assertLess(float(logits.grad[0]), 0.0)
        self.assertTrue(torch.all(logits.grad[1:] > 0))
        torch.testing.assert_close(
            logits.grad[0].abs(),
            logits.grad[1:].sum(),
        )

    def test_balanced_bce_handles_an_all_hold_window(self):
        logits = torch.zeros(3, requires_grad=True)
        labels = torch.zeros(3)

        loss = balanced_guidance_bce(logits, labels)
        loss.backward()

        torch.testing.assert_close(loss.detach(), torch.tensor(math.log(2.0)))
        self.assertTrue(torch.all(logits.grad > 0))

    def test_window_analysis_reports_sparse_positive_labels_and_overlap(self):
        windows = [
            {
                "window_id": 0,
                "start_index": 100,
                "risk": torch.tensor([0.9, 0.1, 0.8, 0.2]),
                "advantage": torch.tensor([0.1, 0.9, -0.2, 0.3]),
            },
            {
                "window_id": 1,
                "start_index": 400,
                "risk": torch.tensor([0.4, 0.3, 0.2]),
                "advantage": torch.tensor([-0.1, -0.2, -0.3]),
            },
        ]

        detail_rows, summary = analyze_guidance_windows(windows, topk=2)

        self.assertEqual(len(detail_rows), 7)
        self.assertEqual(summary["split"], "train")
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["switch_label_count"], 2)
        self.assertEqual(summary["zero_positive_window_count"], 1)
        self.assertLessEqual(summary["max_switch_labels_per_window"], 2)
        selected = [row for row in detail_rows if row["label"] == 1]
        self.assertTrue(all(row["advantage"] > 0 for row in selected))
        self.assertIn("label_overlap_rate", summary)
        self.assertIn("balanced_bce_zero_logit", summary)

    def test_report_is_chinese_and_marks_train_only_scope(self):
        _, summary = analyze_guidance_windows(
            [{
                "risk": torch.tensor([0.8, 0.1]),
                "advantage": torch.tensor([0.2, -0.1]),
            }],
            topk=1,
        )

        report = render_guidance_report(summary, topk=1, rollout_len=300)

        self.assertIn("仅使用训练集", report)
        self.assertIn("300", report)
        self.assertIn("Top-1", report)
        self.assertIn("类别平衡 BCE", report)


if __name__ == "__main__":
    unittest.main()
