import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_hrl_training


class RunHrlTrainingCommandTests(unittest.TestCase):
    def _args_from_cli(self, extra_args):
        argv = ["run_hrl_training.py", *extra_args]
        with patch.object(sys, "argv", argv):
            args = run_hrl_training.parse_args()
        run_hrl_training.normalize_training_schedule(args)
        return args

    def test_disabled_controller_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--no_train_controller",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=42,
        )

        self.assertIn("--no_train_controller", command)
        self.assertNotIn("--train_monitor", command)
        monitor_epochs_idx = command.index("--warmup_monitor_epochs") + 1
        self.assertEqual(command[monitor_epochs_idx], "0")

    def test_enabled_controller_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--train_monitor",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=42,
        )

        self.assertIn("--train_monitor", command)
        self.assertNotIn("--no_train_controller", command)

    def test_frozen_hrl_checkpoint_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--frozen_hrl_checkpoint",
            "/tmp/frozen/hrl_fixed_best.pth",
            "--controller_only_finetune",
            "--controller_no_hold_constraints",
            "--controller_train_fixed_episodes",
            "--controller_episode_batch_size",
            "12",
            "--controller_episode_parallel_workers",
            "12",
            "--controller_window",
            "30",
            "--controller_fixed_pool_limit",
            "8",
            "--controller_hidden_dim",
            "128",
            "--controller_init_exit_bias",
            "-1.25",
            "--controller_pg_logprob_reduction",
            "sum",
            "--controller_eval_decision_mode",
            "daily",
            "--controller_aux_return_coef",
            "0.2",
            "--controller_aux_mdd_coef",
            "0.2",
            "--controller_aux_switch_adv_coef",
            "0.15",
            "--controller_aux_pretrain_offpolicy",
            "--controller_aux_replay_epochs",
            "4",
            "--controller_local_adv_coef",
            "0.3",
            "--controller_local_adv_margin",
            "0.01",
            "--controller_local_adv_loss_type",
            "weighted_bce",
            "--controller_local_adv_balance_classes",
            "--controller_expected_switch_penalty_coef",
            "0.004",
            "--controller_value_coef",
            "0.25",
            "--no_controller_value_normalize_advantage",
            "--controller_max_switches",
            "40",
            "--controller_eval_diagnostics",
            "--controller_eval_diag_thresholds",
            "0.5",
            "0.45",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        checkpoint_idx = command.index("--frozen_hrl_checkpoint") + 1
        self.assertEqual(command[checkpoint_idx], "/tmp/frozen/hrl_fixed_best.pth")
        self.assertIn("--controller_only_finetune", command)
        self.assertNotIn("--controller_first_joint_finetune", command)
        self.assertIn("--controller_no_hold_constraints", command)
        self.assertIn("--controller_train_fixed_episodes", command)
        self.assertEqual(command[command.index("--controller_episode_batch_size") + 1], "12")
        self.assertEqual(command[command.index("--controller_episode_parallel_workers") + 1], "12")
        self.assertEqual(command[command.index("--controller_window") + 1], "30")
        self.assertEqual(command[command.index("--controller_fixed_pool_limit") + 1], "8")
        self.assertEqual(command[command.index("--controller_hidden_dim") + 1], "128")
        self.assertEqual(command[command.index("--controller_init_exit_bias") + 1], "-1.25")
        self.assertEqual(command[command.index("--controller_pg_logprob_reduction") + 1], "sum")
        self.assertEqual(command[command.index("--controller_eval_decision_mode") + 1], "daily")
        self.assertEqual(command[command.index("--controller_aux_return_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_aux_mdd_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_aux_switch_adv_coef") + 1], "0.15")
        self.assertIn("--controller_aux_pretrain_offpolicy", command)
        self.assertEqual(command[command.index("--controller_aux_replay_epochs") + 1], "4")
        self.assertEqual(command[command.index("--controller_local_adv_coef") + 1], "0.3")
        self.assertEqual(command[command.index("--controller_local_adv_margin") + 1], "0.01")
        self.assertEqual(command[command.index("--controller_local_adv_loss_type") + 1], "weighted_bce")
        self.assertIn("--controller_local_adv_balance_classes", command)
        self.assertEqual(command[command.index("--controller_expected_switch_penalty_coef") + 1], "0.004")
        self.assertEqual(command[command.index("--controller_value_coef") + 1], "0.25")
        self.assertIn("--no_controller_value_normalize_advantage", command)
        self.assertEqual(command[command.index("--controller_max_switches") + 1], "40")
        self.assertIn("--controller_eval_diagnostics", command)
        diag_thresholds_idx = command.index("--controller_eval_diag_thresholds") + 1
        self.assertEqual(command[diag_thresholds_idx:diag_thresholds_idx + 2], ["0.5", "0.45"])
        self.assertNotIn("--controller_use_switch_supervision", command)

    def test_controller_switch_supervision_requires_explicit_legacy_flag(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--frozen_hrl_checkpoint",
            "/tmp/frozen/hrl_fixed_best.pth",
            "--controller_only_finetune",
            "--controller_sup_coef",
            "0.2",
            "--controller_use_switch_supervision",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        self.assertIn("--controller_use_switch_supervision", command)

    def test_test_skip_fixed_scenarios_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--test_only_checkpoint",
            "/tmp/controller/best_model.pth",
            "--test_skip_fixed_scenarios",
            "--test_max_days",
            "240",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        self.assertIn("--test_skip_fixed_scenarios", command)
        self.assertEqual(command[command.index("--test_max_days") + 1], "240")


if __name__ == "__main__":
    unittest.main()
