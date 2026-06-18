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
            "--controller_first_joint_finetune",
            "--controller_no_hold_constraints",
            "--controller_train_fixed_episodes",
            "--controller_episode_batch_size",
            "12",
            "--controller_episode_parallel_workers",
            "12",
            "--controller_window",
            "30",
            "--controller_aux_return_coef",
            "0.2",
            "--controller_aux_mdd_coef",
            "0.2",
            "--controller_max_switches",
            "40",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        checkpoint_idx = command.index("--frozen_hrl_checkpoint") + 1
        self.assertEqual(command[checkpoint_idx], "/tmp/frozen/hrl_fixed_best.pth")
        self.assertIn("--controller_first_joint_finetune", command)
        self.assertIn("--controller_no_hold_constraints", command)
        self.assertIn("--controller_train_fixed_episodes", command)
        self.assertEqual(command[command.index("--controller_episode_batch_size") + 1], "12")
        self.assertEqual(command[command.index("--controller_episode_parallel_workers") + 1], "12")
        self.assertEqual(command[command.index("--controller_window") + 1], "30")
        self.assertEqual(command[command.index("--controller_aux_return_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_aux_mdd_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_max_switches") + 1], "40")


if __name__ == "__main__":
    unittest.main()
