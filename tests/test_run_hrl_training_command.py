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


if __name__ == "__main__":
    unittest.main()
