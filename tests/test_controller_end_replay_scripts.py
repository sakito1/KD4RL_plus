import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SH = ROOT / "train_sh"
NAS_SCRIPT = TRAIN_SH / "test_controller_end_nas49.sh"


class ControllerEndReplayScriptTests(unittest.TestCase):
    def _run_echo(self, script: Path):
        with tempfile.TemporaryDirectory(prefix="controller-end-replay-") as tmpdir:
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "RUN_ROOT": str(Path(tmpdir) / "verify"),
            })
            return subprocess.run(
                ["bash", str(script)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )

    def test_nas49_replays_end_checkpoint_with_original_command_config(self):
        result = self._run_echo(NAS_SCRIPT)

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("--market nas", result.stdout)
        self.assertIn("--seed 49", result.stdout)
        self.assertIn("--controller_fixed_pool_limit 12", result.stdout)
        self.assertIn("--controller_init_exit_bias -1.0", result.stdout)
        self.assertIn("--test_only_checkpoint /home/tongwenxuan/KD4RL_plus/results/end/nas_seed49/checkpoints/best_model.pth", result.stdout)


if __name__ == "__main__":
    unittest.main()
