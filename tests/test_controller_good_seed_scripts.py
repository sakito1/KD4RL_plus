import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SH = ROOT / "train_sh"
SH_SCRIPT = TRAIN_SH / "run_controller_good_sh90.sh"
NAS_SCRIPT = TRAIN_SH / "run_controller_good_nas49.sh"


class ControllerGoodSeedScriptTests(unittest.TestCase):
    def _run_echo(self, script: Path):
        with tempfile.TemporaryDirectory(prefix="controller-good-script-") as tmpdir:
            source_root = Path(tmpdir) / "source"
            for market, seed in [("sh", "90"), ("nas", "49")]:
                checkpoint = source_root / market / "ppo" / f"seed_{seed}" / "checkpoints" / "hrl_fixed_best.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()

            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "SOURCE_ROOT": str(source_root),
                "OUTPUT_ROOT": str(Path(tmpdir) / "out"),
                "CUDA_VISIBLE_DEVICES": "0",
            })
            for key in ["NAS_SEEDS", "SH_SEEDS", "RUN_NAME", "CONTROLLER_FIXED_POOL_LIMIT", "CONTROLLER_INIT_EXIT_BIAS"]:
                env.pop(key, None)

            return subprocess.run(
                ["bash", str(script)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )

    def test_sh90_wrapper_uses_confirmed_good_controller_recipe(self):
        result = self._run_echo(SH_SCRIPT)

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("--markets sh", result.stdout)
        self.assertIn("--seeds 90", result.stdout)
        self.assertIn("--run_name lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_b12_sh90_3ep", result.stdout)
        self.assertIn("--controller_fixed_pool_limit 12", result.stdout)
        self.assertIn("--controller_init_exit_bias -1.0", result.stdout)
        self.assertIn("--controller_switch_adv_logit_coef 1.9", result.stdout)
        self.assertIn("--controller_eval_switch_threshold 0.5", result.stdout)
        self.assertNotIn("--markets nas", result.stdout)
        self.assertNotIn("--seeds 49", result.stdout)

    def test_nas49_wrapper_uses_confirmed_good_controller_recipe(self):
        result = self._run_echo(NAS_SCRIPT)

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("--markets nas", result.stdout)
        self.assertIn("--seeds 49", result.stdout)
        self.assertIn("--run_name lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_nas49_50_3ep", result.stdout)
        self.assertIn("--controller_fixed_pool_limit 12", result.stdout)
        self.assertIn("--controller_init_exit_bias -1.0", result.stdout)
        self.assertIn("--controller_switch_adv_logit_coef 1.9", result.stdout)
        self.assertIn("--controller_eval_switch_threshold 0.5", result.stdout)
        self.assertNotIn("--markets sh", result.stdout)
        self.assertNotIn("--seeds 90", result.stdout)


if __name__ == "__main__":
    unittest.main()
