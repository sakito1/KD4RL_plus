import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_sh" / "run_end_to_end_hrl_controller_joint_nas49_sh90.sh"
SUPERVISED_SCRIPT = ROOT / "train_sh" / "controller_5pct_outer_sh77_quick.sh"


class EndToEndHrlControllerJointScriptTests(unittest.TestCase):
    def test_controller_quick_script_runs_normalized_supervised_pretrain_only(self):
        text = SUPERVISED_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--controller_pretrain_only", text)
        self.assertIn("--controller_sup_pretrain_epochs 1", text)
        self.assertIn("--controller_aux_replay_epochs 50", text)
        self.assertIn("--controller_aux_mdd_target_scale 20.0", text)
        self.assertIn("--controller_aux_switch_adv_target_scale 20.0", text)
        self.assertIn("--controller_init_exit_bias 0.0", text)

    def test_script_uses_safe_end_to_end_output_and_no_frozen_source(self):
        env = os.environ.copy()
        for key in [
            "OUTPUT_ROOT",
            "RUN_NAME",
            "SOURCE_ROOT",
            "NAS_SEEDS",
            "SH_SEEDS",
            "CONTROLLER_JOINT_EPOCHS",
        ]:
            env.pop(key, None)
        env["PYTHON_BIN"] = "/bin/echo"

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("Output root: results/end_to_end_hrl_controller_joint_nas49_sh90", result.stdout)
        self.assertIn("Run name: lookback60_hold30_e2e_hrl_controller_joint_nas49_sh90", result.stdout)
        self.assertIn("SH seeds: 90", result.stdout)
        self.assertIn("NAS seeds: 49", result.stdout)
        self.assertIn("--markets sh --seeds 90", result.stdout)
        self.assertIn("--markets nas --seeds 49", result.stdout)
        self.assertIn("--end_to_end_controller_joint", result.stdout)
        self.assertIn("--controller_joint_epochs 1", result.stdout)
        self.assertIn("--train_monitor", result.stdout)
        self.assertIn("--controller_epochs 3", result.stdout)
        self.assertIn("--controller_fixed_pool_limit 12", result.stdout)
        self.assertIn("--controller_pg_logprob_reduction mean", result.stdout)
        self.assertIn("--controller_aux_switch_adv_loss_type smooth_l1", result.stdout)
        self.assertNotIn("--controller_switch_adv_logit_coef", result.stdout)
        self.assertNotIn("--controller_switch_adv_logit_detach", result.stdout)
        self.assertIn("--controller_eval_switch_threshold 0.5", result.stdout)
        self.assertIn("--joint_lr_mult 0.0001", result.stdout)
        self.assertNotIn("--frozen_hrl_checkpoint", result.stdout)
        self.assertNotIn("SOURCE_ROOT", result.stdout)
        self.assertNotIn("results/end ", result.stdout)
        self.assertNotIn("results/hrl_lookback60_hold30_inner_noaux_retrain", result.stdout)

    def test_reproduce_best_mode_delegates_to_two_stage_reproduction_script(self):
        env = os.environ.copy()
        env.update({
            "REPRODUCE_BEST_MODE": "1",
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/echo",
            "OUTPUT_ROOT": "results/e2e-reproduce-test",
        })

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("Full reproduction run", result.stdout)
        self.assertIn(
            "HRL output:        results/hrl_lookback60_hold30_inner_noaux_retrain/lookback60_hold30_inner_noaux_retrain",
            result.stdout,
        )
        self.assertIn("--frozen_hrl_checkpoint", result.stdout)
        self.assertIn("--controller_only_finetune", result.stdout)
        self.assertIn("Archived best floor: enabled", result.stdout)

    def test_reproduce_best_mode_refuses_to_write_into_archived_good_models(self):
        env = os.environ.copy()
        env.update({
            "REPRODUCE_BEST_MODE": "1",
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/echo",
            "OUTPUT_ROOT": "results/end",
        })

        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

        self.assertNotEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("Refusing to write", result.stdout)
        self.assertIn("results/end", result.stdout)

if __name__ == "__main__":
    unittest.main()
