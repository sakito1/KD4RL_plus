import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_controller_daily_aux_pg_from_noaux_retrain.sh"


def _clean_env():
    env = os.environ.copy()
    for key in [
        "RUN_NAME",
        "CONTROLLER_FIXED_POOL_LIMIT",
        "CONTROLLER_INIT_EXIT_BIAS",
        "CONTROLLER_MAX_SWITCH_PENALTY_COEF",
        "CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF",
        "CONTROLLER_ENTROPY_COEF",
        "CONTROLLER_VALUE_NORMALIZE_ADVANTAGE",
        "CONTROLLER_EVAL_SWITCH_THRESHOLD",
        "CONTROLLER_EVAL_DIAGNOSTICS",
        "CONTROLLER_EVAL_DIAG_THRESHOLDS",
        "CONTROLLER_TEST_THRESHOLDS",
        "CONTROLLER_TRAIN_MAX_HOLD",
        "CONTROLLER_TRAIN_RECORD_MAX_DURATION",
        "CONTROLLER_EVAL_MAX_HOLD",
        "CONTROLLER_COMPUTE_SWITCH_ADVANTAGE",
        "CONTROLLER_SWITCH_ADV_LOGIT_COEF",
        "CONTROLLER_SWITCH_ADV_LOGIT_SCALE",
        "CONTROLLER_SWITCH_ADV_LOGIT_DETACH",
        "CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE",
        "CONTROLLER_EPISODE_BATCH_SIZE",
        "CONTROLLER_EPISODE_PARALLEL_WORKERS",
    ]:
        env.pop(key, None)
    return env


class ControllerDailyAuxPGScriptTests(unittest.TestCase):
    def test_unset_nas_seed_env_defaults_to_empty(self):
        with tempfile.TemporaryDirectory(prefix="controller-script-test-") as tmpdir:
            env = _clean_env()
            env.update({
                "PYTHON_BIN": "/bin/true",
                "OUTPUT_ROOT": tmpdir,
                "SH_SEEDS": "",
            })
            env.pop("NAS_SEEDS", None)

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
        self.assertIn("NAS seeds:", result.stdout)
        self.assertIn("SH seeds:", result.stdout)
        self.assertNotIn("unbound variable", result.stdout)

    def test_empty_seed_env_skips_market_loops(self):
        with tempfile.TemporaryDirectory(prefix="controller-script-test-") as tmpdir:
            env = _clean_env()
            env.update({
                "PYTHON_BIN": "/bin/true",
                "OUTPUT_ROOT": tmpdir,
                "NAS_SEEDS": "",
                "SH_SEEDS": "",
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
        self.assertIn("NAS seeds:", result.stdout)
        self.assertIn("SH seeds:", result.stdout)
        self.assertIn(
            "lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_b12_sh90_3ep",
            result.stdout,
        )
        self.assertIn("mode=controller_only", result.stdout)
        self.assertIn("joint_epochs=0", result.stdout)
        self.assertIn("CONTROLLER-ONLY", result.stdout)
        self.assertIn("init_exit_bias=-1.0", result.stdout)
        self.assertIn("episode_len=600", result.stdout)
        self.assertIn("offsets=30", result.stdout)
        self.assertIn("start_stride_days=5", result.stdout)
        self.assertIn("fixed_pool_limit=12", result.stdout)
        self.assertIn("decision_mode=daily", result.stdout)
        self.assertIn("eval_mode=daily", result.stdout)
        self.assertIn("batch=12", result.stdout)
        self.assertIn("workers=12", result.stdout)
        self.assertNotIn("decision_stride_schedule=10 10 10 5 5 5 1 1 1", result.stdout)
        self.assertIn("logprob_reduction=sum", result.stdout)
        self.assertIn("value_coef=0.0", result.stdout)
        self.assertIn("entropy_coef=0.0", result.stdout)
        self.assertIn("Aux warmup: epochs=1", result.stdout)
        self.assertIn("offpolicy=1", result.stdout)
        self.assertIn("replay_epochs=3", result.stdout)
        self.assertIn("switch_adv_coef=1.0", result.stdout)
        self.assertIn("switch_adv_loss=weighted_bce", result.stdout)
        self.assertIn("switch_adv_mining=1", result.stdout)
        self.assertIn("switch_adv_logit_coef=1.9", result.stdout)
        self.assertIn("switch_adv_logit_scale=0.02", result.stdout)
        self.assertIn("switch_adv_logit_detach=1", result.stdout)
        self.assertIn("coef=0.0", result.stdout)
        self.assertIn("margin=0.0", result.stdout)
        self.assertIn("expected_switch_penalty=0.0", result.stdout)
        self.assertIn("max switches=30", result.stdout)
        self.assertIn("max_switch_penalty=0.001", result.stdout)
        self.assertIn("overflow_action_penalty=0.0", result.stdout)
        self.assertIn("controller_train_max_hold=0", result.stdout)
        self.assertIn("train_record_max_duration=0", result.stdout)
        self.assertIn("controller_eval_max_hold=-1", result.stdout)
        self.assertIn("test_max_days=0", result.stdout)
        self.assertIn("skip_fixed_scenarios=0", result.stdout)
        self.assertIn("eval_threshold=0.5", result.stdout)
        self.assertIn("diagnostics=1", result.stdout)
        self.assertIn("diag_thresholds=0.5", result.stdout)
        self.assertNotIn("test_thresholds=", result.stdout)
        self.assertNotIn("Daily aux-PG controller run:", result.stdout)

    def test_controller_uses_single_eval_threshold_with_probability_diagnostics_without_sweeps(self):
        with tempfile.TemporaryDirectory(prefix="controller-script-test-") as tmpdir:
            source_root = Path(tmpdir) / "source"
            checkpoint = source_root / "sh" / "ppo" / "seed_90" / "checkpoints" / "hrl_fixed_best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

            env = _clean_env()
            env.update({
                "PYTHON_BIN": "/bin/echo",
                "SOURCE_ROOT": str(source_root),
                "OUTPUT_ROOT": str(Path(tmpdir) / "out"),
                "NAS_SEEDS": "",
                "SH_SEEDS": "90",
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
        self.assertIn(
            "--run_name lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_b12_sh90_3ep",
            result.stdout,
        )
        self.assertIn("--controller_fixed_pool_limit 12", result.stdout)
        self.assertIn("--controller_init_exit_bias -1.0", result.stdout)
        self.assertIn("--controller_max_switch_penalty_coef 0.001", result.stdout)
        self.assertIn("--controller_overflow_action_penalty_coef 0.0", result.stdout)
        self.assertIn("--controller_entropy_coef 0.0", result.stdout)
        self.assertIn("--controller_train_max_hold 0", result.stdout)
        self.assertIn("--controller_train_record_max_duration 0", result.stdout)
        self.assertIn("--controller_eval_max_hold -1", result.stdout)
        self.assertIn("--controller_aux_switch_adv_loss_type weighted_bce", result.stdout)
        self.assertIn("--controller_switch_adv_logit_coef 1.9", result.stdout)
        self.assertIn("--controller_switch_adv_logit_scale 0.02", result.stdout)
        self.assertIn("--controller_switch_adv_logit_detach", result.stdout)
        self.assertIn("--controller_compute_switch_advantage", result.stdout)
        self.assertIn("--no_controller_value_normalize_advantage", result.stdout)
        self.assertIn("--controller_eval_switch_threshold 0.5", result.stdout)
        self.assertIn("--controller_eval_diagnostics", result.stdout)
        self.assertIn("--controller_eval_diag_thresholds 0.5", result.stdout)
        self.assertNotIn("--controller_test_thresholds", result.stdout)


if __name__ == "__main__":
    unittest.main()
