import os
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_reproduce_hrl_controller_nas49_sh90.sh"


class ReproduceHRLControllerScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCRIPT, "r", encoding="utf-8") as fh:
            cls.script = fh.read()

    def _default_value(self, name):
        match = re.search(rf'{name}="\$\{{{name}:?-([^}}]+)\}}"', self.script)
        self.assertIsNotNone(match, f"{name} default not found")
        return match.group(1)

    def test_default_seed_mapping_reproduces_requested_two_seeds(self):
        self.assertEqual(self._default_value("NAS_SEEDS"), "49")
        self.assertEqual(self._default_value("SH_SEEDS"), "90")
        self.assertIn('run_fixed_hrl_seed nas "$seed"', self.script)
        self.assertIn('run_fixed_hrl_seed sh "$seed"', self.script)
        self.assertIn('run_controller_seed nas "$seed"', self.script)
        self.assertIn('run_controller_seed sh "$seed"', self.script)

    def test_stage_one_trains_fixed_outer_inner_hrl_without_controller(self):
        self.assertEqual(self._default_value("WARMUP_OUTER_EPOCHS"), "2")
        self.assertEqual(self._default_value("WARMUP_INNER_EPOCHS"), "2")
        self.assertEqual(self._default_value("INNER_EPISODE_LEN"), "$((MAX_HOLD * INNER_SEGMENTS_PER_EPISODE))")
        self.assertEqual(self._default_value("INNER_TRAIN_EPISODES_PER_EPOCH"), "30")
        self.assertEqual(self._default_value("INNER_EPISODE_BATCH_SIZE"), "12")
        self.assertEqual(self._default_value("INNER_EPISODE_PARALLEL_WORKERS"), "12")
        self.assertIn("--inner_train_fixed_episodes", self.script)
        self.assertIn("--no_train_controller", self.script)

    def test_stage_two_uses_successful_controller_pg_recipe(self):
        self.assertEqual(self._default_value("JOINT_EPOCHS"), "0")
        self.assertEqual(self._default_value("CONTROLLER_EPOCHS"), "3")
        self.assertEqual(self._default_value("CONTROLLER_FIXED_POOL_LIMIT"), "12")
        self.assertEqual(self._default_value("CONTROLLER_INIT_EXIT_BIAS"), "-1.0")
        self.assertEqual(self._default_value("CONTROLLER_SWITCH_ADV_LOGIT_COEF"), "1.9")
        self.assertEqual(self._default_value("CONTROLLER_SWITCH_ADV_LOGIT_SCALE"), "0.02")
        self.assertEqual(self._default_value("CONTROLLER_EVAL_SWITCH_THRESHOLD"), "0.5")
        self.assertEqual(self._default_value("CONTROLLER_VALUE_NORMALIZE_ADVANTAGE"), "0")
        self.assertIn("--controller_only_finetune", self.script)
        self.assertIn("--frozen_hrl_checkpoint \"$checkpoint\"", self.script)
        self.assertIn("--controller_compute_switch_advantage", self.script)
        self.assertIn("--controller_switch_adv_logit_detach", self.script)
        self.assertIn("--no_controller_value_normalize_advantage", self.script)

    def test_default_output_guard_prevents_stale_checkpoint_reuse(self):
        self.assertEqual(self._default_value("ALLOW_EXISTING_OUTPUT"), "0")
        self.assertIn("Refusing to reuse existing HRL run directory", self.script)
        self.assertIn("Refusing to reuse existing controller run directory", self.script)

    def test_archived_best_floor_is_explicit_and_preserves_candidate_outputs(self):
        self.assertEqual(self._default_value("USE_ARCHIVED_BEST_FLOOR"), "0")
        self.assertIn("candidate_before_archived_floor", self.script)
        self.assertIn("archived_best_floor.json", self.script)


if __name__ == "__main__":
    unittest.main()
