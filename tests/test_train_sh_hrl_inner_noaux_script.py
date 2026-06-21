import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "train_sh", "run_hrl_fixed60_inner_noaux_retrain.sh")


class TrainShInnerNoauxRetrainScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCRIPT, "r", encoding="utf-8") as fh:
            cls.script = fh.read()

    def _default_value(self, name):
        match = re.search(rf'{name}="\$\{{{name}:-([^}}]+)\}}"', self.script)
        self.assertIsNotNone(match, f"{name} default not found")
        return match.group(1)

    def test_defaults_target_best_hrl_seeds(self):
        self.assertEqual(self._default_value("NAS_SEEDS"), "49")
        self.assertEqual(self._default_value("SH_SEEDS"), "90")

    def test_defaults_to_reproduction_output_root(self):
        self.assertEqual(
            self._default_value("OUTPUT_ROOT"),
            "results/hrl_reproduce_good_hrl_nas49_sh90",
        )

    def test_runs_both_markets_with_good_hrl_recipe(self):
        self.assertIn('--markets sh', self.script)
        self.assertIn('--markets nas', self.script)
        self.assertIn('--warmup_outer_epochs "$WARMUP_OUTER_EPOCHS"', self.script)
        self.assertIn('--warmup_inner_epochs "$WARMUP_INNER_EPOCHS"', self.script)
        self.assertIn('--inner_episode_len "$INNER_EPISODE_LEN"', self.script)
        self.assertIn('--inner_train_episodes_per_epoch "$INNER_TRAIN_EPISODES_PER_EPOCH"', self.script)
        self.assertIn('--inner_episode_batch_size "$INNER_EPISODE_BATCH_SIZE"', self.script)
        self.assertIn('--inner_episode_parallel_workers "$INNER_EPISODE_PARALLEL_WORKERS"', self.script)
        self.assertIn("--no_train_controller", self.script)
        self.assertIn("--model_selection_metric sharpe", self.script)
        self.assertIn("--inner_selection_metric return", self.script)

    def test_good_hrl_numeric_defaults(self):
        self.assertEqual(self._default_value("OUTER_WINDOW"), "60")
        self.assertEqual(self._default_value("MIN_HOLD"), "30")
        self.assertEqual(self._default_value("MAX_HOLD"), "30")
        self.assertEqual(self._default_value("WARMUP_OUTER_EPOCHS"), "2")
        self.assertEqual(self._default_value("WARMUP_INNER_EPOCHS"), "2")
        self.assertEqual(self._default_value("INNER_SEGMENTS_PER_EPISODE"), "20")
        self.assertEqual(self._default_value("INNER_START_STRIDE_DAYS"), "1")
        self.assertEqual(self._default_value("INNER_TRAIN_EPISODES_PER_EPOCH"), "30")
        self.assertEqual(self._default_value("INNER_EPISODE_BATCH_SIZE"), "12")
        self.assertEqual(self._default_value("INNER_EPISODE_PARALLEL_WORKERS"), "12")


if __name__ == "__main__":
    unittest.main()
