import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_hrl_fixed60_inner_noaux_retrain.sh")


class InnerNoauxRetrainScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(SCRIPT, "r", encoding="utf-8") as fh:
            cls.script = fh.read()

    def _default_value(self, name):
        match = re.search(rf'{name}="\$\{{{name}:-([^}}]+)\}}"', self.script)
        self.assertIsNotNone(match, f"{name} default not found")
        return match.group(1)

    def test_inner_warmup_uses_dense_parallel_600_day_episode_batches(self):
        self.assertEqual(self._default_value("INNER_SEGMENTS_PER_EPISODE"), "20")
        self.assertEqual(self._default_value("INNER_EPISODE_LEN"), "$((MAX_HOLD * INNER_SEGMENTS_PER_EPISODE))")
        self.assertEqual(self._default_value("INNER_START_STRIDE_DAYS"), "1")
        self.assertEqual(self._default_value("INNER_EPISODE_BATCH_SIZE"), "12")
        self.assertEqual(self._default_value("INNER_EPISODE_PARALLEL_WORKERS"), "12")
        self.assertEqual(self._default_value("INNER_ROLLOUT_UPDATE_STEPS"), "$INNER_EPISODE_LEN")
        self.assertIn('--inner_episode_len "$INNER_EPISODE_LEN"', self.script)
        self.assertIn('--inner_start_stride_days "$INNER_START_STRIDE_DAYS"', self.script)
        self.assertIn('--inner_episode_batch_size "$INNER_EPISODE_BATCH_SIZE"', self.script)
        self.assertIn('--inner_rollout_update_steps "$INNER_ROLLOUT_UPDATE_STEPS"', self.script)
        self.assertIn("--no_train_controller", self.script)


if __name__ == "__main__":
    unittest.main()
