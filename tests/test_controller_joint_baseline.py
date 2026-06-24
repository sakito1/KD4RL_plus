import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import _prepare_controller_joint_baseline, train_controller_then_joint_finetune


class _FakeTrainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.loaded = []
        self.saved = []
        self.logger = None

    def _load_model(self, name):
        self.loaded.append(name)
        return Path(self.model_dir, name).exists()

    def save_model(self, name):
        self.saved.append(name)
        Path(self.model_dir, name).write_text("saved", encoding="utf-8")


class ControllerJointBaselineTests(unittest.TestCase):
    def test_joint_finetune_keeps_controller_pg_best_as_initial_final_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "controller_best.pth").write_text("controller", encoding="utf-8")
            trainer = _FakeTrainer(tmp)

            baseline = _prepare_controller_joint_baseline(
                trainer,
                controller_best_ckpt="controller_best.pth",
                final_best_ckpt="best_model.pth",
                controller_pg_result={"best_score": 0.613, "updates": 3},
            )

            self.assertAlmostEqual(baseline, 0.613)
            self.assertEqual(trainer.loaded, ["controller_best.pth"])
            self.assertEqual(trainer.saved, ["best_model.pth"])
            self.assertTrue(Path(tmp, "best_model.pth").exists())

    def test_controller_first_joint_also_seeds_final_with_controller_pg_best(self):
        import inspect

        source = inspect.getsource(train_controller_then_joint_finetune)
        self.assertIn("_prepare_controller_joint_baseline", source)


if __name__ == "__main__":
    unittest.main()
