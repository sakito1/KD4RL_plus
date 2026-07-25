import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass


class _FakeAgent:
    def __init__(self):
        self.statuses = []
        self.lr_multipliers = []
        self.net = SimpleNamespace(eval=lambda: None, train=lambda: None)

    def set_module_status(self, status):
        self.statuses.append(status)

    def set_lr_multiplier(self, multiplier):
        self.lr_multipliers.append(multiplier)
        return {"monitor": multiplier, "outer": multiplier, "inner": multiplier}


class _FakeEnv:
    def __init__(self):
        self.modes = []

    def set_mode(self, mode):
        self.modes.append(mode)


class _FakeJointTrainer(_FakeTrainer):
    def __init__(self, model_dir):
        super().__init__(model_dir)
        self.logger = _NoopLogger()
        self.agent = _FakeAgent()
        self.env = _FakeEnv()
        self.run_episode_calls = []
        self.cfg = SimpleNamespace(
            train_episodes_per_epoch=1,
            val_interval=1,
            controller_pg_disable_inner=True,
            joint_epochs=1,
            rollout_update_steps_by_stage={},
            rollout_update_steps=0,
            joint_lr_mult=1.0,
            controller_selection_metric="return",
        )

    def run_episode(self, env, **kwargs):
        self.run_episode_calls.append(kwargs)
        return {
            "history": [1000.0, 1001.0],
            "loss_log": {},
            "update_count": 0,
            "switch_count": 0,
        }

    def _compute_metrics(self, history):
        return {"sharpe": 0.0, "total_ret": 0.001, "max_dd": 0.0}

    def _validation_score(self, metrics, cfg, phase="joint"):
        return float(metrics["total_ret"])


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

    def test_controller_first_joint_disables_inner_during_joint_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _FakeJointTrainer(tmp)

            train_controller_then_joint_finetune(
                trainer,
                controller_episodes=0,
                joint_episodes=1,
                fixed_cycle=30,
                val_interval=1,
                train_monitor=True,
            )

            joint_calls = [
                call for call in trainer.run_episode_calls
                if call.get("phase") == "joint" and call.get("fixed_cycle") is None
            ]
            self.assertEqual(len(joint_calls), 2)
            self.assertTrue(all(call.get("disable_inner") is True for call in joint_calls))


if __name__ == "__main__":
    unittest.main()
