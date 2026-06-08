import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import HRL_Trainer


class DummyEnv:
    def __init__(self):
        self.train_episode_to_end = True
        self.train_episodes_per_epoch = 5
        self.train_episode_count = 5
        self.train_start_stride_days = 10
        self.train_episode_start_stride = 10
        self.episode_len = 960
        self.stride = 200
        self.train_ptr = 3
        self.completed_train_epoch_count = 2
        self._train_pool_signature = ("old",)
        self.mode_calls = []

    def set_mode(self, mode_name):
        self.mode_calls.append((
            mode_name,
            self.train_episode_to_end,
            self.train_episodes_per_epoch,
            self.train_start_stride_days,
            self.episode_len,
            self.stride,
        ))


class InnerTrainingScheduleTests(unittest.TestCase):
    def test_inner_schedule_temporarily_reconfigures_and_restores_train_pool(self):
        trainer = object.__new__(HRL_Trainer)
        trainer.env = DummyEnv()

        previous = trainer._apply_train_episode_config(
            train_episode_to_end=False,
            train_episodes_per_epoch=30,
            train_start_stride_days=120,
            episode_len=400,
        )

        self.assertFalse(trainer.env.train_episode_to_end)
        self.assertEqual(trainer.env.train_episodes_per_epoch, 30)
        self.assertEqual(trainer.env.train_start_stride_days, 120)
        self.assertEqual(trainer.env.episode_len, 400)
        self.assertEqual(trainer.env.stride, 120)
        self.assertFalse(hasattr(trainer.env, "_train_pool_signature"))
        self.assertEqual(trainer.env.mode_calls[-1], ("train", False, 30, 120, 400, 120))

        trainer._restore_train_episode_config(previous)

        self.assertTrue(trainer.env.train_episode_to_end)
        self.assertEqual(trainer.env.train_episodes_per_epoch, 5)
        self.assertEqual(trainer.env.train_start_stride_days, 10)
        self.assertEqual(trainer.env.episode_len, 960)
        self.assertEqual(trainer.env.stride, 200)
        self.assertFalse(hasattr(trainer.env, "_train_pool_signature"))
        self.assertEqual(trainer.env.mode_calls[-1], ("train", True, 5, 10, 960, 200))


if __name__ == "__main__":
    unittest.main()
