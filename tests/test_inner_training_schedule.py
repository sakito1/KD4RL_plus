import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import HRL_Trainer
from env.PPO_env import PPO_Env


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

    def test_fixed_train_pool_uses_offsets_then_cuts_long_sequences_into_episodes(self):
        starts = PPO_Env._build_fixed_train_pool(
            raw_indices=list(range(100, 3730)),
            total_days=4300,
            episode_len=600,
            stride_days=1,
            start_offsets=30,
        )

        self.assertEqual(starts[:8], [100, 700, 1300, 1900, 2500, 3100, 101, 701])
        self.assertEqual(len(starts), 180)
        self.assertTrue(all(start + 600 <= 3730 for start in starts))


class DummyBatchBuffer:
    def __init__(self):
        self.data = {"rew_mon": []}
        self.mark_count = 0
        self.clear_count = 0

    def mark_episode_start(self):
        self.mark_count += 1

    def get_batch(self):
        return {"rew_mon": list(self.data["rew_mon"])}

    def clear(self):
        self.clear_count += 1
        self.data = {"rew_mon": []}


class DummyBatchAgent:
    def __init__(self):
        self.update_calls = []

    def update(self, batch, phase, train_monitor=None):
        self.update_calls.append((len(batch["rew_mon"]), phase, train_monitor))
        return {"inn_pi": float(len(batch["rew_mon"])), "inn_pred": 0.5}


class DummyFinishedBuffer:
    def __init__(self, value):
        self.data = {"rew_mon": [value]}


class InnerEpisodeBatchUpdateTests(unittest.TestCase):
    def test_fixed_inner_pool_branch_accepts_single_episode_batches(self):
        train_path = os.path.join(ROOT, "Train", "PPO_train.py")
        with open(train_path, "r", encoding="utf-8") as fh:
            source = fh.read()

        self.assertIn("if inner_train_fixed_episodes:", source)
        self.assertNotIn("if inner_train_fixed_episodes and inner_episode_batch_size > 1:", source)

    def test_inner_episode_batch_updates_once_after_all_batch_episodes(self):
        trainer = object.__new__(HRL_Trainer)
        trainer.buffer = DummyBatchBuffer()
        trainer.agent = DummyBatchAgent()
        trainer.cfg = object()
        trainer.env = object()
        trainer.run_calls = []

        def fake_run_episode(env, **kwargs):
            trainer.run_calls.append(kwargs)
            trainer.buffer.data["rew_mon"].append(len(trainer.run_calls))
            return {"total_steps": 300, "history": [1.0, 1.1], "loss_log": {}, "update_count": 0}

        trainer.run_episode = fake_run_episode

        result = trainer._run_inner_episode_batch(
            episode_count=4,
            fixed_cycle=30,
            train_monitor=False,
        )

        self.assertEqual(len(trainer.run_calls), 4)
        self.assertEqual(len(trainer.agent.update_calls), 1)
        self.assertEqual(trainer.agent.update_calls[0], (4, "warmup_inner", False))
        self.assertEqual(trainer.buffer.clear_count, 1)
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["update_count"], 1)

    def test_inner_episode_batch_parallel_workers_collect_distinct_windows_then_update_once(self):
        trainer = object.__new__(HRL_Trainer)
        trainer.buffer = DummyBatchBuffer()
        trainer.agent = DummyBatchAgent()
        trainer.cfg = SimpleNamespace(inner_episode_parallel_workers=4, clear_cuda_cache_on_update=False)
        trainer.device = SimpleNamespace(type="cpu")
        trainer.env = SimpleNamespace(
            train_indices_pool=[10, 20, 30, 40],
            train_ptr=0,
            train_episode_to_end=False,
            episode_len=300,
        )
        trainer.worker_calls = []

        def sequential_run_episode(*args, **kwargs):
            raise AssertionError("parallel inner batch should not use sequential run_episode")

        def fake_worker(start_idx, stop_idx, fixed_cycle, train_monitor):
            trainer.worker_calls.append((start_idx, stop_idx, fixed_cycle, train_monitor))
            return {
                "episodes": 1,
                "total_steps": stop_idx - start_idx,
                "history": [1.0, 1.1],
                "loss_log": {},
                "update_count": 0,
                "buffer": DummyFinishedBuffer(start_idx),
            }

        trainer.run_episode = sequential_run_episode
        trainer._run_inner_episode_worker = fake_worker

        result = trainer._run_inner_episode_batch(
            episode_count=4,
            fixed_cycle=30,
            train_monitor=False,
        )

        self.assertEqual(
            sorted(trainer.worker_calls),
            [(10, 310, 30, False), (20, 320, 30, False), (30, 330, 30, False), (40, 340, 30, False)],
        )
        self.assertEqual(trainer.agent.update_calls, [(4, "warmup_inner", False)])
        self.assertEqual(result["episodes"], 4)
        self.assertEqual(result["update_count"], 1)


class JointFullTrainEpisodeTests(unittest.TestCase):
    def test_joint_full_train_episode_uses_single_train_start_and_restores_schedule(self):
        trainer = object.__new__(HRL_Trainer)
        trainer.env = DummyEnv()
        trainer.cfg = object()
        trainer.run_calls = []

        def fake_run_episode(env, **kwargs):
            trainer.run_calls.append((
                kwargs,
                env.train_episode_to_end,
                env.train_episodes_per_epoch,
                env.train_start_stride_days,
            ))
            return {"history": [1.0, 1.2], "loss_log": {"out_pi": 1.0}, "update_count": 1}

        trainer.run_episode = fake_run_episode

        ret = trainer._run_joint_full_train_episode(
            fixed_cycle=30,
            rollout_update_steps=300,
            train_monitor=False,
        )

        self.assertEqual(ret["history"], [1.0, 1.2])
        self.assertEqual(len(trainer.run_calls), 1)
        kwargs, to_end, episodes_per_epoch, stride = trainer.run_calls[0]
        self.assertEqual(kwargs["phase"], "joint")
        self.assertTrue(to_end)
        self.assertEqual(episodes_per_epoch, 1)
        self.assertEqual(stride, 1)
        self.assertTrue(trainer.env.train_episode_to_end)
        self.assertEqual(trainer.env.train_episodes_per_epoch, 5)
        self.assertEqual(trainer.env.train_start_stride_days, 10)


if __name__ == "__main__":
    unittest.main()
