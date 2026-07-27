import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_hrl_training
from agent.PPO_agent import HRL_Buffer


class RunHrlTrainingCommandTests(unittest.TestCase):
    def _args_from_cli(self, extra_args):
        argv = ["run_hrl_training.py", *extra_args]
        with patch.object(sys, "argv", argv):
            args = run_hrl_training.parse_args()
        run_hrl_training.normalize_training_schedule(args)
        return args

    def test_disabled_controller_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--no_train_controller",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=42,
        )

        self.assertIn("--no_train_controller", command)
        self.assertNotIn("--train_monitor", command)
        monitor_epochs_idx = command.index("--warmup_monitor_epochs") + 1
        self.assertEqual(command[monitor_epochs_idx], "0")

    def test_enabled_controller_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--train_monitor",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=42,
        )

        self.assertIn("--train_monitor", command)
        self.assertNotIn("--no_train_controller", command)

    def test_end_to_end_controller_joint_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--train_monitor",
            "--end_to_end_controller_joint",
            "--controller_joint_epochs",
            "2",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        self.assertIn("--end_to_end_controller_joint", command)
        self.assertEqual(command[command.index("--controller_joint_epochs") + 1], "2")
        self.assertNotIn("--frozen_hrl_checkpoint", command)
        self.assertNotIn("--controller_only_finetune", command)
        self.assertNotIn("--controller_first_joint_finetune", command)

    def test_controller_deterministic_rollout_sampling_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--controller_deterministic_rollout_sampling",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=49,
        )

        self.assertIn("--controller_deterministic_rollout_sampling", command)

    def test_controller_guidance_probe_is_forwarded_with_topk(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--frozen_hrl_checkpoint", "/tmp/outer.pth",
            "--controller_guidance_probe_only",
            "--controller_guidance_topk", "20",
            "--controller_rollout_len", "300",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=77,
        )

        self.assertIn("--controller_guidance_probe_only", command)
        self.assertEqual(
            command[command.index("--controller_guidance_topk") + 1],
            "20",
        )
        self.assertEqual(
            command[command.index("--controller_rollout_len") + 1],
            "300",
        )

    def test_controller_economic_guidance_thresholds_are_forwarded(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--controller_guidance_risk_threshold", "0.10",
            "--controller_guidance_risk_min_advantage_threshold", "0.02",
            "--controller_guidance_advantage_threshold", "0.10",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=44,
        )

        self.assertEqual(
            command[command.index("--controller_guidance_risk_threshold") + 1],
            "0.1",
        )
        self.assertEqual(
            command[
                command.index(
                    "--controller_guidance_risk_min_advantage_threshold"
                ) + 1
            ],
            "0.02",
        )
        self.assertEqual(
            command[command.index("--controller_guidance_advantage_threshold") + 1],
            "0.1",
        )

    def test_frozen_hrl_checkpoint_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--frozen_hrl_checkpoint",
            "/tmp/frozen/hrl_fixed_best.pth",
            "--controller_only_finetune",
            "--controller_no_hold_constraints",
            "--controller_train_fixed_episodes",
            "--controller_train_max_hold",
            "0",
            "--controller_train_record_max_duration",
            "30",
            "--controller_eval_max_hold",
            "0",
            "--controller_compute_switch_advantage",
            "--controller_episode_batch_size",
            "12",
            "--controller_episode_parallel_workers",
            "12",
            "--controller_window",
            "30",
            "--controller_fixed_pool_limit",
            "8",
            "--controller_hidden_dim",
            "128",
            "--controller_init_exit_bias",
            "-1.25",
            "--controller_pg_logprob_reduction",
            "sum",
            "--controller_eval_decision_mode",
            "daily",
            "--controller_aux_return_coef",
            "0.2",
            "--controller_aux_mdd_coef",
            "0.2",
            "--controller_aux_switch_adv_coef",
            "0.15",
            "--controller_aux_switch_adv_loss_type",
            "mse",
            "--controller_switch_adv_logit_coef",
            "2.0",
            "--controller_switch_adv_logit_scale",
            "0.02",
            "--controller_switch_adv_logit_detach",
            "--controller_aux_pretrain_offpolicy",
            "--controller_aux_replay_epochs",
            "4",
            "--controller_local_adv_coef",
            "0.3",
            "--controller_local_adv_margin",
            "0.01",
            "--controller_local_adv_loss_type",
            "weighted_bce",
            "--controller_local_adv_balance_classes",
            "--controller_expected_switch_penalty_coef",
            "0.004",
            "--controller_overflow_action_penalty_coef",
            "0.02",
            "--controller_value_coef",
            "0.25",
            "--no_controller_value_normalize_advantage",
            "--controller_max_switches",
            "40",
            "--controller_eval_diagnostics",
            "--controller_eval_diag_thresholds",
            "0.5",
            "0.45",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        checkpoint_idx = command.index("--frozen_hrl_checkpoint") + 1
        self.assertEqual(command[checkpoint_idx], "/tmp/frozen/hrl_fixed_best.pth")
        self.assertIn("--controller_only_finetune", command)
        self.assertNotIn("--controller_first_joint_finetune", command)
        self.assertIn("--controller_no_hold_constraints", command)
        self.assertIn("--controller_train_fixed_episodes", command)
        self.assertEqual(command[command.index("--controller_train_max_hold") + 1], "0")
        self.assertEqual(command[command.index("--controller_train_record_max_duration") + 1], "30")
        self.assertEqual(command[command.index("--controller_eval_max_hold") + 1], "0")
        self.assertIn("--controller_compute_switch_advantage", command)
        self.assertEqual(command[command.index("--controller_episode_batch_size") + 1], "12")
        self.assertEqual(command[command.index("--controller_episode_parallel_workers") + 1], "12")
        self.assertEqual(command[command.index("--controller_window") + 1], "30")
        self.assertEqual(command[command.index("--controller_fixed_pool_limit") + 1], "8")
        self.assertEqual(command[command.index("--controller_hidden_dim") + 1], "128")
        self.assertEqual(command[command.index("--controller_init_exit_bias") + 1], "-1.25")
        self.assertEqual(command[command.index("--controller_pg_logprob_reduction") + 1], "sum")
        self.assertEqual(command[command.index("--controller_eval_decision_mode") + 1], "daily")
        self.assertEqual(command[command.index("--controller_aux_return_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_aux_mdd_coef") + 1], "0.2")
        self.assertEqual(command[command.index("--controller_aux_switch_adv_coef") + 1], "0.15")
        self.assertEqual(command[command.index("--controller_aux_switch_adv_loss_type") + 1], "mse")
        self.assertEqual(command[command.index("--controller_switch_adv_logit_coef") + 1], "2.0")
        self.assertEqual(command[command.index("--controller_switch_adv_logit_scale") + 1], "0.02")
        self.assertIn("--controller_switch_adv_logit_detach", command)
        self.assertIn("--controller_aux_pretrain_offpolicy", command)
        self.assertEqual(command[command.index("--controller_aux_replay_epochs") + 1], "4")
        self.assertEqual(command[command.index("--controller_local_adv_coef") + 1], "0.3")
        self.assertEqual(command[command.index("--controller_local_adv_margin") + 1], "0.01")
        self.assertEqual(command[command.index("--controller_local_adv_loss_type") + 1], "weighted_bce")
        self.assertIn("--controller_local_adv_balance_classes", command)
        self.assertEqual(command[command.index("--controller_expected_switch_penalty_coef") + 1], "0.004")
        self.assertEqual(command[command.index("--controller_overflow_action_penalty_coef") + 1], "0.02")
        self.assertEqual(command[command.index("--controller_value_coef") + 1], "0.25")
        self.assertIn("--no_controller_value_normalize_advantage", command)
        self.assertEqual(command[command.index("--controller_max_switches") + 1], "40")
        self.assertIn("--controller_eval_diagnostics", command)
        diag_thresholds_idx = command.index("--controller_eval_diag_thresholds") + 1
        self.assertEqual(command[diag_thresholds_idx:diag_thresholds_idx + 2], ["0.5", "0.45"])
        self.assertNotIn("--controller_use_switch_supervision", command)

    def test_controller_switch_supervision_requires_explicit_legacy_flag(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--frozen_hrl_checkpoint",
            "/tmp/frozen/hrl_fixed_best.pth",
            "--controller_only_finetune",
            "--controller_sup_coef",
            "0.2",
            "--controller_use_switch_supervision",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        self.assertIn("--controller_use_switch_supervision", command)

    def test_economic_guidance_configuration_is_forwarded_to_child(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--controller_guidance_risk_threshold", "0.05",
            "--controller_guidance_advantage_threshold", "0.05",
            "--controller_guidance_pretrain_coef", "1.0",
            "--controller_sup_coef", "0.1",
            "--controller_use_switch_supervision",
            "--controller_sup_pretrain_rollout_len", "300",
            "--controller_rollout_len", "300",
            "--controller_aux_mdd_coef", "0.1",
            "--controller_aux_switch_adv_coef", "1.0",
            "--controller_entropy_coef", "0.01",
            "--controller_switch_rate_penalty_coef", "5.0",
            "--controller_switch_rate_min", "0.05",
            "--controller_switch_rate_max", "0.15",
            "--controller_switch_rate_margin", "0.1",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=77,
        )

        self.assertEqual(command[command.index("--controller_guidance_risk_threshold") + 1], "0.05")
        self.assertEqual(command[command.index("--controller_guidance_advantage_threshold") + 1], "0.05")
        self.assertEqual(command[command.index("--controller_guidance_pretrain_coef") + 1], "1.0")
        self.assertIn("--controller_use_switch_supervision", command)
        self.assertEqual(command[command.index("--controller_sup_coef") + 1], "0.1")
        self.assertEqual(command[command.index("--controller_sup_pretrain_rollout_len") + 1], "300")
        self.assertEqual(command[command.index("--controller_rollout_len") + 1], "300")
        self.assertEqual(command[command.index("--controller_switch_rate_penalty_coef") + 1], "5.0")
        self.assertEqual(command[command.index("--controller_switch_rate_min") + 1], "0.05")
        self.assertEqual(command[command.index("--controller_switch_rate_max") + 1], "0.15")
        self.assertEqual(command[command.index("--controller_switch_rate_margin") + 1], "0.1")

    def test_controller_pretrain_only_is_forwarded_to_child(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--controller_pretrain_only",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=77,
        )

        self.assertIn("--controller_pretrain_only", command)

    def test_test_skip_fixed_scenarios_is_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "sh",
            "--test_only_checkpoint",
            "/tmp/controller/best_model.pth",
            "--test_skip_fixed_scenarios",
            "--test_max_days",
            "240",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="sh",
            run_root=Path("/tmp/hrl-test"),
            seed=90,
        )

        self.assertIn("--test_skip_fixed_scenarios", command)
        self.assertEqual(command[command.index("--test_max_days") + 1], "240")

    def test_reward_modes_are_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--outer_reward_mode",
            "segment_sharpe",
            "--controller_reward_mode",
            "relative_return_mdd",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=49,
        )

        self.assertEqual(command[command.index("--outer_reward_mode") + 1], "segment_sharpe")
        self.assertEqual(command[command.index("--controller_reward_mode") + 1], "relative_return_mdd")

    def test_downside_controller_reward_args_are_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--controller_reward_mode",
            "relative_downside_mdd",
            "--controller_downside_coef",
            "0.5",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=50,
        )

        self.assertEqual(command[command.index("--controller_reward_mode") + 1], "relative_downside_mdd")
        self.assertEqual(command[command.index("--controller_downside_coef") + 1], "0.5")

    def test_staged_controller_and_inner_flags_are_forwarded_to_child_command(self):
        args = self._args_from_cli([
            "--markets", "nas",
            "--frozen_hrl_checkpoint",
            "/tmp/stage/outer/best_model.pth",
            "--controller_only_finetune",
            "--controller_pg_disable_inner",
            "--test_controller_no_inner_scenario",
        ])

        command = run_hrl_training.build_child_command(
            args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=41,
        )

        self.assertIn("--controller_pg_disable_inner", command)
        self.assertIn("--test_controller_no_inner_scenario", command)

        inner_args = self._args_from_cli([
            "--markets", "nas",
            "--frozen_hrl_checkpoint",
            "/tmp/stage/controller/best_model.pth",
            "--inner_only_finetune",
        ])
        inner_command = run_hrl_training.build_child_command(
            inner_args,
            market="nas",
            run_root=Path("/tmp/hrl-test"),
            seed=41,
        )

        self.assertIn("--inner_only_finetune", inner_command)
        self.assertNotIn("--controller_only_finetune", inner_command)

    def test_outer_segment_sharpe_reward_mode_uses_segment_returns(self):
        buffer = HRL_Buffer(
            capacity=16,
            device=torch.device("cpu"),
            gamma=1.0,
            gae_lambda=1.0,
            outer_reward_scale=1.0,
            outer_reward_mode="segment_sharpe",
        )
        for idx, ret in enumerate([0.01, 0.03, -0.01]):
            buffer.store_daily({
                "rew_mon": torch.tensor(0.0),
                "rew_alpha": torch.tensor(0.0),
                "rew_outer_raw": torch.tensor(ret),
                "val_mon": torch.tensor(0.0),
                "val_inn": torch.tensor(0.0),
                "val_out": torch.tensor(0.0),
                "is_switch": torch.tensor(1 if idx == 0 else 0),
                "dones": torch.tensor(1 if idx == 2 else 0),
            })

        buffer.finish_episode({"val_mon": 0.0, "val_inn": 0.0, "val_out": 0.0})

        returns = torch.tensor([0.01, 0.03, -0.01])
        expected = returns.mean() / (returns.std(unbiased=False) + 1e-8) * (252.0 ** 0.5)
        self.assertAlmostEqual(buffer.data["ret_out"][0].item(), expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
