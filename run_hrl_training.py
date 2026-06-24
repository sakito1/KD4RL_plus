#!/usr/bin/env python
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import utils.config as runtime_config
import utils.config_Nas as nas_config
import utils.config_SH as sh_config
from Train.PPO_train import (
    HRL_Networks,
    HRL_Trainer,
    set_seed,
    train_controller_only_from_frozen,
    train_controller_then_joint_finetune,
    train_warmup_then_joint_with_monitor,
)
from agent import HRL_Buffer, HRL_PPO_Agent
from env import PPO_Env
from utils.Log import create_logger


ROOT = Path(__file__).resolve().parent
MARKET_CONFIGS = {
    "nas": ("NAS100", nas_config),
    "sh": ("A-share", sh_config),
}

TRAINING_STAGES = ("warmup_outer", "warmup_inner", "warmup_monitor", "joint")
DEFAULT_PHASE_ROLLOUT_SEGMENTS = {
    "warmup_outer": 10,
    "warmup_inner": 10,
    "warmup_monitor": 10,
    "joint": 10,
}


def default_train_start_stride_days(min_hold, max_hold):
    return max(1, (int(max_hold) - int(min_hold)) // 3)


def apply_market_config(config_module):
    for name, value in vars(config_module).items():
        if not name.startswith("__"):
            setattr(runtime_config, name, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HRL PPO training for NAS100 and A-share with fixed DeepAries-style train episodes."
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--market", choices=sorted(MARKET_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--run_root", default=None, help=argparse.SUPPRESS)

    parser.add_argument(
        "--markets",
        nargs="+",
        choices=sorted(MARKET_CONFIGS),
        default=["nas", "sh"],
        help="Markets to run in order. Default: nas sh.",
    )
    parser.add_argument(
        "--output_root",
        default="checkpoints/hrl_training",
        help="Root directory for HRL training outputs.",
    )
    parser.add_argument("--run_name", default=None, help="Run folder name. Default: current timestamp.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child runs.")
    parser.add_argument("--seed", type=int, default=None, help="Single seed shorthand. Ignored if --seeds is set.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run for each market.")

    parser.add_argument(
        "--warmup_outer_epochs",
        type=int,
        default=5,
        help="Outer warmup train epochs. One epoch runs all fixed train episodes once.",
    )
    parser.add_argument(
        "--warmup_inner_epochs",
        type=int,
        default=5,
        help="Inner warmup train epochs. One epoch runs all fixed train episodes once.",
    )
    parser.add_argument(
        "--warmup_monitor_epochs",
        type=int,
        default=10,
        help="Controller counterfactual-PG train epochs. Default: 10.",
    )
    parser.add_argument("--controller_epochs", type=int, default=None, help="Alias for --warmup_monitor_epochs.")
    parser.add_argument(
        "--joint_epochs",
        type=int,
        default=1,
        help="Outer+Inner joint finetune train epochs. Default: 1.",
    )
    parser.add_argument(
        "--joint_single_full_episode",
        dest="joint_single_full_episode",
        action="store_true",
        help="Run each joint epoch as one full train-to-end episode from the first train start.",
    )
    parser.add_argument(
        "--no_joint_single_full_episode",
        dest="joint_single_full_episode",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(joint_single_full_episode=False)
    parser.add_argument("--joint_outer_inner_epochs", type=int, default=None, help="Alias for --joint_epochs.")
    parser.add_argument(
        "--frozen_hrl_checkpoint",
        default=None,
        help=(
            "Optional pretrained HRL checkpoint. When set, load outer/inner from this checkpoint "
            "before training the controller."
        ),
    )
    parser.add_argument(
        "--controller_first_joint_finetune",
        action="store_true",
        help=(
            "Run the controller-first schedule: load frozen outer/inner, train controller, "
            "then jointly finetune controller/outer/inner."
        ),
    )
    parser.add_argument(
        "--controller_only_finetune",
        action="store_true",
        help="Load frozen outer/inner and train only the controller; no outer/inner joint finetune.",
    )
    parser.add_argument(
        "--end_to_end_controller_joint",
        action="store_true",
        help=(
            "In the from-scratch HRL schedule, continue after controller PG with a controller-active "
            "joint finetune of monitor/outer/inner."
        ),
    )
    parser.add_argument(
        "--controller_joint_epochs",
        type=int,
        default=0,
        help="Controller-active joint finetune epochs after from-scratch controller PG. Default: 0.",
    )
    parser.add_argument("--warmup_outer_episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--warmup_inner_episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--warmup_monitor_episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--joint_episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--val_interval", type=int, default=9)
    parser.add_argument(
        "--model_selection_metric",
        choices=["sharpe", "return", "mdd"],
        default="sharpe",
        help="Checkpoint selection metric for outer/inner/joint phases. Default: sharpe.",
    )
    parser.add_argument(
        "--inner_selection_metric",
        choices=["sharpe", "return", "mdd"],
        default="return",
        help="Checkpoint selection metric for inner warmup. Default: return.",
    )
    parser.add_argument(
        "--controller_selection_metric",
        choices=["risk_return", "return", "mdd", "sharpe"],
        default="risk_return",
        help="Checkpoint selection metric for controller PG. Default: risk_return.",
    )
    parser.add_argument(
        "--fixed_cycle",
        type=int,
        default=None,
        help="Legacy alias for --max_hold. If set, overrides --max_hold.",
    )
    parser.add_argument(
        "--min_hold",
        type=int,
        default=10,
        help="Minimum holding days before controller may switch. Default: 10.",
    )
    parser.add_argument(
        "--max_hold",
        type=int,
        default=40,
        help="Maximum holding days before forced switch. Default: 40.",
    )
    parser.add_argument(
        "--outer_window",
        type=int,
        default=40,
        help="Outer lookback window in trading days. Default: 40.",
    )
    parser.add_argument(
        "--outer_rollout_segments",
        type=int,
        default=None,
        help=(
            "Legacy fallback: run this many fixed-hold segments before each PPO update. "
            "If omitted, phase-specific defaults are used."
        ),
    )
    parser.add_argument("--warmup_outer_rollout_segments", type=int, default=None)
    parser.add_argument("--warmup_inner_rollout_segments", type=int, default=None)
    parser.add_argument("--warmup_monitor_rollout_segments", type=int, default=None)
    parser.add_argument("--joint_rollout_segments", type=int, default=None)

    parser.add_argument("--lr_monitor", type=float, default=1e-3)
    parser.add_argument("--lr_outer", type=float, default=1e-3)
    parser.add_argument("--lr_inner", type=float, default=1e-3)
    parser.add_argument(
        "--joint_lr_mult",
        type=float,
        default=0.001,
        help="Learning-rate multiplier applied to all PPO optimizers during joint fine-tuning. Default: 0.001.",
    )
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument(
        "--inner_ppo_epochs",
        type=int,
        default=3,
        help="PPO epochs used only during the inner warmup stage. Default: 3.",
    )
    parser.add_argument(
        "--inner_rollout_update_steps",
        type=int,
        default=160,
        help="Daily rollout steps accumulated before each inner warmup PPO update. Default: 160.",
    )
    parser.add_argument(
        "--inner_episode_batch_size",
        type=int,
        default=1,
        help=(
            "When inner fixed episodes are enabled, run this many fixed-length episodes "
            "before one Inner PPO update. Default 1 keeps legacy per-episode updates."
        ),
    )
    parser.add_argument(
        "--inner_episode_parallel_workers",
        type=int,
        default=1,
        help="Run up to this many inner warmup fixed episodes concurrently before one update. Default: 1.",
    )
    parser.add_argument("--inner_batch_size", type=int, default=1200)
    parser.add_argument(
        "--outer_update_batch_size",
        type=int,
        default=16,
        help="Mini-batch size for Outer PPO updates; keep small because each sample runs a long LSTM window.",
    )
    parser.add_argument("--trade_num", type=int, default=10)
    parser.add_argument("--ssm_dim", type=int, default=16)
    parser.add_argument(
        "--outer_pred_coef",
        type=float,
        default=0.1,
        help="SmoothL1 auxiliary weight for the outer return-prediction head against future max-hold stock returns.",
    )
    parser.add_argument(
        "--inner_pred_coef",
        type=float,
        default=0.0,
        help="SmoothL1 auxiliary weight for the inner return-prediction head against next-day stock returns.",
    )
    parser.add_argument(
        "--inner_pred_target_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to next-day stock-return targets before Inner return-prediction SmoothL1 supervision. "
            "Use 10.0 to make a 1%% log-return target supervise as roughly 0.1 score."
        ),
    )
    parser.add_argument(
        "--inner_gate_reg_coef",
        type=float,
        default=0.0,
        help="Deprecated compatibility arg. Inner actor now follows KD4RL and has no feature gate.",
    )
    parser.add_argument(
        "--inner_use_topk",
        action="store_true",
        help="Use the newer topK-cropped inner action space. Default off keeps the original KD4RL full-universe inner actor.",
    )
    parser.add_argument(
        "--inner_feature_gate",
        action="store_true",
        help="Deprecated compatibility flag. Ignored; inner actor follows KD4RL without feature gate.",
    )
    parser.add_argument(
        "--inner_norm_mode",
        choices=["legacy", "log_zscore"],
        default="legacy",
        help="Inner feature normalization. legacy matches /home/tongwenxuan/KD4RL; log_zscore uses the newer normalization.",
    )
    parser.add_argument(
        "--inner_train_fixed_episodes",
        dest="inner_train_fixed_episodes",
        action="store_true",
        help="Use fixed-length local train episodes for inner warmup. Default on.",
    )
    parser.add_argument(
        "--no_inner_train_fixed_episodes",
        dest="inner_train_fixed_episodes",
        action="store_false",
        help="Use the global train episode schedule for inner warmup.",
    )
    parser.set_defaults(inner_train_fixed_episodes=True)
    parser.add_argument(
        "--inner_episode_len",
        type=int,
        default=400,
        help="Fixed inner warmup episode length in trading days. Default: 400.",
    )
    parser.add_argument(
        "--inner_train_episodes_per_epoch",
        type=int,
        default=30,
        help="Fixed inner warmup episodes per epoch. Default: 30.",
    )
    parser.add_argument(
        "--inner_start_stride_days",
        type=int,
        default=120,
        help="Stride between fixed inner warmup episode starts. Default: 120.",
    )
    parser.add_argument("--controller_sup_coef", type=float, default=0.0)
    parser.add_argument(
        "--controller_use_switch_supervision",
        action="store_true",
        help=(
            "Legacy option: train switch head with environment-provided switch labels. "
            "Default off so switch decisions are learned end-to-end by controller PG."
        ),
    )
    parser.add_argument("--controller_sup_pretrain_epochs", type=int, default=0)
    parser.add_argument("--controller_sup_pretrain_rollout_len", type=int, default=0)
    parser.add_argument("--controller_aux_pretrain_offpolicy", action="store_true")
    parser.add_argument("--controller_aux_replay_epochs", type=int, default=1)
    parser.add_argument(
        "--controller_check_stride_days",
        type=int,
        default=1,
        help=(
            "Only run the trainable controller every N days inside the free holding window. "
            "Days between checks are forced-hold and locked. Use 1 for daily controller checks."
        ),
    )
    parser.add_argument(
        "--clear_cuda_cache_on_update",
        action="store_true",
        help="Call torch.cuda.empty_cache() after PPO updates. Default off for speed.",
    )
    parser.add_argument("--reward_scale_outer", type=float, default=100.0)
    parser.add_argument("--reward_scale_inner", type=float, default=2000.0)
    parser.add_argument("--reward_scale_controller", type=float, default=100.0)
    parser.add_argument("--controller_algorithm", choices=["pg"], default="pg")
    parser.add_argument("--controller_decision_mode", choices=["daily", "fixed_window", "stride"], default="daily")
    parser.add_argument("--controller_eval_decision_mode", choices=["daily", "fixed_window", "stride"], default=None)
    parser.add_argument("--controller_decision_stride", type=int, default=1)
    parser.add_argument("--controller_eval_decision_stride", type=int, default=0)
    parser.add_argument("--controller_decision_stride_schedule", nargs="*", type=int, default=None)
    parser.add_argument("--controller_fixed_decision_window", type=int, default=0)
    parser.add_argument(
        "--controller_no_hold_constraints",
        action="store_true",
        help=(
            "Disable min-hold for trainable controller decisions. max_hold remains a hard forced-switch "
            "boundary; controller_max_switches is used as a soft reward-penalty threshold."
        ),
    )
    parser.add_argument(
        "--controller_train_max_hold",
        type=int,
        default=-1,
        help=(
            "Controller-PG training-only max-hold override. -1 uses global max_hold; "
            "0 disables forced max-hold inside controller training rollouts; positive values override."
        ),
    )
    parser.add_argument(
        "--controller_train_record_max_duration",
        type=int,
        default=0,
        help=(
            "Controller-PG training-only log-prob record cap. 0 records all free decisions; "
            "positive values record only decisions with current holding duration below this cap."
        ),
    )
    parser.add_argument(
        "--controller_eval_max_hold",
        type=int,
        default=-1,
        help=(
            "Controller eval/test max-hold override for controller_no_hold_constraints. "
            "-1 uses global max_hold; 0 disables forced max-hold; positive values override."
        ),
    )
    parser.add_argument("--controller_rollout_len", type=int, default=400)
    parser.add_argument("--controller_max_segments", type=int, default=25)
    parser.add_argument("--controller_pg_batch_windows", type=int, default=4)
    parser.add_argument(
        "--controller_pg_logprob_reduction",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "How to reduce controller action log-probs inside one episode. "
            "'mean' preserves legacy behavior; 'sum' uses trajectory PG over all controller decisions."
        ),
    )
    parser.add_argument("--controller_windows_per_epoch", type=int, default=5)
    parser.add_argument("--controller_fixed_pool_limit", type=int, default=0)
    parser.add_argument("--controller_train_fixed_episodes", action="store_true")
    parser.add_argument("--controller_episode_batch_size", type=int, default=4)
    parser.add_argument("--controller_episode_parallel_workers", type=int, default=1)
    parser.add_argument(
        "--controller_deterministic_rollout_sampling",
        action="store_true",
        help="Sample controller PG rollout actions from per-window deterministic RNG seeds.",
    )
    parser.add_argument("--controller_start_stride_days", type=int, default=40)
    parser.add_argument("--controller_window", type=int, default=15)
    parser.add_argument("--controller_hidden_dim", type=int, default=32)
    parser.add_argument("--controller_init_exit_bias", type=float, default=None)
    parser.add_argument("--controller_entropy_coef", type=float, default=0.01)
    parser.add_argument("--controller_aux_return_coef", type=float, default=0.0)
    parser.add_argument("--controller_aux_mdd_coef", type=float, default=0.0)
    parser.add_argument("--controller_aux_switch_adv_coef", type=float, default=0.0)
    parser.add_argument(
        "--controller_aux_switch_adv_loss_type",
        choices=["smooth_l1", "weighted_bce", "bce"],
        default="smooth_l1",
    )
    parser.add_argument("--controller_switch_adv_logit_coef", type=float, default=0.0)
    parser.add_argument("--controller_switch_adv_logit_scale", type=float, default=0.02)
    parser.add_argument("--controller_switch_adv_logit_detach", action="store_true")
    parser.add_argument("--controller_compute_switch_advantage", action="store_true")
    parser.add_argument("--controller_aux_return_target_scale", type=float, default=1.0)
    parser.add_argument("--controller_aux_mdd_target_scale", type=float, default=1.0)
    parser.add_argument("--controller_aux_switch_adv_target_scale", type=float, default=1.0)
    parser.add_argument("--controller_local_adv_coef", type=float, default=0.0)
    parser.add_argument("--controller_local_adv_scale", type=float, default=0.05)
    parser.add_argument("--controller_local_adv_clip", type=float, default=10.0)
    parser.add_argument("--controller_local_adv_margin", type=float, default=0.0)
    parser.add_argument(
        "--controller_local_adv_loss_type",
        choices=["linear", "weighted_bce"],
        default="linear",
    )
    parser.add_argument("--controller_local_adv_balance_classes", action="store_true")
    parser.add_argument("--controller_expected_switch_penalty_coef", type=float, default=0.0)
    parser.add_argument("--controller_overflow_action_penalty_coef", type=float, default=0.0)
    parser.add_argument("--controller_value_coef", type=float, default=0.0)
    parser.add_argument(
        "--no_controller_value_normalize_advantage",
        dest="controller_value_normalize_advantage",
        action="store_false",
        help="Use raw reward-value residuals for controller actor-critic instead of normalized residuals.",
    )
    parser.set_defaults(controller_value_normalize_advantage=True)
    parser.add_argument("--controller_mdd_coef", type=float, default=0.0)
    parser.add_argument("--controller_return_coef", type=float, default=1.0)
    parser.add_argument("--controller_count_min", type=int, default=0)
    parser.add_argument("--controller_count_max", type=int, default=0)
    parser.add_argument("--controller_count_penalty_coef", type=float, default=0.5)
    parser.add_argument(
        "--controller_max_switches",
        type=int,
        default=0,
        help="Optional manual max switch count per controller rollout; 0 uses rollout_len // min_hold.",
    )
    parser.add_argument(
        "--controller_max_switch_penalty_coef",
        type=float,
        default=0.5,
        help="Penalty coefficient for max(0, actual_switches - max_allowed_switches)^2.",
    )
    parser.add_argument("--controller_switch_coef", type=float, default=0.0)
    parser.add_argument("--controller_turnover_coef", type=float, default=0.0)
    parser.add_argument("--controller_val_interval_epochs", type=int, default=1)
    parser.add_argument("--controller_skip_val", action="store_true")
    parser.add_argument("--controller_tau_min", type=float, default=0.5)
    parser.add_argument("--controller_tau_max", type=float, default=0.9)
    parser.add_argument("--controller_policy_temperature", type=float, default=10.0)
    parser.add_argument("--controller_eval_switch_threshold", type=float, default=0.5)
    parser.add_argument("--controller_eval_force_max_hold", action="store_true")
    parser.add_argument("--controller_eval_diagnostics", action="store_true")
    parser.add_argument("--controller_eval_diag_thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--controller_test_thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--controller_state_return_scale", type=float, default=0.05)
    parser.add_argument("--controller_state_drawdown_scale", type=float, default=0.10)
    parser.add_argument(
        "--outer_pred_coefs",
        nargs="+",
        type=float,
        default=None,
        help="Optional grid of outer stock-return prediction loss weights.",
    )
    parser.add_argument(
        "--inner_pred_coefs",
        nargs="+",
        type=float,
        default=None,
        help="Optional grid of inner next-day-return prediction loss weights.",
    )
    parser.add_argument(
        "--train_monitor",
        action="store_true",
        help="Compatibility flag; controller training is enabled by default.",
    )
    parser.add_argument(
        "--no_train_controller",
        dest="train_monitor",
        action="store_false",
        help="Disable PPO controller training and use only forced hold/switch constraints.",
    )
    parser.set_defaults(train_monitor=True)
    parser.add_argument(
        "--rule_switch_threshold",
        type=float,
        default=0.5,
        help="Threshold used by the rule switch monitor when --train_monitor is off.",
    )
    parser.add_argument(
        "--max_rule_consecutive_low",
        type=int,
        default=None,
        help="Override max consecutive low held_p days before the rule switch monitor forces a switch.",
    )

    parser.add_argument(
        "--train_episodes_per_epoch",
        type=int,
        default=5,
        help="Number of fixed train starts per epoch. Default: 5.",
    )
    parser.add_argument(
        "--train_start_stride_days",
        type=int,
        default=None,
        help=(
            "Calendar-index stride between fixed train episode starts. "
            "Default: (max_hold - min_hold) // 3."
        ),
    )
    parser.add_argument("--train_episode_count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train_episode_start_stride", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_train_episode_to_end",
        dest="train_episode_to_end",
        action="store_false",
        help="Use legacy fixed-length train episodes instead of running each train episode to train_end.",
    )
    parser.set_defaults(train_episode_to_end=True)
    parser.add_argument("--episode_len", type=int, default=None, help="Only used when train_episode_to_end is disabled.")
    parser.add_argument(
        "--nas_ssm_data_path",
        default=None,
        help="Optional NAS100 feature_ssm directory produced by alpha-state export.",
    )
    parser.add_argument(
        "--sh_ssm_data_path",
        default=None,
        help="Optional A-share feature_ssm directory produced by alpha-state export.",
    )
    parser.add_argument("--ssm_data_path", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--device", default="cuda", help="Device string for training. Default: cuda if available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU by hiding CUDA devices.")
    parser.add_argument("--skip_test", action="store_true", help="Train only; skip trainer.test after training.")
    parser.add_argument("--test_only_checkpoint", default=None, help="Skip training and only run trainer.test on this checkpoint.")
    parser.add_argument("--test_skip_fixed_scenarios", action="store_true")
    parser.add_argument("--test_max_days", type=int, default=0, help="Optional cap on test episode length for fast diagnostics; 0 uses the full test range.")
    parser.add_argument(
        "--heartbeat_seconds",
        type=int,
        default=60,
        help="Print a monitor heartbeat when a child process is quiet.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny connectivity run: short fixed episodes and one episode per phase.",
    )
    return parser.parse_args()


def stream_process(command, cwd, env, log_path, prefix, heartbeat_seconds):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(map(str, command)) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = queue.Queue()

        def reader():
            assert process.stdout is not None
            for line in process.stdout:
                lines.put(line)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        start = time.time()
        last_output = start
        while process.poll() is None or not lines.empty():
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if now - last_output >= heartbeat_seconds:
                    elapsed = int(now - start)
                    msg = f"[{prefix}] still running, elapsed={elapsed}s, log={log_path}"
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    last_output = now
                continue

            text = f"[{prefix}] {line}"
            print(text, end="", flush=True)
            log.write(text)
            log.flush()
            last_output = time.time()

        thread.join(timeout=2.0)
        return process.returncode


def resolve_seeds(args, config_module):
    if args.seeds:
        return args.seeds
    if args.seed is not None:
        return [args.seed]
    base_seed = int(getattr(config_module, "seed", 42))
    return [base_seed + i for i in range(5)]


def _coef_tag(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def supervised_coef_grid(args):
    outer_values = args.outer_pred_coefs if args.outer_pred_coefs is not None else [args.outer_pred_coef]
    inner_values = args.inner_pred_coefs if args.inner_pred_coefs is not None else [args.inner_pred_coef]
    combos = []
    for outer_coef in outer_values:
        for inner_coef in inner_values:
            tag = f"op{_coef_tag(outer_coef)}_ip{_coef_tag(inner_coef)}"
            combos.append((float(outer_coef), float(inner_coef), tag))
    return combos


def normalize_training_schedule(args):
    if getattr(args, "controller_epochs", None) is not None:
        args.warmup_monitor_epochs = int(args.controller_epochs)
    if getattr(args, "joint_outer_inner_epochs", None) is not None:
        args.joint_epochs = int(args.joint_outer_inner_epochs)
    args.controller_joint_epochs = max(0, int(getattr(args, "controller_joint_epochs", 0) or 0))
    if getattr(args, "train_episode_count", None) is not None:
        args.train_episodes_per_epoch = args.train_episode_count
    if getattr(args, "train_episode_start_stride", None) is not None:
        args.train_start_stride_days = args.train_episode_start_stride

    if getattr(args, "fixed_cycle", None) is not None:
        args.max_hold = int(args.fixed_cycle)
    args.min_hold = max(1, int(args.min_hold))
    args.max_hold = max(args.min_hold, int(args.max_hold))
    args.fixed_cycle = args.max_hold

    fallback_rollout_segments = getattr(args, "outer_rollout_segments", None)
    for stage in TRAINING_STAGES:
        attr = f"{stage}_rollout_segments"
        value = getattr(args, attr, None)
        if value is None:
            value = fallback_rollout_segments
        if value is None:
            value = DEFAULT_PHASE_ROLLOUT_SEGMENTS[stage]
        setattr(args, attr, max(1, int(value)))
    if fallback_rollout_segments is None:
        args.outer_rollout_segments = int(args.warmup_outer_rollout_segments)
    else:
        args.outer_rollout_segments = max(1, int(fallback_rollout_segments))
    args.rollout_segments_by_stage = {
        stage: int(getattr(args, f"{stage}_rollout_segments"))
        for stage in TRAINING_STAGES
    }
    args.rollout_update_steps_by_stage = {
        stage: int(args.max_hold) * segments
        for stage, segments in args.rollout_segments_by_stage.items()
    }
    args.inner_train_fixed_episodes = bool(args.inner_train_fixed_episodes)
    args.inner_feature_gate = False
    args.inner_gate_reg_coef = 0.0
    args.inner_episode_len = max(1, int(args.inner_episode_len))
    args.inner_train_episodes_per_epoch = max(1, int(args.inner_train_episodes_per_epoch))
    args.inner_start_stride_days = max(1, int(args.inner_start_stride_days))
    args.inner_rollout_update_steps = max(0, int(args.inner_rollout_update_steps))
    args.inner_ppo_epochs = max(1, int(args.inner_ppo_epochs))
    args.inner_episode_batch_size = max(1, int(args.inner_episode_batch_size))
    args.inner_episode_parallel_workers = max(1, int(args.inner_episode_parallel_workers))
    if args.inner_train_fixed_episodes and args.inner_rollout_update_steps > 0:
        args.rollout_update_steps_by_stage["warmup_inner"] = int(args.inner_rollout_update_steps)
    args.controller_algorithm = "pg"
    args.controller_rollout_len = max(1, int(args.controller_rollout_len))
    args.controller_max_segments = max(1, int(args.controller_max_segments))
    args.controller_count_min = max(0, int(args.controller_count_min))
    args.controller_count_max = max(0, int(args.controller_count_max))
    if args.controller_count_max < args.controller_count_min:
        args.controller_count_min, args.controller_count_max = args.controller_count_max, args.controller_count_min
    args.controller_count_penalty_coef = max(0.0, float(args.controller_count_penalty_coef))
    args.controller_max_switches = max(0, int(args.controller_max_switches))
    args.controller_max_switch_penalty_coef = max(0.0, float(args.controller_max_switch_penalty_coef))
    args.controller_tau_min = min(max(float(args.controller_tau_min), 0.0), 0.99)
    args.controller_tau_max = min(max(float(args.controller_tau_max), args.controller_tau_min + 1e-6), 0.999)
    args.controller_policy_temperature = max(1e-6, float(args.controller_policy_temperature))
    args.controller_switch_adv_logit_scale = max(1e-8, float(args.controller_switch_adv_logit_scale))
    args.controller_state_return_scale = max(1e-6, float(args.controller_state_return_scale))
    args.controller_state_drawdown_scale = max(1e-6, float(args.controller_state_drawdown_scale))
    args.controller_pg_batch_windows = max(1, int(args.controller_pg_batch_windows))
    args.controller_windows_per_epoch = max(1, int(args.controller_windows_per_epoch))
    args.controller_train_fixed_episodes = bool(args.controller_train_fixed_episodes)
    args.controller_episode_batch_size = max(1, int(args.controller_episode_batch_size))
    args.controller_episode_parallel_workers = max(1, int(args.controller_episode_parallel_workers))
    args.controller_start_stride_days = max(1, int(args.controller_start_stride_days))
    args.controller_window = max(1, int(args.controller_window))
    args.controller_hidden_dim = max(1, int(args.controller_hidden_dim))
    args.controller_val_interval_epochs = (
        None if args.controller_val_interval_epochs is None
        else max(1, int(args.controller_val_interval_epochs))
    )
    if not args.train_monitor:
        args.warmup_monitor_epochs = 0

    raw_stride = getattr(args, "train_start_stride_days", None)
    args.train_start_stride_auto = raw_stride is None or int(raw_stride) <= 0
    if args.train_start_stride_auto:
        args.train_start_stride_days = default_train_start_stride_days(args.min_hold, args.max_hold)

    args.train_episodes_per_epoch = max(1, int(args.train_episodes_per_epoch))
    args.train_start_stride_days = max(1, int(args.train_start_stride_days))
    args.train_start_stride_formula_days = default_train_start_stride_days(args.min_hold, args.max_hold)
    args.train_start_stride_matches_formula = (
        args.train_start_stride_days == args.train_start_stride_formula_days
    )

    for stage in TRAINING_STAGES:
        epoch_attr = f"{stage}_epochs"
        legacy_episode_attr = f"{stage}_episodes"
        total_episode_attr = f"{stage}_episode_total"

        legacy_episode_total = getattr(args, legacy_episode_attr, None)
        if legacy_episode_total is not None:
            legacy_episode_total = int(legacy_episode_total)
            if legacy_episode_total < 0:
                raise ValueError(f"{legacy_episode_attr} must be non-negative.")
            stage_episodes_per_epoch = (
                args.inner_train_episodes_per_epoch
                if stage == "warmup_inner" and args.inner_train_fixed_episodes
                else args.train_episodes_per_epoch
            )
            if legacy_episode_total % stage_episodes_per_epoch != 0:
                raise ValueError(
                    f"{legacy_episode_attr}={legacy_episode_total} is not a complete number of train epochs. "
                    f"Use a multiple of episodes_per_epoch={stage_episodes_per_epoch}."
                )
            setattr(args, epoch_attr, legacy_episode_total // stage_episodes_per_epoch)
            setattr(args, total_episode_attr, legacy_episode_total)
            continue

        epoch_total = int(getattr(args, epoch_attr))
        if epoch_total < 0:
            raise ValueError(f"{epoch_attr} must be non-negative.")
        setattr(args, epoch_attr, epoch_total)
        stage_episodes_per_epoch = (
            args.inner_train_episodes_per_epoch
            if stage == "warmup_inner" and args.inner_train_fixed_episodes
            else args.train_episodes_per_epoch
        )
        setattr(args, total_episode_attr, epoch_total * stage_episodes_per_epoch)

def stage_episode_total(args, stage):
    return int(getattr(args, f"{stage}_episode_total"))


def stage_epoch_total(args, stage):
    return int(getattr(args, f"{stage}_epochs"))


def total_train_epochs(args):
    return sum(stage_epoch_total(args, stage) for stage in TRAINING_STAGES)


def apply_smoke_overrides(args):
    args.train_episodes_per_epoch = 1
    args.train_start_stride_days = 5
    args.train_episode_count = None
    args.train_episode_start_stride = None
    for stage in TRAINING_STAGES:
        setattr(args, f"{stage}_epochs", 1)
        setattr(args, f"{stage}_episodes", None)
    if not args.train_monitor:
        args.warmup_monitor_epochs = 0
    args.val_interval = 999
    args.train_episode_to_end = False
    args.episode_len = 12
    args.min_hold = 2
    args.max_hold = 4
    args.fixed_cycle = None
    args.outer_rollout_segments = 1
    for stage in TRAINING_STAGES:
        setattr(args, f"{stage}_rollout_segments", 1)
    args.controller_rollout_len = 8
    args.controller_max_segments = 4
    args.controller_count_min = 1
    args.controller_count_max = 4
    args.controller_count_penalty_coef = 0.1
    args.controller_pg_batch_windows = 1
    args.controller_windows_per_epoch = 1
    args.controller_train_fixed_episodes = True
    args.controller_episode_batch_size = 1
    args.controller_episode_parallel_workers = 1
    args.controller_start_stride_days = 4
    args.controller_window = 2
    args.controller_val_interval_epochs = 999
    args.ppo_epochs = 1
    args.inner_ppo_epochs = 1
    args.inner_train_fixed_episodes = True
    args.inner_episode_len = 8
    args.inner_train_episodes_per_epoch = 1
    args.inner_start_stride_days = 4
    args.inner_rollout_update_steps = 4
    args.skip_test = True


def build_child_command(args, market, run_root, seed):
    cmd = [
        args.python,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--market",
        market,
        "--run_root",
        str(run_root),
        "--seed",
        str(seed),
        "--warmup_outer_epochs",
        str(args.warmup_outer_epochs),
        "--warmup_inner_epochs",
        str(args.warmup_inner_epochs),
        "--warmup_monitor_epochs",
        str(args.warmup_monitor_epochs),
        "--joint_epochs",
        str(args.joint_epochs),
        "--joint_single_full_episode" if args.joint_single_full_episode else "--no_joint_single_full_episode",
        "--val_interval",
        str(args.val_interval),
        "--fixed_cycle",
        str(args.fixed_cycle),
        "--min_hold",
        str(args.min_hold),
        "--max_hold",
        str(args.max_hold),
        "--outer_window",
        str(args.outer_window),
        "--outer_rollout_segments",
        str(args.outer_rollout_segments),
        "--warmup_outer_rollout_segments",
        str(args.warmup_outer_rollout_segments),
        "--warmup_inner_rollout_segments",
        str(args.warmup_inner_rollout_segments),
        "--warmup_monitor_rollout_segments",
        str(args.warmup_monitor_rollout_segments),
        "--joint_rollout_segments",
        str(args.joint_rollout_segments),
        "--lr_monitor",
        str(args.lr_monitor),
        "--lr_outer",
        str(args.lr_outer),
        "--lr_inner",
        str(args.lr_inner),
        "--joint_lr_mult",
        str(args.joint_lr_mult),
        "--ppo_epochs",
        str(args.ppo_epochs),
        "--inner_ppo_epochs",
        str(args.inner_ppo_epochs),
        "--inner_rollout_update_steps",
        str(args.inner_rollout_update_steps),
        "--inner_episode_batch_size",
        str(args.inner_episode_batch_size),
        "--inner_episode_parallel_workers",
        str(args.inner_episode_parallel_workers),
        "--inner_batch_size",
        str(args.inner_batch_size),
        "--outer_update_batch_size",
        str(args.outer_update_batch_size),
        "--trade_num",
        str(args.trade_num),
        "--ssm_dim",
        str(args.ssm_dim),
        "--outer_pred_coef",
        str(args.outer_pred_coef),
        "--inner_pred_coef",
        str(args.inner_pred_coef),
        "--inner_pred_target_scale",
        str(args.inner_pred_target_scale),
        "--inner_gate_reg_coef",
        str(args.inner_gate_reg_coef),
        "--inner_norm_mode",
        str(args.inner_norm_mode),
        "--inner_episode_len",
        str(args.inner_episode_len),
        "--inner_train_episodes_per_epoch",
        str(args.inner_train_episodes_per_epoch),
        "--inner_start_stride_days",
        str(args.inner_start_stride_days),
        "--controller_sup_coef",
        str(args.controller_sup_coef),
        "--controller_sup_pretrain_epochs",
        str(args.controller_sup_pretrain_epochs),
        "--controller_sup_pretrain_rollout_len",
        str(args.controller_sup_pretrain_rollout_len),
        "--controller_aux_replay_epochs",
        str(args.controller_aux_replay_epochs),
        "--controller_check_stride_days",
        str(args.controller_check_stride_days),
        "--reward_scale_outer",
        str(args.reward_scale_outer),
        "--reward_scale_inner",
        str(args.reward_scale_inner),
        "--reward_scale_controller",
        str(args.reward_scale_controller),
        "--controller_algorithm",
        str(args.controller_algorithm),
        "--controller_decision_mode",
        str(args.controller_decision_mode),
        "--controller_decision_stride",
        str(args.controller_decision_stride),
        "--controller_eval_decision_stride",
        str(args.controller_eval_decision_stride),
        "--controller_fixed_decision_window",
        str(args.controller_fixed_decision_window),
        "--controller_rollout_len",
        str(args.controller_rollout_len),
        "--controller_max_segments",
        str(args.controller_max_segments),
        "--controller_pg_batch_windows",
        str(args.controller_pg_batch_windows),
        "--controller_pg_logprob_reduction",
        str(args.controller_pg_logprob_reduction),
        "--controller_windows_per_epoch",
        str(args.controller_windows_per_epoch),
        "--controller_fixed_pool_limit",
        str(args.controller_fixed_pool_limit),
        "--controller_episode_batch_size",
        str(args.controller_episode_batch_size),
        "--controller_episode_parallel_workers",
        str(args.controller_episode_parallel_workers),
        "--controller_start_stride_days",
        str(args.controller_start_stride_days),
        "--controller_train_max_hold",
        str(args.controller_train_max_hold),
        "--controller_train_record_max_duration",
        str(args.controller_train_record_max_duration),
        "--controller_eval_max_hold",
        str(args.controller_eval_max_hold),
        "--controller_window",
        str(args.controller_window),
        "--controller_hidden_dim",
        str(args.controller_hidden_dim),
        "--controller_entropy_coef",
        str(args.controller_entropy_coef),
        "--controller_aux_return_coef",
        str(args.controller_aux_return_coef),
        "--controller_aux_mdd_coef",
        str(args.controller_aux_mdd_coef),
        "--controller_aux_switch_adv_coef",
        str(args.controller_aux_switch_adv_coef),
        "--controller_aux_switch_adv_loss_type",
        str(args.controller_aux_switch_adv_loss_type),
        "--controller_switch_adv_logit_coef",
        str(args.controller_switch_adv_logit_coef),
        "--controller_switch_adv_logit_scale",
        str(args.controller_switch_adv_logit_scale),
        "--controller_aux_return_target_scale",
        str(args.controller_aux_return_target_scale),
        "--controller_aux_mdd_target_scale",
        str(args.controller_aux_mdd_target_scale),
        "--controller_aux_switch_adv_target_scale",
        str(args.controller_aux_switch_adv_target_scale),
        "--controller_local_adv_coef",
        str(args.controller_local_adv_coef),
        "--controller_local_adv_scale",
        str(args.controller_local_adv_scale),
        "--controller_local_adv_clip",
        str(args.controller_local_adv_clip),
        "--controller_local_adv_margin",
        str(args.controller_local_adv_margin),
        "--controller_local_adv_loss_type",
        str(args.controller_local_adv_loss_type),
        "--controller_expected_switch_penalty_coef",
        str(args.controller_expected_switch_penalty_coef),
        "--controller_overflow_action_penalty_coef",
        str(args.controller_overflow_action_penalty_coef),
        "--controller_value_coef",
        str(args.controller_value_coef),
        "--controller_mdd_coef",
        str(args.controller_mdd_coef),
        "--controller_return_coef",
        str(args.controller_return_coef),
        "--controller_count_min",
        str(args.controller_count_min),
        "--controller_count_max",
        str(args.controller_count_max),
        "--controller_count_penalty_coef",
        str(args.controller_count_penalty_coef),
        "--controller_max_switches",
        str(args.controller_max_switches),
        "--controller_max_switch_penalty_coef",
        str(args.controller_max_switch_penalty_coef),
        "--controller_switch_coef",
        str(args.controller_switch_coef),
        "--controller_tau_min",
        str(args.controller_tau_min),
        "--controller_tau_max",
        str(args.controller_tau_max),
        "--controller_policy_temperature",
        str(args.controller_policy_temperature),
        "--controller_eval_switch_threshold",
        str(args.controller_eval_switch_threshold),
        "--controller_state_return_scale",
        str(args.controller_state_return_scale),
        "--controller_state_drawdown_scale",
        str(args.controller_state_drawdown_scale),
        "--model_selection_metric",
        str(args.model_selection_metric),
        "--inner_selection_metric",
        str(args.inner_selection_metric),
        "--controller_selection_metric",
        str(args.controller_selection_metric),
        "--rule_switch_threshold",
        str(args.rule_switch_threshold),
        "--train_episodes_per_epoch",
        str(args.train_episodes_per_epoch),
        "--train_start_stride_days",
        str(args.train_start_stride_days),
        "--device",
        str(args.device),
        "--test_max_days",
        str(args.test_max_days),
    ]
    ssm_data_path = args.ssm_data_path
    if ssm_data_path is None:
        if market == "nas":
            ssm_data_path = args.nas_ssm_data_path
        elif market == "sh":
            ssm_data_path = args.sh_ssm_data_path
    if ssm_data_path:
        cmd.extend(["--ssm_data_path", str(ssm_data_path)])
    if args.episode_len is not None:
        cmd.extend(["--episode_len", str(args.episode_len)])
    if args.max_rule_consecutive_low is not None:
        cmd.extend(["--max_rule_consecutive_low", str(args.max_rule_consecutive_low)])
    if args.controller_turnover_coef is not None:
        cmd.extend(["--controller_turnover_coef", str(args.controller_turnover_coef)])
    if args.controller_val_interval_epochs is not None:
        cmd.extend(["--controller_val_interval_epochs", str(args.controller_val_interval_epochs)])
    if args.controller_skip_val:
        cmd.append("--controller_skip_val")
    if args.controller_use_switch_supervision:
        cmd.append("--controller_use_switch_supervision")
    if args.controller_aux_pretrain_offpolicy:
        cmd.append("--controller_aux_pretrain_offpolicy")
    if args.controller_compute_switch_advantage:
        cmd.append("--controller_compute_switch_advantage")
    if args.controller_switch_adv_logit_detach:
        cmd.append("--controller_switch_adv_logit_detach")
    if args.controller_local_adv_balance_classes:
        cmd.append("--controller_local_adv_balance_classes")
    if not args.controller_value_normalize_advantage:
        cmd.append("--no_controller_value_normalize_advantage")
    if args.controller_init_exit_bias is not None:
        cmd.extend(["--controller_init_exit_bias", str(args.controller_init_exit_bias)])
    if args.controller_no_hold_constraints:
        cmd.append("--controller_no_hold_constraints")
    if args.controller_train_fixed_episodes:
        cmd.append("--controller_train_fixed_episodes")
    if args.controller_deterministic_rollout_sampling:
        cmd.append("--controller_deterministic_rollout_sampling")
    if args.frozen_hrl_checkpoint:
        cmd.extend(["--frozen_hrl_checkpoint", str(args.frozen_hrl_checkpoint)])
    if args.controller_first_joint_finetune:
        cmd.append("--controller_first_joint_finetune")
    if args.controller_only_finetune:
        cmd.append("--controller_only_finetune")
    if args.end_to_end_controller_joint:
        cmd.append("--end_to_end_controller_joint")
    if int(getattr(args, "controller_joint_epochs", 0) or 0) > 0:
        cmd.extend(["--controller_joint_epochs", str(args.controller_joint_epochs)])
    if not args.train_episode_to_end:
        cmd.append("--no_train_episode_to_end")
    if args.inner_train_fixed_episodes:
        cmd.append("--inner_train_fixed_episodes")
    else:
        cmd.append("--no_inner_train_fixed_episodes")
    if args.train_monitor:
        cmd.append("--train_monitor")
    else:
        cmd.append("--no_train_controller")
    if args.inner_use_topk:
        cmd.append("--inner_use_topk")
    if args.clear_cuda_cache_on_update:
        cmd.append("--clear_cuda_cache_on_update")
    if args.cpu:
        cmd.append("--cpu")
    if args.skip_test:
        cmd.append("--skip_test")
    if args.test_only_checkpoint:
        cmd.extend(["--test_only_checkpoint", str(args.test_only_checkpoint)])
    if args.test_skip_fixed_scenarios:
        cmd.append("--test_skip_fixed_scenarios")
    if args.controller_test_thresholds:
        cmd.append("--controller_test_thresholds")
        cmd.extend(str(x) for x in args.controller_test_thresholds)
    if args.controller_decision_stride_schedule:
        cmd.append("--controller_decision_stride_schedule")
        cmd.extend(str(x) for x in args.controller_decision_stride_schedule)
    if args.controller_eval_decision_mode:
        cmd.extend(["--controller_eval_decision_mode", str(args.controller_eval_decision_mode)])
    if args.controller_eval_force_max_hold:
        cmd.append("--controller_eval_force_max_hold")
    if args.controller_eval_diagnostics:
        cmd.append("--controller_eval_diagnostics")
    if args.controller_eval_diag_thresholds:
        cmd.append("--controller_eval_diag_thresholds")
        cmd.extend(str(x) for x in args.controller_eval_diag_thresholds)
    return cmd


def set_runtime_training_args(args, market_root, seed):
    runtime_config.seed = int(seed)
    runtime_config.cun_path = str(market_root / "ppo")
    runtime_config.lr_monitor = float(args.lr_monitor)
    runtime_config.lr_outer = float(args.lr_outer)
    runtime_config.lr_inner = float(args.lr_inner)
    runtime_config.joint_lr_mult = float(args.joint_lr_mult)
    runtime_config.warmup_inner_epochs = int(args.warmup_inner_epochs)
    runtime_config.joint_epochs = int(args.joint_epochs)
    runtime_config.end_to_end_controller_joint = bool(args.end_to_end_controller_joint)
    runtime_config.controller_joint_epochs = int(args.controller_joint_epochs)
    runtime_config.joint_single_full_episode = bool(args.joint_single_full_episode)
    if args.ssm_data_path:
        runtime_config.dataset = dict(runtime_config.dataset)
        runtime_config.dataset["ssm_data_path"] = str(args.ssm_data_path)
    runtime_config.ppo_epochs = int(args.ppo_epochs)
    runtime_config.inner_ppo_epochs = int(args.inner_ppo_epochs)
    runtime_config.inner_rollout_update_steps = int(args.inner_rollout_update_steps)
    runtime_config.inner_episode_batch_size = int(args.inner_episode_batch_size)
    runtime_config.inner_episode_parallel_workers = int(args.inner_episode_parallel_workers)
    runtime_config.inner_batch_size = int(args.inner_batch_size)
    runtime_config.outer_update_batch_size = int(args.outer_update_batch_size)
    runtime_config.trade_num = int(args.trade_num)
    runtime_config.ssm_dim = int(args.ssm_dim)
    runtime_config.outer_pred_coef = float(args.outer_pred_coef)
    runtime_config.inner_pred_coef = float(args.inner_pred_coef)
    runtime_config.inner_pred_target_scale = float(args.inner_pred_target_scale)
    runtime_config.inner_gate_reg_coef = 0.0
    runtime_config.inner_use_topk = bool(args.inner_use_topk)
    runtime_config.inner_feature_gate = False
    runtime_config.inner_norm_mode = str(args.inner_norm_mode)
    runtime_config.inner_train_fixed_episodes = bool(args.inner_train_fixed_episodes)
    runtime_config.inner_episode_len = int(args.inner_episode_len)
    runtime_config.inner_train_episodes_per_epoch = int(args.inner_train_episodes_per_epoch)
    runtime_config.inner_train_start_stride_days = int(args.inner_start_stride_days)
    runtime_config.controller_sup_coef = float(args.controller_sup_coef)
    runtime_config.controller_use_switch_supervision = bool(args.controller_use_switch_supervision)
    runtime_config.controller_sup_pretrain_epochs = max(0, int(args.controller_sup_pretrain_epochs))
    runtime_config.controller_sup_pretrain_rollout_len = max(0, int(args.controller_sup_pretrain_rollout_len))
    runtime_config.controller_aux_pretrain_offpolicy = bool(args.controller_aux_pretrain_offpolicy)
    runtime_config.controller_check_stride_days = max(1, int(args.controller_check_stride_days))
    runtime_config.controller_no_hold_constraints = bool(args.controller_no_hold_constraints)
    runtime_config.clear_cuda_cache_on_update = bool(args.clear_cuda_cache_on_update)
    runtime_config.reward_scale_outer = float(args.reward_scale_outer)
    runtime_config.reward_scale_inner = float(args.reward_scale_inner)
    runtime_config.reward_scale_monitor = float(args.reward_scale_controller)
    runtime_config.reward_scale_controller = float(args.reward_scale_controller)
    runtime_config.controller_algorithm = str(args.controller_algorithm)
    runtime_config.controller_decision_mode = str(args.controller_decision_mode)
    runtime_config.controller_eval_decision_mode = (
        str(args.controller_eval_decision_mode)
        if args.controller_eval_decision_mode
        else None
    )
    runtime_config.controller_decision_stride = max(1, int(args.controller_decision_stride))
    runtime_config.controller_eval_decision_stride = max(0, int(args.controller_eval_decision_stride))
    runtime_config.controller_decision_stride_schedule = (
        [max(1, int(x)) for x in args.controller_decision_stride_schedule]
        if args.controller_decision_stride_schedule
        else None
    )
    runtime_config.controller_fixed_decision_window = max(0, int(args.controller_fixed_decision_window))
    runtime_config.controller_rollout_len = int(args.controller_rollout_len)
    runtime_config.controller_max_segments = int(args.controller_max_segments)
    runtime_config.controller_pg_batch_windows = int(args.controller_pg_batch_windows)
    runtime_config.controller_pg_logprob_reduction = str(args.controller_pg_logprob_reduction)
    runtime_config.controller_windows_per_epoch = int(args.controller_windows_per_epoch)
    runtime_config.controller_fixed_pool_limit = max(0, int(args.controller_fixed_pool_limit))
    runtime_config.controller_train_fixed_episodes = bool(args.controller_train_fixed_episodes)
    runtime_config.controller_episode_batch_size = int(args.controller_episode_batch_size)
    runtime_config.controller_episode_parallel_workers = int(args.controller_episode_parallel_workers)
    runtime_config.controller_deterministic_rollout_sampling = bool(args.controller_deterministic_rollout_sampling)
    runtime_config.controller_start_stride_days = int(args.controller_start_stride_days)
    runtime_config.controller_train_max_hold = max(-1, int(args.controller_train_max_hold))
    runtime_config.controller_train_record_max_duration = max(0, int(args.controller_train_record_max_duration))
    runtime_config.controller_eval_max_hold = max(-1, int(args.controller_eval_max_hold))
    runtime_config.controller_window = int(args.controller_window)
    runtime_config.controller_hidden_dim = int(args.controller_hidden_dim)
    runtime_config.controller_init_exit_bias = (
        None if args.controller_init_exit_bias is None else float(args.controller_init_exit_bias)
    )
    runtime_config.controller_entropy_coef = float(args.controller_entropy_coef)
    runtime_config.controller_aux_return_coef = float(args.controller_aux_return_coef)
    runtime_config.controller_aux_mdd_coef = float(args.controller_aux_mdd_coef)
    runtime_config.controller_aux_switch_adv_coef = float(args.controller_aux_switch_adv_coef)
    runtime_config.controller_aux_switch_adv_loss_type = str(args.controller_aux_switch_adv_loss_type)
    runtime_config.controller_switch_adv_logit_coef = float(args.controller_switch_adv_logit_coef)
    runtime_config.controller_switch_adv_logit_scale = max(1e-8, float(args.controller_switch_adv_logit_scale))
    runtime_config.controller_switch_adv_logit_detach = bool(args.controller_switch_adv_logit_detach)
    runtime_config.controller_aux_return_target_scale = float(args.controller_aux_return_target_scale)
    runtime_config.controller_aux_mdd_target_scale = float(args.controller_aux_mdd_target_scale)
    runtime_config.controller_aux_switch_adv_target_scale = float(args.controller_aux_switch_adv_target_scale)
    runtime_config.controller_aux_replay_epochs = max(1, int(args.controller_aux_replay_epochs))
    runtime_config.controller_local_adv_coef = float(args.controller_local_adv_coef)
    runtime_config.controller_local_adv_scale = max(1e-8, float(args.controller_local_adv_scale))
    runtime_config.controller_local_adv_clip = max(0.0, float(args.controller_local_adv_clip))
    runtime_config.controller_local_adv_margin = max(0.0, float(args.controller_local_adv_margin))
    runtime_config.controller_local_adv_loss_type = str(args.controller_local_adv_loss_type)
    runtime_config.controller_local_adv_balance_classes = bool(args.controller_local_adv_balance_classes)
    runtime_config.controller_expected_switch_penalty_coef = max(0.0, float(args.controller_expected_switch_penalty_coef))
    runtime_config.controller_overflow_action_penalty_coef = max(0.0, float(args.controller_overflow_action_penalty_coef))
    runtime_config.controller_value_coef = max(0.0, float(args.controller_value_coef))
    runtime_config.controller_value_normalize_advantage = bool(args.controller_value_normalize_advantage)
    runtime_config.controller_compute_switch_advantage = (
        bool(args.controller_compute_switch_advantage)
        or runtime_config.controller_local_adv_coef > 0.0
        or runtime_config.controller_aux_switch_adv_coef > 0.0
    )
    runtime_config.controller_mdd_coef = float(args.controller_mdd_coef)
    runtime_config.controller_return_coef = float(args.controller_return_coef)
    runtime_config.controller_count_min = int(args.controller_count_min)
    runtime_config.controller_count_max = int(args.controller_count_max)
    runtime_config.controller_count_penalty_coef = float(args.controller_count_penalty_coef)
    runtime_config.controller_max_switches = int(args.controller_max_switches)
    runtime_config.controller_max_switch_penalty_coef = float(args.controller_max_switch_penalty_coef)
    runtime_config.controller_switch_coef = float(args.controller_switch_coef)
    runtime_config.controller_tau_min = float(args.controller_tau_min)
    runtime_config.controller_tau_max = float(args.controller_tau_max)
    runtime_config.controller_policy_temperature = float(args.controller_policy_temperature)
    runtime_config.controller_eval_switch_threshold = min(1.0, max(0.0, float(args.controller_eval_switch_threshold)))
    runtime_config.controller_eval_force_max_hold = bool(args.controller_eval_force_max_hold)
    runtime_config.controller_eval_diagnostics = bool(args.controller_eval_diagnostics)
    runtime_config.controller_eval_diag_thresholds = (
        [min(1.0, max(0.0, float(x))) for x in args.controller_eval_diag_thresholds]
        if args.controller_eval_diag_thresholds
        else None
    )
    runtime_config.controller_test_thresholds = (
        [min(1.0, max(0.0, float(x))) for x in args.controller_test_thresholds]
        if args.controller_test_thresholds
        else None
    )
    runtime_config.test_max_days = max(0, int(args.test_max_days))
    runtime_config.test_skip_fixed_scenarios = bool(args.test_skip_fixed_scenarios)
    runtime_config.controller_state_return_scale = float(args.controller_state_return_scale)
    runtime_config.controller_state_drawdown_scale = float(args.controller_state_drawdown_scale)
    runtime_config.model_selection_metric = str(args.model_selection_metric)
    runtime_config.inner_selection_metric = str(args.inner_selection_metric)
    runtime_config.controller_selection_metric = str(args.controller_selection_metric)
    if args.controller_turnover_coef is not None:
        runtime_config.controller_turnover_coef = float(args.controller_turnover_coef)
    elif hasattr(runtime_config, "controller_turnover_coef"):
        delattr(runtime_config, "controller_turnover_coef")
    if args.controller_val_interval_epochs is not None:
        runtime_config.controller_val_interval_epochs = int(args.controller_val_interval_epochs)
    elif hasattr(runtime_config, "controller_val_interval_epochs"):
        delattr(runtime_config, "controller_val_interval_epochs")
    runtime_config.controller_skip_val = bool(args.controller_skip_val)
    runtime_config.controller_sup_horizon = int(args.min_hold)
    runtime_config.train_monitor_enabled = bool(args.train_monitor)
    runtime_config.use_rule_switch_train = False
    runtime_config.rule_switch_threshold = float(args.rule_switch_threshold)
    if args.max_rule_consecutive_low is not None:
        runtime_config.max_rule_consecutive_low = int(args.max_rule_consecutive_low)
    runtime_config.val_interval = int(args.val_interval)
    runtime_config.train_episodes_per_epoch = int(args.train_episodes_per_epoch)
    runtime_config.train_start_stride_days = int(args.train_start_stride_days)
    runtime_config.train_episode_count = runtime_config.train_episodes_per_epoch
    runtime_config.train_episode_start_stride = runtime_config.train_start_stride_days
    runtime_config.train_episode_to_end = bool(args.train_episode_to_end)
    if args.episode_len is not None:
        runtime_config.episode_len = int(args.episode_len)
    runtime_config.outer_window = int(args.outer_window)
    runtime_config.min_hold = int(args.min_hold)
    runtime_config.max_hold = int(args.max_hold)
    runtime_config.fixed_cycle = int(args.max_hold)
    runtime_config.outer_rollout_segments = int(args.outer_rollout_segments)
    for stage in TRAINING_STAGES:
        setattr(runtime_config, f"{stage}_rollout_segments", int(getattr(args, f"{stage}_rollout_segments")))
    runtime_config.rollout_segments_by_stage = dict(args.rollout_segments_by_stage)
    runtime_config.rollout_update_steps_by_stage = dict(args.rollout_update_steps_by_stage)
    if runtime_config.inner_train_fixed_episodes and runtime_config.inner_rollout_update_steps > 0:
        runtime_config.rollout_update_steps_by_stage["warmup_inner"] = runtime_config.inner_rollout_update_steps
    runtime_config.rollout_update_steps = int(args.rollout_update_steps_by_stage.get("warmup_outer", 0))

    device_name = "cpu" if args.cpu else args.device
    runtime_config.device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")


def write_child_metadata(args, market_root, label, seed, fixed_cycle):
    metadata = {
        "market": args.market,
        "label": label,
        "seed": int(seed),
        "split": {
            "train_start": runtime_config.train_start_date,
            "train_end": runtime_config.train_end_date,
            "valid_start": runtime_config.valid_start_date,
            "valid_end": runtime_config.valid_end_date,
            "test_start": runtime_config.test_start_date,
            "test_end": runtime_config.test_end_date,
        },
        "episode": {
            "train_episodes_per_epoch": runtime_config.train_episodes_per_epoch,
            "train_start_stride_days": runtime_config.train_start_stride_days,
            "train_start_stride_formula_days": getattr(args, "train_start_stride_formula_days", None),
            "train_start_stride_matches_formula": getattr(args, "train_start_stride_matches_formula", None),
            "train_episode_to_end": runtime_config.train_episode_to_end,
            "episode_len": getattr(runtime_config, "episode_len", None),
            "inner_train_fixed_episodes": getattr(runtime_config, "inner_train_fixed_episodes", None),
            "inner_episode_len": getattr(runtime_config, "inner_episode_len", None),
            "inner_train_episodes_per_epoch": getattr(runtime_config, "inner_train_episodes_per_epoch", None),
            "inner_train_start_stride_days": getattr(runtime_config, "inner_train_start_stride_days", None),
            "outer_window": getattr(runtime_config, "outer_window", None),
            "min_hold": getattr(runtime_config, "min_hold", None),
            "max_hold": getattr(runtime_config, "max_hold", None),
        },
        "training": {
            "total_train_epochs": total_train_epochs(args),
            "warmup_outer_epochs": args.warmup_outer_epochs,
            "warmup_inner_epochs": args.warmup_inner_epochs,
            "warmup_monitor_epochs": args.warmup_monitor_epochs,
            "joint_epochs": args.joint_epochs,
            "joint_single_full_episode": getattr(runtime_config, "joint_single_full_episode", None),
            "warmup_outer_episode_total": stage_episode_total(args, "warmup_outer"),
            "warmup_inner_episode_total": stage_episode_total(args, "warmup_inner"),
            "warmup_monitor_episode_total": stage_episode_total(args, "warmup_monitor"),
            "joint_episode_total": stage_episode_total(args, "joint"),
            "joint_effective_episode_total": args.joint_epochs if args.joint_single_full_episode else stage_episode_total(args, "joint"),
            "val_interval": args.val_interval,
            "fixed_cycle": fixed_cycle,
            "min_hold": getattr(runtime_config, "min_hold", None),
            "max_hold": getattr(runtime_config, "max_hold", None),
            "outer_rollout_segments": getattr(runtime_config, "outer_rollout_segments", None),
            "rollout_update_steps": getattr(runtime_config, "rollout_update_steps", None),
            "rollout_segments_by_stage": getattr(runtime_config, "rollout_segments_by_stage", None),
            "rollout_update_steps_by_stage": getattr(runtime_config, "rollout_update_steps_by_stage", None),
            "ppo_epochs": args.ppo_epochs,
            "inner_ppo_epochs": getattr(runtime_config, "inner_ppo_epochs", None),
            "inner_rollout_update_steps": getattr(runtime_config, "inner_rollout_update_steps", None),
            "inner_episode_batch_size": getattr(runtime_config, "inner_episode_batch_size", None),
            "inner_episode_parallel_workers": getattr(runtime_config, "inner_episode_parallel_workers", None),
            "joint_lr_mult": args.joint_lr_mult,
            "frozen_hrl_checkpoint": args.frozen_hrl_checkpoint,
            "controller_first_joint_finetune": args.controller_first_joint_finetune,
            "controller_only_finetune": args.controller_only_finetune,
            "end_to_end_controller_joint": args.end_to_end_controller_joint,
            "controller_joint_epochs": args.controller_joint_epochs,
            "inner_batch_size": args.inner_batch_size,
            "outer_update_batch_size": args.outer_update_batch_size,
            "trade_num": args.trade_num,
            "ssm_dim": args.ssm_dim,
            "outer_pred_coef": args.outer_pred_coef,
            "inner_pred_coef": args.inner_pred_coef,
            "inner_pred_target_scale": args.inner_pred_target_scale,
            "inner_gate_reg_coef": 0.0,
            "inner_use_topk": args.inner_use_topk,
            "inner_feature_gate": False,
            "inner_norm_mode": args.inner_norm_mode,
            "controller_sup_coef": args.controller_sup_coef,
            "controller_use_switch_supervision": getattr(runtime_config, "controller_use_switch_supervision", None),
            "controller_sup_pretrain_epochs": getattr(runtime_config, "controller_sup_pretrain_epochs", None),
            "controller_sup_pretrain_rollout_len": getattr(runtime_config, "controller_sup_pretrain_rollout_len", None),
            "controller_aux_pretrain_offpolicy": getattr(runtime_config, "controller_aux_pretrain_offpolicy", None),
            "controller_check_stride_days": getattr(runtime_config, "controller_check_stride_days", None),
            "controller_no_hold_constraints": getattr(runtime_config, "controller_no_hold_constraints", None),
            "controller_algorithm": getattr(runtime_config, "controller_algorithm", None),
            "controller_decision_mode": getattr(runtime_config, "controller_decision_mode", None),
            "controller_eval_decision_mode": getattr(runtime_config, "controller_eval_decision_mode", None),
            "controller_decision_stride": getattr(runtime_config, "controller_decision_stride", None),
            "controller_fixed_pool_limit": getattr(runtime_config, "controller_fixed_pool_limit", None),
            "controller_eval_decision_stride": getattr(runtime_config, "controller_eval_decision_stride", None),
            "controller_decision_stride_schedule": getattr(runtime_config, "controller_decision_stride_schedule", None),
            "controller_fixed_decision_window": getattr(runtime_config, "controller_fixed_decision_window", None),
            "controller_rollout_len": getattr(runtime_config, "controller_rollout_len", None),
            "controller_max_segments": getattr(runtime_config, "controller_max_segments", None),
            "controller_pg_batch_windows": getattr(runtime_config, "controller_pg_batch_windows", None),
            "controller_pg_logprob_reduction": getattr(runtime_config, "controller_pg_logprob_reduction", None),
            "controller_windows_per_epoch": getattr(runtime_config, "controller_windows_per_epoch", None),
            "controller_train_fixed_episodes": getattr(runtime_config, "controller_train_fixed_episodes", None),
            "controller_episode_batch_size": getattr(runtime_config, "controller_episode_batch_size", None),
            "controller_episode_parallel_workers": getattr(runtime_config, "controller_episode_parallel_workers", None),
            "controller_start_stride_days": getattr(runtime_config, "controller_start_stride_days", None),
            "controller_train_max_hold": getattr(runtime_config, "controller_train_max_hold", None),
            "controller_train_record_max_duration": getattr(runtime_config, "controller_train_record_max_duration", None),
            "controller_eval_max_hold": getattr(runtime_config, "controller_eval_max_hold", None),
            "controller_window": getattr(runtime_config, "controller_window", None),
            "controller_hidden_dim": getattr(runtime_config, "controller_hidden_dim", None),
            "controller_init_exit_bias": getattr(runtime_config, "controller_init_exit_bias", None),
            "controller_entropy_coef": getattr(runtime_config, "controller_entropy_coef", None),
            "controller_aux_return_coef": getattr(runtime_config, "controller_aux_return_coef", None),
            "controller_aux_mdd_coef": getattr(runtime_config, "controller_aux_mdd_coef", None),
            "controller_aux_switch_adv_coef": getattr(runtime_config, "controller_aux_switch_adv_coef", None),
            "controller_aux_switch_adv_loss_type": getattr(runtime_config, "controller_aux_switch_adv_loss_type", None),
            "controller_switch_adv_logit_coef": getattr(runtime_config, "controller_switch_adv_logit_coef", None),
            "controller_switch_adv_logit_scale": getattr(runtime_config, "controller_switch_adv_logit_scale", None),
            "controller_switch_adv_logit_detach": getattr(runtime_config, "controller_switch_adv_logit_detach", None),
            "controller_compute_switch_advantage": getattr(runtime_config, "controller_compute_switch_advantage", None),
            "controller_aux_return_target_scale": getattr(runtime_config, "controller_aux_return_target_scale", None),
            "controller_aux_mdd_target_scale": getattr(runtime_config, "controller_aux_mdd_target_scale", None),
            "controller_aux_switch_adv_target_scale": getattr(runtime_config, "controller_aux_switch_adv_target_scale", None),
            "controller_aux_replay_epochs": getattr(runtime_config, "controller_aux_replay_epochs", None),
            "controller_local_adv_coef": getattr(runtime_config, "controller_local_adv_coef", None),
            "controller_local_adv_scale": getattr(runtime_config, "controller_local_adv_scale", None),
            "controller_local_adv_clip": getattr(runtime_config, "controller_local_adv_clip", None),
            "controller_local_adv_margin": getattr(runtime_config, "controller_local_adv_margin", None),
            "controller_local_adv_loss_type": getattr(runtime_config, "controller_local_adv_loss_type", None),
            "controller_local_adv_balance_classes": getattr(runtime_config, "controller_local_adv_balance_classes", None),
            "controller_expected_switch_penalty_coef": getattr(runtime_config, "controller_expected_switch_penalty_coef", None),
            "controller_overflow_action_penalty_coef": getattr(runtime_config, "controller_overflow_action_penalty_coef", None),
            "controller_value_coef": getattr(runtime_config, "controller_value_coef", None),
            "controller_value_normalize_advantage": getattr(runtime_config, "controller_value_normalize_advantage", None),
            "controller_mdd_coef": getattr(runtime_config, "controller_mdd_coef", None),
            "controller_return_coef": getattr(runtime_config, "controller_return_coef", None),
            "controller_count_min": getattr(runtime_config, "controller_count_min", None),
            "controller_count_max": getattr(runtime_config, "controller_count_max", None),
            "controller_count_penalty_coef": getattr(runtime_config, "controller_count_penalty_coef", None),
            "controller_max_switches": getattr(runtime_config, "controller_max_switches", None),
            "controller_max_switch_penalty_coef": getattr(runtime_config, "controller_max_switch_penalty_coef", None),
            "controller_switch_coef": getattr(runtime_config, "controller_switch_coef", None),
            "controller_turnover_coef": getattr(runtime_config, "controller_turnover_coef", None),
            "controller_val_interval_epochs": getattr(runtime_config, "controller_val_interval_epochs", None),
            "controller_skip_val": getattr(runtime_config, "controller_skip_val", None),
            "controller_tau_min": getattr(runtime_config, "controller_tau_min", None),
            "controller_tau_max": getattr(runtime_config, "controller_tau_max", None),
            "controller_policy_temperature": getattr(runtime_config, "controller_policy_temperature", None),
            "controller_eval_switch_threshold": getattr(runtime_config, "controller_eval_switch_threshold", None),
            "controller_eval_force_max_hold": getattr(runtime_config, "controller_eval_force_max_hold", None),
            "controller_eval_diagnostics": getattr(runtime_config, "controller_eval_diagnostics", None),
            "controller_eval_diag_thresholds": getattr(runtime_config, "controller_eval_diag_thresholds", None),
            "controller_test_thresholds": getattr(runtime_config, "controller_test_thresholds", None),
            "test_max_days": getattr(runtime_config, "test_max_days", None),
            "controller_state_return_scale": getattr(runtime_config, "controller_state_return_scale", None),
            "controller_state_drawdown_scale": getattr(runtime_config, "controller_state_drawdown_scale", None),
            "model_selection_metric": getattr(runtime_config, "model_selection_metric", None),
            "inner_selection_metric": getattr(runtime_config, "inner_selection_metric", None),
            "controller_selection_metric": getattr(runtime_config, "controller_selection_metric", None),
            "clear_cuda_cache_on_update": args.clear_cuda_cache_on_update,
            "reward_scale_outer": args.reward_scale_outer,
            "reward_scale_inner": args.reward_scale_inner,
            "reward_scale_controller": args.reward_scale_controller,
            "train_monitor_enabled": runtime_config.train_monitor_enabled,
            "use_rule_switch_train": runtime_config.use_rule_switch_train,
            "rule_switch_threshold": getattr(runtime_config, "rule_switch_threshold", None),
            "max_rule_consecutive_low": getattr(runtime_config, "max_rule_consecutive_low", None),
            "device": str(runtime_config.device),
        },
        "paths": {
            "ssm_data_path": runtime_config.dataset["ssm_data_path"],
            "stocks_path": runtime_config.dataset["stocks_path"],
            "run_dir": str(market_root / "ppo" / f"seed_{seed}"),
        },
    }
    market_root.mkdir(parents=True, exist_ok=True)
    with (market_root / f"seed_{seed}_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def run_child(args):
    normalize_training_schedule(args)
    label, config_module = MARKET_CONFIGS[args.market]
    apply_market_config(config_module)

    market_root = Path(args.run_root).resolve() / args.market
    set_runtime_training_args(args, market_root, args.seed)
    fixed_cycle = int(args.max_hold)
    runtime_config.min_hold = int(args.min_hold)
    runtime_config.max_hold = fixed_cycle
    runtime_config.fixed_cycle = fixed_cycle
    runtime_config.rollout_segments_by_stage = {
        stage: int(getattr(args, f"{stage}_rollout_segments"))
        for stage in TRAINING_STAGES
    }
    runtime_config.rollout_update_steps_by_stage = {
        stage: fixed_cycle * segments
        for stage, segments in runtime_config.rollout_segments_by_stage.items()
    }
    if getattr(runtime_config, "inner_train_fixed_episodes", False) and getattr(runtime_config, "inner_rollout_update_steps", 0) > 0:
        runtime_config.rollout_update_steps_by_stage["warmup_inner"] = int(runtime_config.inner_rollout_update_steps)
    runtime_config.rollout_update_steps = runtime_config.rollout_update_steps_by_stage["warmup_outer"]

    logger = create_logger(str(market_root / "ppo" / "logs" / f"seed_{args.seed}"))
    logger.info("===== HRL training: %s (%s), seed=%s =====", label, args.market, args.seed)
    logger.info(
        "Split: train [%s, %s], val [%s, %s], test [%s, %s]",
        runtime_config.train_start_date,
        runtime_config.train_end_date,
        runtime_config.valid_start_date,
        runtime_config.valid_end_date,
        runtime_config.test_start_date,
        runtime_config.test_end_date,
    )
    logger.info("SSM data path: %s", runtime_config.dataset["ssm_data_path"])
    logger.info(
        "Train epoch: episodes_per_epoch=%s start_stride_days=%s formula_stride=%s matches_formula=%s to_end=%s",
        runtime_config.train_episodes_per_epoch,
        runtime_config.train_start_stride_days,
        getattr(args, "train_start_stride_formula_days", None),
        getattr(args, "train_start_stride_matches_formula", None),
        runtime_config.train_episode_to_end,
    )
    joint_effective_episode_total = args.joint_epochs if args.joint_single_full_episode else stage_episode_total(args, "joint")
    logger.info(
        "Stages: outer=%s epochs/%s episodes, inner=%s epochs/%s episodes, "
        "controller=%s epochs/%s episodes, joint=%s epochs/%s episodes, total=%s epochs, hold=[%s,%s]",
        args.warmup_outer_epochs,
        stage_episode_total(args, "warmup_outer"),
        args.warmup_inner_epochs,
        stage_episode_total(args, "warmup_inner"),
        args.warmup_monitor_epochs,
        stage_episode_total(args, "warmup_monitor"),
        args.joint_epochs,
        joint_effective_episode_total,
        total_train_epochs(args),
        runtime_config.min_hold,
        fixed_cycle,
    )
    logger.info("Joint schedule: single_full_episode=%s", runtime_config.joint_single_full_episode)
    logger.info(
        "Inner warmup schedule: fixed=%s episodes_per_epoch=%s episode_len=%s start_stride_days=%s "
        "update_steps=%s episode_batch_size=%s parallel_workers=%s ppo_epochs=%s",
        getattr(runtime_config, "inner_train_fixed_episodes", False),
        getattr(runtime_config, "inner_train_episodes_per_epoch", None),
        getattr(runtime_config, "inner_episode_len", None),
        getattr(runtime_config, "inner_train_start_stride_days", None),
        getattr(runtime_config, "inner_rollout_update_steps", None),
        getattr(runtime_config, "inner_episode_batch_size", None),
        getattr(runtime_config, "inner_episode_parallel_workers", None),
        getattr(runtime_config, "inner_ppo_epochs", None),
    )
    logger.info(
        "Efficiency config: outer_window=%s min_hold=%s max_hold=%s rollout_segments_by_stage=%s rollout_update_steps_by_stage=%s",
        runtime_config.outer_window,
        runtime_config.min_hold,
        fixed_cycle,
        runtime_config.rollout_segments_by_stage,
        runtime_config.rollout_update_steps_by_stage,
    )
    logger.info(
        "Return-prediction auxiliary loss: outer_pred_coef=%s inner_pred_coef=%s inner_target_scale=%s",
        runtime_config.outer_pred_coef,
        runtime_config.inner_pred_coef,
        runtime_config.inner_pred_target_scale,
    )
    logger.info(
        "Checkpoint selection: fixed_hrl=%s inner=%s controller=%s",
        getattr(runtime_config, "model_selection_metric", "sharpe"),
        getattr(runtime_config, "inner_selection_metric", "return"),
        getattr(runtime_config, "controller_selection_metric", "risk_return"),
    )
    logger.info(
        "Controller policy: %s, switch_sup=%s, sup_coef=%s, check_stride_days=%s",
        "counterfactual PG controller" if runtime_config.train_monitor_enabled else "forced hold/switch constraints only",
        getattr(runtime_config, "controller_use_switch_supervision", False),
        getattr(runtime_config, "controller_sup_coef", None),
        getattr(runtime_config, "controller_check_stride_days", None),
    )
    logger.info(
        "Controller auxiliary heads: remaining_hold_return_coef=%s remaining_hold_mdd_coef=%s "
        "target_scale=(return:%s, mdd:%s)",
        getattr(runtime_config, "controller_aux_return_coef", None),
        getattr(runtime_config, "controller_aux_mdd_coef", None),
        getattr(runtime_config, "controller_aux_return_target_scale", None),
        getattr(runtime_config, "controller_aux_mdd_target_scale", None),
    )
    logger.info(
        "Controller PG config: rollout_len=%s max_switches=%s batch_windows=%s windows_per_epoch=%s "
        "reward=(ret:%s, overflow_penalty:%s) tau=[%s,%s] temp=%s entropy=%s",
        getattr(runtime_config, "controller_rollout_len", None),
        getattr(runtime_config, "controller_max_switches", None) or "rollout_len//min_hold",
        getattr(runtime_config, "controller_pg_batch_windows", None),
        getattr(runtime_config, "controller_windows_per_epoch", None),
        getattr(runtime_config, "controller_return_coef", None),
        getattr(runtime_config, "controller_max_switch_penalty_coef", None),
        getattr(runtime_config, "controller_tau_min", None),
        getattr(runtime_config, "controller_tau_max", None),
        getattr(runtime_config, "controller_policy_temperature", None),
        getattr(runtime_config, "controller_entropy_coef", None),
    )
    write_child_metadata(args, market_root, label, args.seed, fixed_cycle)

    set_seed(int(args.seed), logger)
    env = PPO_Env(
        logger=logger,
        episode_len=args.episode_len,
        outer_window=args.outer_window,
        max_hold=fixed_cycle,
        min_hold=args.min_hold,
        train_episodes_per_epoch=args.train_episodes_per_epoch,
        train_start_stride_days=args.train_start_stride_days,
        train_episode_to_end=args.train_episode_to_end,
    )
    networks = HRL_Networks(args.ssm_dim, env.num_stocks, runtime_config).to(runtime_config.device)
    agent = HRL_PPO_Agent(networks, runtime_config)
    buffer = HRL_Buffer(
        capacity=3000,
        device=runtime_config.device,
        outer_reward_scale=getattr(runtime_config, "reward_scale_outer", 1.0),
    )
    trainer = HRL_Trainer(agent, env, buffer, runtime_config, logger)
    env.set_mode("train")
    if args.train_episode_to_end and len(env.train_indices_pool) != args.train_episodes_per_epoch:
        raise RuntimeError(
            "Fixed train episode pool does not match train_episodes_per_epoch: "
            f"{len(env.train_indices_pool)} != {args.train_episodes_per_epoch}"
        )

    if args.test_only_checkpoint:
        checkpoint_path = Path(args.test_only_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Test checkpoint not found: {checkpoint_path}")
        trainer.test(str(checkpoint_path))
        result = {"best_ckpt": str(checkpoint_path), "test_only": True}
        with (market_root / f"seed_{args.seed}_train_result.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        logger.info("HRL test-only completed: %s", result)
        return

    if args.controller_first_joint_finetune and args.controller_only_finetune:
        raise ValueError("--controller_first_joint_finetune and --controller_only_finetune are mutually exclusive.")
    if args.end_to_end_controller_joint and (
            args.controller_first_joint_finetune
            or args.controller_only_finetune
            or args.frozen_hrl_checkpoint
    ):
        raise ValueError(
            "--end_to_end_controller_joint starts from scratch and cannot be combined with "
            "--frozen_hrl_checkpoint, --controller_first_joint_finetune, or --controller_only_finetune."
        )

    if args.controller_first_joint_finetune:
        if not args.frozen_hrl_checkpoint:
            raise ValueError("--controller_first_joint_finetune requires --frozen_hrl_checkpoint.")
        checkpoint_path = Path(args.frozen_hrl_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Frozen HRL checkpoint not found: {checkpoint_path}")
        trainer.load_frozen_hrl_checkpoint(str(checkpoint_path))
        result = train_controller_then_joint_finetune(
            trainer,
            controller_episodes=stage_episode_total(args, "warmup_monitor"),
            joint_episodes=stage_episode_total(args, "joint"),
            fixed_cycle=fixed_cycle,
            val_interval=args.val_interval,
            train_monitor=runtime_config.train_monitor_enabled,
            save_prefix=(
                f"ctrl_first_{args.market}_seed{args.seed}_"
                f"op{_coef_tag(args.outer_pred_coef)}_ip{_coef_tag(args.inner_pred_coef)}"
            ),
        )
    elif args.controller_only_finetune:
        if not args.frozen_hrl_checkpoint:
            raise ValueError("--controller_only_finetune requires --frozen_hrl_checkpoint.")
        checkpoint_path = Path(args.frozen_hrl_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Frozen HRL checkpoint not found: {checkpoint_path}")
        trainer.load_frozen_hrl_checkpoint(str(checkpoint_path))
        result = train_controller_only_from_frozen(
            trainer,
            controller_episodes=stage_episode_total(args, "warmup_monitor"),
            fixed_cycle=fixed_cycle,
            val_interval=args.val_interval,
            train_monitor=runtime_config.train_monitor_enabled,
            save_prefix=(
                f"ctrl_only_{args.market}_seed{args.seed}_"
                f"op{_coef_tag(args.outer_pred_coef)}_ip{_coef_tag(args.inner_pred_coef)}"
            ),
        )
    else:
        if args.frozen_hrl_checkpoint:
            checkpoint_path = Path(args.frozen_hrl_checkpoint).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Frozen HRL checkpoint not found: {checkpoint_path}")
            trainer.load_frozen_hrl_checkpoint(str(checkpoint_path))
        result = train_warmup_then_joint_with_monitor(
            trainer,
            warmup_outer_episodes=stage_episode_total(args, "warmup_outer"),
            warmup_inner_episodes=stage_episode_total(args, "warmup_inner"),
            warmup_monitor_episodes=stage_episode_total(args, "warmup_monitor"),
            joint_episodes=stage_episode_total(args, "joint"),
            fixed_cycle=fixed_cycle,
            val_interval=args.val_interval,
            train_monitor=runtime_config.train_monitor_enabled,
            use_rule_switch_train=runtime_config.use_rule_switch_train,
            save_prefix=(
                f"hrl_{args.market}_seed{args.seed}_"
                f"op{_coef_tag(args.outer_pred_coef)}_ip{_coef_tag(args.inner_pred_coef)}"
            ),
        )
    with (market_root / f"seed_{args.seed}_train_result.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    if not args.skip_test:
        trainer.test(result["best_ckpt"])

    logger.info("HRL training completed: %s", result)


def run_parent(args):
    if args.smoke:
        apply_smoke_overrides(args)
    normalize_training_schedule(args)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (ROOT / args.output_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"HRL training run root: {run_root}", flush=True)

    failures = []
    coef_grid = supervised_coef_grid(args)
    multi_coef = len(coef_grid) > 1
    for outer_coef, inner_coef, coef_tag in coef_grid:
        combo_args = argparse.Namespace(**vars(args))
        combo_args.outer_pred_coef = outer_coef
        combo_args.inner_pred_coef = inner_coef
        combo_root = run_root / coef_tag if multi_coef else run_root
        if multi_coef:
            print(
                f"\n===== Supervised coef combo: outer={outer_coef:g}, inner={inner_coef:g} ({coef_tag}) =====",
                flush=True,
            )
        for market in combo_args.markets:
            label, config_module = MARKET_CONFIGS[market]
            seeds = resolve_seeds(combo_args, config_module)
            for seed in seeds:
                prefix = f"{coef_tag}-{market}-s{seed}" if multi_coef else f"{market}-s{seed}"
                print(
                    f"\n===== HRL training: {label} ({market}), seed={seed}, "
                    f"outer_pred={outer_coef:g}, inner_pred={inner_coef:g} =====",
                    flush=True,
                )
                command = build_child_command(combo_args, market, combo_root, seed)
                env = os.environ.copy()
                env.setdefault("MKL_THREADING_LAYER", "GNU")
                env.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                env["PYTHONUNBUFFERED"] = "1"
                if combo_args.cpu:
                    env["CUDA_VISIBLE_DEVICES"] = ""

                run_dir = combo_root / market
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / f"seed_{seed}_command.json").open("w", encoding="utf-8") as fh:
                    json.dump({"command": command, "cwd": str(ROOT)}, fh, ensure_ascii=False, indent=2)

                rc = stream_process(
                    command,
                    cwd=ROOT,
                    env=env,
                    log_path=combo_root / market / f"seed_{seed}.log",
                    prefix=prefix,
                    heartbeat_seconds=combo_args.heartbeat_seconds,
                )
                if rc != 0:
                    failures.append((prefix, rc))
                    print(f"[{prefix}] FAILED with exit code {rc}", flush=True)
                    if not combo_args.continue_on_error:
                        raise SystemExit(rc)
                else:
                    print(f"[{prefix}] completed. Outputs: {combo_root / market}", flush=True)

    if failures:
        print("\nFailures:", flush=True)
        for prefix, rc in failures:
            print(f"  - {prefix}: exit code {rc}", flush=True)
        raise SystemExit(1)
    print(f"\nAll HRL runs completed. Run root: {run_root}", flush=True)


def main():
    args = parse_args()
    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
