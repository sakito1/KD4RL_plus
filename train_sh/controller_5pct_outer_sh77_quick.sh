#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl-controller-guidance-quick-sh77 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/tongwenxuan/conda/envs/xuangu/bin/python -u run_hrl_training.py \
  --markets sh \
  --seeds 77 \
  --output_root results/controller_guidance_supervised_sh77 \
  --run_name controller_5pct_outer_sh77_alltrigger_pretrain50 \
  --device cuda \
  --frozen_hrl_checkpoint results/cmtflow_5stage_sh77/cmtflow_4_2_1_3_1_sh77/sh/ppo/seed_77/checkpoints/temp_warmup_outer.pth \
  --controller_only_finetune \
  --trade_num 10 \
  --outer_window 60 \
  --min_hold 30 \
  --max_hold 30 \
  --train_episodes_per_epoch 1 \
  --train_start_stride_days 1 \
  --warmup_outer_epochs 0 \
  --warmup_inner_epochs 0 \
  --joint_epochs 0 \
  --controller_epochs 1 \
  --controller_pretrain_only \
  --controller_pg_disable_inner \
  --controller_use_switch_supervision \
  --controller_sup_coef 0.1 \
  --controller_sup_pretrain_epochs 1 \
  --controller_sup_pretrain_rollout_len 300 \
  --controller_guidance_risk_threshold 0.05 \
  --controller_guidance_advantage_threshold 0.05 \
  --controller_guidance_pretrain_coef 1.0 \
  --controller_aux_pretrain_offpolicy \
  --controller_aux_replay_epochs 50 \
  --controller_rollout_len 300 \
  --controller_windows_per_epoch 12 \
  --controller_fixed_pool_limit 12 \
  --controller_pg_batch_windows 12 \
  --controller_pg_logprob_reduction mean \
  --controller_train_fixed_episodes \
  --controller_episode_batch_size 12 \
  --controller_episode_parallel_workers 6 \
  --controller_start_stride_days 5 \
  --controller_train_max_hold -1 \
  --controller_eval_max_hold -1 \
  --controller_window 30 \
  --controller_hidden_dim 64 \
  --controller_init_exit_bias 0.0 \
  --controller_return_coef 1.0 \
  --controller_mdd_coef 0.0 \
  --controller_max_switches 0 \
  --controller_max_switch_penalty_coef 0.001 \
  --controller_value_coef 0.0 \
  --no_controller_value_normalize_advantage \
  --controller_entropy_coef 0.01 \
  --controller_aux_return_coef 0.0 \
  --controller_aux_mdd_coef 0.1 \
  --controller_aux_mdd_target_scale 20.0 \
  --controller_aux_switch_adv_coef 1.0 \
  --controller_aux_switch_adv_target_scale 20.0 \
  --controller_aux_switch_adv_loss_type mse \
  --controller_compute_switch_advantage \
  --controller_reward_mode return_uplift \
  --controller_selection_metric return \
  --controller_no_hold_constraints \
  --controller_decision_mode daily \
  --controller_eval_decision_mode daily \
  --controller_eval_switch_threshold 0.5 \
  --controller_eval_diagnostics \
  --controller_eval_diag_thresholds 0.5 \
  --controller_val_interval_epochs 1 \
  --val_interval 1 \
  --train_monitor \
  --skip_test \
  --heartbeat_seconds 60
