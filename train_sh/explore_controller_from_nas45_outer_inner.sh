#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

CONTROLLER_SEED="${CONTROLLER_SEED:-45}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/controller_nas45_exploration}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-results/outer_inner_seed_sweep_k5/outer_inner_4_3_2_k5/nas/ppo/seed_45/checkpoints/hrl_fixed_best.pth}"
RUN_NAME="${RUN_NAME:-nas45_controller_sup_pg_seed${CONTROLLER_SEED}}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing frozen Outer+Inner checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME/nas/ppo/seed_${CONTROLLER_SEED}"
if [[ "$DRY_RUN" != "1" && "$ALLOW_EXISTING_OUTPUT" != "1" && -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing output: $RUN_DIR" >&2
  echo "Set ALLOW_EXISTING_OUTPUT=1 or choose another RUN_NAME." >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -u run_hrl_training.py
  --markets nas
  --seeds "$CONTROLLER_SEED"
  --output_root "$OUTPUT_ROOT"
  --run_name "$RUN_NAME"
  --device cuda
  --frozen_hrl_checkpoint "$SOURCE_CHECKPOINT"
  --controller_only_finetune
  --trade_num 5
  --outer_window 60
  --min_hold 30
  --max_hold 30
  --train_episodes_per_epoch 1
  --train_start_stride_days 1
  --warmup_outer_epochs 0
  --warmup_inner_epochs 0
  --joint_epochs 0
  --controller_epochs 5
  --controller_sup_coef 0.01
  --controller_sup_pretrain_epochs 4
  --controller_sup_pretrain_rollout_len 300
  --controller_aux_replay_epochs 30
  --controller_guidance_risk_threshold 0.05
  --controller_guidance_advantage_threshold 0.05
  --controller_guidance_pretrain_coef 1.0
  --controller_rollout_len 300
  --controller_windows_per_epoch 12
  --controller_fixed_pool_limit 12
  --controller_pg_batch_windows 12
  --controller_pg_logprob_reduction mean
  --controller_train_fixed_episodes
  --controller_episode_batch_size 12
  --controller_episode_parallel_workers 6
  --controller_start_stride_days 5
  --controller_train_max_hold 30
  --controller_eval_max_hold 30
  --controller_window 30
  --controller_hidden_dim 64
  --controller_init_exit_bias 0.0
  --controller_return_coef 1.0
  --controller_downside_coef 0.0
  --controller_mdd_coef 0.0
  --controller_reward_mode return_uplift
  --controller_selection_metric return
  --controller_no_hold_constraints
  --controller_decision_mode daily
  --controller_eval_decision_mode daily
  --controller_eval_switch_threshold 0.5
  --controller_eval_diagnostics
  --controller_eval_diag_thresholds 0.3 0.4 0.5 0.6 0.7
  --controller_max_switches 0
  --controller_max_switch_penalty_coef 0.0
  --controller_switch_rate_penalty_coef 0.01
  --controller_switch_rate_min 0.05
  --controller_switch_rate_max 0.15
  --controller_switch_rate_margin 0.1
  --controller_entropy_coef 0.001
  --controller_aux_return_coef 0.0
  --controller_aux_mdd_coef 0.01
  --controller_aux_mdd_target_scale 20.0
  --controller_aux_switch_adv_coef 0.01
  --controller_aux_switch_adv_target_scale 20.0
  --controller_aux_switch_adv_loss_type mse
  --controller_compute_switch_advantage
  --controller_use_switch_supervision
  --controller_val_interval_epochs 1
  --val_interval 1
  --model_selection_metric return
  --inner_selection_metric return
  --ppo_epochs 1
  --train_monitor
  --skip_test
  --heartbeat_seconds "$HEARTBEAT_SECONDS"
)

echo "Mode: sup_pg"
echo "Controller seed: $CONTROLLER_SEED"
echo "Frozen Outer+Inner: $SOURCE_CHECKPOINT"
echo "Output: $RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'COMMAND:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/${RUN_NAME}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-nas45-controller-sup-pg-${CONTROLLER_SEED}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
