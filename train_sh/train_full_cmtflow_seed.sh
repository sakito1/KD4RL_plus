#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MARKET="${MARKET:-nas}"
SEED="${SEED:-44}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/full_cmtflow_seed_sweep_4gpu}"
RUN_NAME="${RUN_NAME:-${MARKET}_full_42135_seed${SEED}}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

if [[ "$MARKET" != "nas" && "$MARKET" != "sh" ]]; then
  echo "MARKET must be nas or sh: $MARKET" >&2
  exit 1
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: $SEED" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME/$MARKET/ppo/seed_${SEED}"
if [[ "$DRY_RUN" != "1" && "$ALLOW_EXISTING_OUTPUT" != "1" && -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing output: $RUN_DIR" >&2
  echo "Set ALLOW_EXISTING_OUTPUT=1 or choose another RUN_NAME." >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -u run_hrl_training.py
  --markets "$MARKET"
  --seeds "$SEED"
  --output_root "$OUTPUT_ROOT"
  --run_name "$RUN_NAME"
  --device cuda
  --trade_num 5
  --outer_window 60
  --min_hold 30
  --max_hold 30
  --train_episodes_per_epoch 5
  --train_start_stride_days 1
  --warmup_outer_epochs 4
  --warmup_inner_epochs 2
  --joint_epochs 1
  --joint_single_full_episode
  --inner_train_fixed_episodes
  --inner_episode_len 600
  --inner_train_episodes_per_epoch 30
  --inner_start_stride_days 1
  --inner_episode_batch_size 12
  --inner_episode_parallel_workers 6
  --inner_rollout_update_steps 600
  --inner_ppo_epochs 1
  --joint_lr_mult 0.001
  --outer_pred_coef 0.1
  --inner_pred_coef 0.05
  --inner_pred_target_scale 10
  --outer_reward_mode return
  --model_selection_metric sharpe
  --inner_selection_metric return
  --controller_epochs 5
  --controller_sup_coef 0.01
  --controller_sup_pretrain_epochs 3
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
  --ppo_epochs 1
  --train_monitor
  --heartbeat_seconds "$HEARTBEAT_SECONDS"
)

echo "Full CMTFlow training and test"
echo "Schedule: Outer 4 -> Inner 2 -> Outer+Inner joint 1 -> Controller pretrain 3 -> Controller PG 5 -> Test"
echo "Market: $MARKET"
echo "Seed: $SEED"
echo "GPU: $GPU_ID"
echo "Output: $RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q COMMAND:' "$GPU_ID"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/${RUN_NAME}.log"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-full-cmtflow-${MARKET}-${SEED}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
