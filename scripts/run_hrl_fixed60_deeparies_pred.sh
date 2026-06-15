#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_lookback60_hold30_bank}"
RUN_NAME="${RUN_NAME:-lookback60_hold30_inner_episode_batch_deeparies_pred}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

# Fixed-hold HRL seed bank. Controller is intentionally disabled here.
OUTER_WINDOW="${OUTER_WINDOW:-60}"
MIN_HOLD="${MIN_HOLD:-30}"
MAX_HOLD="${MAX_HOLD:-30}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.001}"
WARMUP_OUTER_EPOCHS="${WARMUP_OUTER_EPOCHS:-2}"
WARMUP_INNER_EPOCHS="${WARMUP_INNER_EPOCHS:-1}"
OUTER_PPO_EPOCHS="${OUTER_PPO_EPOCHS:-1}"
INNER_SEGMENTS_PER_EPISODE="${INNER_SEGMENTS_PER_EPISODE:-10}"
INNER_EPISODE_LEN="${INNER_EPISODE_LEN:-$((MAX_HOLD * INNER_SEGMENTS_PER_EPISODE))}"
INNER_START_STRIDE_DAYS="${INNER_START_STRIDE_DAYS:-30}"
INNER_EPISODE_BATCH_SIZE="${INNER_EPISODE_BATCH_SIZE:-12}"
INNER_EPISODE_PARALLEL_WORKERS="${INNER_EPISODE_PARALLEL_WORKERS:-12}"
INNER_UPDATE_HOLD_PERIODS="${INNER_UPDATE_HOLD_PERIODS:-10}"
INNER_ROLLOUT_UPDATE_STEPS="${INNER_ROLLOUT_UPDATE_STEPS:-0}"
INNER_PPO_EPOCHS="${INNER_PPO_EPOCHS:-1}"

# DeepAries-style auxiliary supervision:
# action heads choose portfolios; pred_heads predict stock-level returns.
OUTER_PRED_COEF="${OUTER_PRED_COEF:-0.1}"
INNER_PRED_COEF="${INNER_PRED_COEF:-0.05}"
INNER_PRED_TARGET_SCALE="${INNER_PRED_TARGET_SCALE:-10}"

NAS_SEEDS="${NAS_SEEDS:-42 43 44 45 46}"
# SH_SEEDS="${SH_SEEDS:-42 43 44 45 46 47 48 49 50 51 52}"
SH_SEEDS="${SH_SEEDS:-45 46 47 48 49 50 51 52 53 54 55}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kd4rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f run_hrl_training.py ]]; then
  echo "run_hrl_training.py not found. Please run this script from KD4RL_plus." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"

echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON_BIN"
echo "Outer lookback window: $OUTER_WINDOW"
echo "Fixed holding: min_hold=$MIN_HOLD max_hold=$MAX_HOLD"
echo "Training epochs: outer=$WARMUP_OUTER_EPOCHS inner=$WARMUP_INNER_EPOCHS joint=1 controller=skipped"
echo "PPO epochs: outer/joint=$OUTER_PPO_EPOCHS inner=$INNER_PPO_EPOCHS"
echo "Inner schedule: episode_len=${INNER_EPISODE_LEN} (${INNER_SEGMENTS_PER_EPISODE} fixed-hold periods), start_stride=${INNER_START_STRIDE_DAYS}, episode_batch_size=${INNER_EPISODE_BATCH_SIZE}, parallel_workers=${INNER_EPISODE_PARALLEL_WORKERS}, update_after_batch=1, in_episode_update_steps=${INNER_ROLLOUT_UPDATE_STEPS}, ppo_epochs=$INNER_PPO_EPOCHS"
echo "Joint schedule: one full train-to-end episode per joint epoch"
echo "DeepAries-style pred heads: outer=$OUTER_PRED_COEF inner=$INNER_PRED_COEF inner_target_scale=$INNER_PRED_TARGET_SCALE"
echo "NAS seeds: $NAS_SEEDS"
echo "SH seeds: $SH_SEEDS"

# "$PYTHON_BIN" -u run_hrl_training.py \
#   --markets nas \
#   --seeds $NAS_SEEDS \
#   --output_root "$OUTPUT_ROOT" \
#   --run_name "$RUN_NAME" \
#   --device cuda \
#   --outer_window "$OUTER_WINDOW" \
#   --min_hold "$MIN_HOLD" \
#   --max_hold "$MAX_HOLD" \
#   --outer_pred_coef "$OUTER_PRED_COEF" \
#   --inner_pred_coef "$INNER_PRED_COEF" \
#   --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
#   --joint_lr_mult "$JOINT_LR_MULT" \
#   --ppo_epochs "$OUTER_PPO_EPOCHS" \
#   --joint_single_full_episode \
#   --warmup_outer_epochs "$WARMUP_OUTER_EPOCHS" \
#   --warmup_inner_epochs "$WARMUP_INNER_EPOCHS" \
#   --inner_train_fixed_episodes \
#   --inner_episode_len "$INNER_EPISODE_LEN" \
#   --inner_start_stride_days "$INNER_START_STRIDE_DAYS" \
#   --inner_episode_batch_size "$INNER_EPISODE_BATCH_SIZE" \
#   --inner_episode_parallel_workers "$INNER_EPISODE_PARALLEL_WORKERS" \
#   --inner_rollout_update_steps "$INNER_ROLLOUT_UPDATE_STEPS" \
#   --inner_ppo_epochs "$INNER_PPO_EPOCHS" \
#   --model_selection_metric sharpe \
#   --inner_selection_metric return \
#   --no_train_controller \
#   --heartbeat_seconds "$HEARTBEAT_SECONDS" \
#   --continue_on_error \
#   2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_nas.log"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets sh \
  --seeds $SH_SEEDS \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --outer_window "$OUTER_WINDOW" \
  --min_hold "$MIN_HOLD" \
  --max_hold "$MAX_HOLD" \
  --outer_pred_coef "$OUTER_PRED_COEF" \
  --inner_pred_coef "$INNER_PRED_COEF" \
  --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --ppo_epochs "$OUTER_PPO_EPOCHS" \
  --joint_single_full_episode \
  --warmup_outer_epochs "$WARMUP_OUTER_EPOCHS" \
  --warmup_inner_epochs "$WARMUP_INNER_EPOCHS" \
  --inner_train_fixed_episodes \
  --inner_episode_len "$INNER_EPISODE_LEN" \
  --inner_start_stride_days "$INNER_START_STRIDE_DAYS" \
  --inner_episode_batch_size "$INNER_EPISODE_BATCH_SIZE" \
  --inner_episode_parallel_workers "$INNER_EPISODE_PARALLEL_WORKERS" \
  --inner_rollout_update_steps "$INNER_ROLLOUT_UPDATE_STEPS" \
  --inner_ppo_epochs "$INNER_PPO_EPOCHS" \
  --model_selection_metric sharpe \
  --inner_selection_metric return \
  --no_train_controller \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_sh.log"

echo "All requested lookback-60 hold-30 HRL seed-bank runs finished: $OUTPUT_ROOT/$RUN_NAME"
