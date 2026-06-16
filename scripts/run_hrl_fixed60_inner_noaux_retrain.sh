#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_lookback60_hold30_inner_noaux_retrain}"
RUN_NAME="${RUN_NAME:-lookback60_hold30_inner_noaux_retrain}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

OUTER_WINDOW="${OUTER_WINDOW:-60}"
MIN_HOLD="${MIN_HOLD:-30}"
MAX_HOLD="${MAX_HOLD:-30}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.001}"
WARMUP_OUTER_EPOCHS="${WARMUP_OUTER_EPOCHS:-2}"
WARMUP_INNER_EPOCHS="${WARMUP_INNER_EPOCHS:-2}"
OUTER_PPO_EPOCHS="${OUTER_PPO_EPOCHS:-1}"
INNER_SEGMENTS_PER_EPISODE="${INNER_SEGMENTS_PER_EPISODE:-20}"
INNER_EPISODE_LEN="${INNER_EPISODE_LEN:-$((MAX_HOLD * INNER_SEGMENTS_PER_EPISODE))}"
INNER_START_STRIDE_DAYS="${INNER_START_STRIDE_DAYS:-1}"
INNER_TRAIN_EPISODES_PER_EPOCH="${INNER_TRAIN_EPISODES_PER_EPOCH:-30}"
INNER_EPISODE_BATCH_SIZE="${INNER_EPISODE_BATCH_SIZE:-12}"
INNER_EPISODE_PARALLEL_WORKERS="${INNER_EPISODE_PARALLEL_WORKERS:-12}"
INNER_ROLLOUT_UPDATE_STEPS="${INNER_ROLLOUT_UPDATE_STEPS:-$INNER_EPISODE_LEN}"
INNER_PPO_EPOCHS="${INNER_PPO_EPOCHS:-1}"

OUTER_PRED_COEF="${OUTER_PRED_COEF:-0.1}"
INNER_PRED_COEF="${INNER_PRED_COEF:-0.05}"
INNER_PRED_TARGET_SCALE="${INNER_PRED_TARGET_SCALE:-10}"

NAS_SEEDS="${NAS_SEEDS:-44 46}"
SH_SEEDS="${SH_SEEDS:-48 49 77 78 79 80 81 82 83 84 85 86 87 88 89 90}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kd4rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

mkdir -p "$OUTPUT_ROOT/logs"

echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Inner auxiliary head coef: $INNER_PRED_COEF"
echo "NAS seeds: $NAS_SEEDS"
echo "SH seeds: $SH_SEEDS"

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
  --inner_train_episodes_per_epoch "$INNER_TRAIN_EPISODES_PER_EPOCH" \
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

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets nas \
  --seeds $NAS_SEEDS \
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
  --inner_train_episodes_per_epoch "$INNER_TRAIN_EPISODES_PER_EPOCH" \
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
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_nas.log"



echo "Inner-noaux retrain finished: $OUTPUT_ROOT/$RUN_NAME"
