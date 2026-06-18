#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-results/hrl_lookback60_hold30_inner_noaux_retrain/lookback60_hold30_inner_noaux_retrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_controller_first_joint}"
RUN_NAME="${RUN_NAME:-lookback60_noaux_frozen_controller_joint}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

NAS_SEEDS="${NAS_SEEDS:-49 50}"
SH_SEEDS="${SH_SEEDS:-90 83}"

OUTER_WINDOW="${OUTER_WINDOW:-60}"
MIN_HOLD="${MIN_HOLD:-30}"
MAX_HOLD="${MAX_HOLD:-30}"
TRAIN_EPISODES_PER_EPOCH="${TRAIN_EPISODES_PER_EPOCH:-5}"
TRAIN_START_STRIDE_DAYS="${TRAIN_START_STRIDE_DAYS:-10}"

CONTROLLER_EPOCHS="${CONTROLLER_EPOCHS:-10}"
CONTROLLER_ROLLOUT_LEN="${CONTROLLER_ROLLOUT_LEN:-600}"
CONTROLLER_WINDOWS_PER_EPOCH="${CONTROLLER_WINDOWS_PER_EPOCH:-30}"
CONTROLLER_EPISODE_BATCH_SIZE="${CONTROLLER_EPISODE_BATCH_SIZE:-12}"
CONTROLLER_EPISODE_PARALLEL_WORKERS="${CONTROLLER_EPISODE_PARALLEL_WORKERS:-12}"
CONTROLLER_START_STRIDE_DAYS="${CONTROLLER_START_STRIDE_DAYS:-1}"
CONTROLLER_WINDOW="${CONTROLLER_WINDOW:-30}"
CONTROLLER_RETURN_COEF="${CONTROLLER_RETURN_COEF:-1.0}"
CONTROLLER_MAX_SWITCHES="${CONTROLLER_MAX_SWITCHES:-40}"
CONTROLLER_MAX_SWITCH_PENALTY_COEF="${CONTROLLER_MAX_SWITCH_PENALTY_COEF:-0.00001}"
CONTROLLER_ENTROPY_COEF="${CONTROLLER_ENTROPY_COEF:-0.01}"
CONTROLLER_AUX_RETURN_COEF="${CONTROLLER_AUX_RETURN_COEF:-0.2}"
CONTROLLER_AUX_MDD_COEF="${CONTROLLER_AUX_MDD_COEF:-0.2}"
CONTROLLER_AUX_RETURN_TARGET_SCALE="${CONTROLLER_AUX_RETURN_TARGET_SCALE:-1.0}"
CONTROLLER_AUX_MDD_TARGET_SCALE="${CONTROLLER_AUX_MDD_TARGET_SCALE:-1.0}"

JOINT_EPOCHS="${JOINT_EPOCHS:-1}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.001}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

OUTER_PRED_COEF="${OUTER_PRED_COEF:-0.1}"
INNER_PRED_COEF="${INNER_PRED_COEF:-0.05}"
INNER_PRED_TARGET_SCALE="${INNER_PRED_TARGET_SCALE:-10}"

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
echo "Source root: $SOURCE_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NAS seeds: $NAS_SEEDS"
echo "SH seeds: $SH_SEEDS"
echo "Schedule: frozen outer/inner -> unconstrained controller PG -> controller-active joint finetune"
echo "Fixed baseline cycle: $MAX_HOLD, controller min-hold disabled, max switches per episode=$CONTROLLER_MAX_SWITCHES, outer_window=$OUTER_WINDOW"
echo "Controller dataset: window=$CONTROLLER_WINDOW, episode_len=$CONTROLLER_ROLLOUT_LEN, offsets=$CONTROLLER_WINDOWS_PER_EPOCH, batch=$CONTROLLER_EPISODE_BATCH_SIZE, workers=$CONTROLLER_EPISODE_PARALLEL_WORKERS"

run_one_seed() {
  local market="$1"
  local seed="$2"
  local checkpoint="$SOURCE_ROOT/$market/ppo/seed_${seed}/checkpoints/hrl_fixed_best.pth"
  local log_file="$OUTPUT_ROOT/logs/${RUN_NAME}_${market}_seed${seed}.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for ${market} seed ${seed}: $checkpoint" >&2
    return 1
  fi

  echo
  echo "===== Controller-first joint run: market=${market}, seed=${seed} ====="
  echo "Frozen HRL checkpoint: $checkpoint"

  "$PYTHON_BIN" -u run_hrl_training.py \
    --markets "$market" \
    --seeds "$seed" \
    --output_root "$OUTPUT_ROOT" \
    --run_name "$RUN_NAME" \
    --device cuda \
    --outer_window "$OUTER_WINDOW" \
    --min_hold "$MIN_HOLD" \
    --max_hold "$MAX_HOLD" \
    --train_episodes_per_epoch "$TRAIN_EPISODES_PER_EPOCH" \
    --train_start_stride_days "$TRAIN_START_STRIDE_DAYS" \
    --warmup_outer_epochs 0 \
    --warmup_inner_epochs 0 \
    --controller_epochs "$CONTROLLER_EPOCHS" \
    --controller_val_interval_epochs 1 \
    --controller_rollout_len "$CONTROLLER_ROLLOUT_LEN" \
    --controller_windows_per_epoch "$CONTROLLER_WINDOWS_PER_EPOCH" \
    --controller_pg_batch_windows "$CONTROLLER_EPISODE_BATCH_SIZE" \
    --controller_train_fixed_episodes \
    --controller_episode_batch_size "$CONTROLLER_EPISODE_BATCH_SIZE" \
    --controller_episode_parallel_workers "$CONTROLLER_EPISODE_PARALLEL_WORKERS" \
    --controller_start_stride_days "$CONTROLLER_START_STRIDE_DAYS" \
    --controller_window "$CONTROLLER_WINDOW" \
    --controller_return_coef "$CONTROLLER_RETURN_COEF" \
    --controller_mdd_coef 0.0 \
    --controller_count_min 0 \
    --controller_count_max 0 \
    --controller_max_switches "$CONTROLLER_MAX_SWITCHES" \
    --controller_max_switch_penalty_coef "$CONTROLLER_MAX_SWITCH_PENALTY_COEF" \
    --controller_switch_coef 0.0 \
    --controller_turnover_coef 0.0 \
    --controller_entropy_coef "$CONTROLLER_ENTROPY_COEF" \
    --controller_aux_return_coef "$CONTROLLER_AUX_RETURN_COEF" \
    --controller_aux_mdd_coef "$CONTROLLER_AUX_MDD_COEF" \
    --controller_aux_return_target_scale "$CONTROLLER_AUX_RETURN_TARGET_SCALE" \
    --controller_aux_mdd_target_scale "$CONTROLLER_AUX_MDD_TARGET_SCALE" \
    --controller_selection_metric return \
    --controller_no_hold_constraints \
    --joint_epochs "$JOINT_EPOCHS" \
    --joint_single_full_episode \
    --joint_lr_mult "$JOINT_LR_MULT" \
    --ppo_epochs "$PPO_EPOCHS" \
    --outer_pred_coef "$OUTER_PRED_COEF" \
    --inner_pred_coef "$INNER_PRED_COEF" \
    --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
    --model_selection_metric return \
    --inner_selection_metric return \
    --frozen_hrl_checkpoint "$checkpoint" \
    --controller_first_joint_finetune \
    --train_monitor \
    --heartbeat_seconds "$HEARTBEAT_SECONDS" \
    --continue_on_error \
    2>&1 | tee "$log_file"
}

for seed in $NAS_SEEDS; do
  run_one_seed nas "$seed"
done

for seed in $SH_SEEDS; do
  run_one_seed sh "$seed"
done

echo "Controller-first joint runs finished: $OUTPUT_ROOT/$RUN_NAME"
