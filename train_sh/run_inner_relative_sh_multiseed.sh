#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/outer_inner_relative_sh_multiseed}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-outer_inner_relative_sh}"
SH_SEEDS="${SH_SEEDS-44 46 49 54}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"

ASU_COEF="${ASU_COEF:-0.05}"
OUTER_EPOCHS="${OUTER_EPOCHS:-4}"
INNER_EPOCHS="${INNER_EPOCHS:-5}"
JOINT_EPOCHS="${JOINT_EPOCHS:-2}"
TRAIN_EPISODES_PER_EPOCH="${TRAIN_EPISODES_PER_EPOCH:-5}"
INNER_EPISODES_PER_EPOCH="${INNER_EPISODES_PER_EPOCH:-30}"
INNER_EPISODE_LEN="${INNER_EPISODE_LEN:-600}"
INNER_EPISODE_BATCH_SIZE="${INNER_EPISODE_BATCH_SIZE:-12}"
INNER_PARALLEL_WORKERS="${INNER_PARALLEL_WORKERS:-12}"
INNER_BATCH_SIZE="${INNER_BATCH_SIZE:-1200}"
INNER_LR="${INNER_LR:-0.001}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kd4rl-inner-relative-sh}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED_LIST=()
if [[ -n "${SH_SEEDS//[[:space:]]/}" ]]; then
  read -r -a SEED_LIST <<<"$SH_SEEDS"
fi

if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
  echo "No SH seeds were provided." >&2
  exit 1
fi
for epoch_setting in \
  "OUTER_EPOCHS=$OUTER_EPOCHS" \
  "INNER_EPOCHS=$INNER_EPOCHS" \
  "JOINT_EPOCHS=$JOINT_EPOCHS"; do
  epoch_value="${epoch_setting#*=}"
  if ! [[ "$epoch_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${epoch_setting%%=*} must be a positive integer: $epoch_value" >&2
    exit 1
  fi
done
for seed in "${SEED_LIST[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid SH seed: $seed" >&2
    exit 1
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
  fi
  mkdir -p "$OUTPUT_ROOT/logs"
fi

echo "SH Outer + Relative Inner seed sweep"
echo "Seeds: ${SEED_LIST[*]}"
echo "Schedule: Outer $OUTER_EPOCHS -> Inner $INNER_EPOCHS -> Joint $JOINT_EPOCHS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Output root: $OUTPUT_ROOT"

for seed in "${SEED_LIST[@]}"; do
  run_name="${RUN_NAME_PREFIX}_seed${seed}"
  log_file="$OUTPUT_ROOT/logs/${run_name}.log"
  command=(
    "$PYTHON_BIN" -u run_hrl_training.py
    --markets sh
    --seeds "$seed"
    --output_root "$OUTPUT_ROOT"
    --run_name "$run_name"
    --device cuda
    --inner_encoder_mode relative_tcn_attn
    --inner_norm_mode close_anchor
    --inner_asu_coef "$ASU_COEF"
    --inner_pred_coef 0
    --outer_pred_coef 0.1
    --inner_timing_diagnostics
    --trade_num 10
    --outer_window 60
    --min_hold 30
    --max_hold 30
    --train_episodes_per_epoch "$TRAIN_EPISODES_PER_EPOCH"
    --train_start_stride_days 1
    --warmup_outer_epochs "$OUTER_EPOCHS"
    --warmup_inner_epochs "$INNER_EPOCHS"
    --warmup_monitor_epochs 0
    --joint_epochs "$JOINT_EPOCHS"
    --joint_single_full_episode
    --joint_lr_mult 0.001
    --lr_inner "$INNER_LR"
    --ppo_epochs 1
    --inner_ppo_epochs 1
    --inner_train_fixed_episodes
    --inner_episode_len "$INNER_EPISODE_LEN"
    --inner_train_episodes_per_epoch "$INNER_EPISODES_PER_EPOCH"
    --inner_start_stride_days 1
    --inner_episode_batch_size "$INNER_EPISODE_BATCH_SIZE"
    --inner_episode_parallel_workers "$INNER_PARALLEL_WORKERS"
    --inner_rollout_update_steps "$INNER_EPISODE_LEN"
    --inner_batch_size "$INNER_BATCH_SIZE"
    --outer_reward_mode return
    --model_selection_metric sharpe
    --inner_selection_metric return
    --no_train_controller
    --heartbeat_seconds "$HEARTBEAT_SECONDS"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi

  echo "Starting SH seed=$seed; log=$log_file"
  "${command[@]}" 2>&1 | tee "$log_file"
done

echo "All SH Outer + Relative Inner seed runs completed."
