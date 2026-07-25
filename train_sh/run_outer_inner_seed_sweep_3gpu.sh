#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/outer_inner_seed_sweep_k5}"
RUN_NAME="${RUN_NAME:-outer_inner_4_1_1_k5}"
DRY_RUN="${DRY_RUN:-0}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
NAS_SEEDS="${NAS_SEEDS-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60}"
SH_SEEDS="${SH_SEEDS-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60}"

NAS_SEED_LIST=()
SH_SEED_LIST=()
if [[ -n "${NAS_SEEDS//[[:space:]]/}" ]]; then
  read -r -a NAS_SEED_LIST <<<"$NAS_SEEDS"
fi
if [[ -n "${SH_SEEDS//[[:space:]]/}" ]]; then
  read -r -a SH_SEED_LIST <<<"$SH_SEEDS"
fi

mkdir -p "$OUTPUT_ROOT/logs"

COMMON_ARGS=(
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
  --warmup_inner_epochs 1
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
  --ppo_epochs 1
  --joint_lr_mult 0.001
  --outer_pred_coef 0.1
  --inner_pred_coef 0.05
  --inner_pred_target_scale 10
  --outer_reward_mode return
  --model_selection_metric sharpe
  --inner_selection_metric return
  --no_train_controller
  --heartbeat_seconds 300
  --continue_on_error
)

run_job() {
  local gpu="$1"
  local market="$2"
  local seed="$3"
  local tag="${market}_seed${seed}_gpu${gpu}"
  local log_file="$OUTPUT_ROOT/logs/${tag}.log"
  local command=(
    "$PYTHON_BIN" -u run_hrl_training.py
    --markets "$market"
    --seeds "$seed"
    "${COMMON_ARGS[@]}"
  )

  echo "GPU ${gpu} starting ${market} seed=${seed}; log=$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    local rendered
    printf -v rendered ' %q' "${command[@]}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}${rendered}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  MPLCONFIGDIR="/tmp/mpl-outer-inner-${tag}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${command[@]}" >"$log_file" 2>&1
}

run_queue() {
  local gpu="$1"
  local queue_name="$2"
  local -n queue_ref="$queue_name"
  local job
  local market
  local seed
  local queue_status=0

  echo "GPU ${gpu} queue: ${queue_ref[*]:-(empty)}"
  for job in "${queue_ref[@]}"; do
    market="${job%%:*}"
    seed="${job#*:}"
    if ! run_job "$gpu" "$market" "$seed"; then
      echo "GPU ${gpu} task failed: ${market} seed=${seed}" >&2
      queue_status=1
    fi
  done
  return "$queue_status"
}

if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "Outer+Inner seed sweep"
echo "Output: $OUTPUT_ROOT/$RUN_NAME"
echo "Schedule: Outer 4 -> Inner 1 -> Joint 1"
echo "Top-K: 5; transaction cost is read from config as 1e-4"
echo "NAS seeds: ${NAS_SEED_LIST[*]:-(none)}"
echo "SH seeds: ${SH_SEED_LIST[*]:-(none)}"

jobs=()
for seed in "${NAS_SEED_LIST[@]}"; do
  jobs+=("nas:${seed}")
done
for seed in "${SH_SEED_LIST[@]}"; do
  jobs+=("sh:${seed}")
done
if [[ "${#jobs[@]}" -eq 0 ]]; then
  echo "No seed jobs were provided." >&2
  exit 1
fi

GPU0_QUEUE=()
GPU1_QUEUE=()
GPU2_QUEUE=()
for index in "${!jobs[@]}"; do
  case $((index % 3)) in
    0) GPU0_QUEUE+=("${jobs[$index]}") ;;
    1) GPU1_QUEUE+=("${jobs[$index]}") ;;
    2) GPU2_QUEUE+=("${jobs[$index]}") ;;
  esac
done

pids=()
names=()

run_queue "$GPU0" GPU0_QUEUE &
pids+=("$!")
names+=("gpu${GPU0}")

run_queue "$GPU1" GPU1_QUEUE &
pids+=("$!")
names+=("gpu${GPU1}")

run_queue "$GPU2" GPU2_QUEUE &
pids+=("$!")
names+=("gpu${GPU2}")

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]} completed."
  else
    echo "${names[$index]} failed; inspect its log." >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "All Outer+Inner seed runs completed: $OUTPUT_ROOT/$RUN_NAME"
