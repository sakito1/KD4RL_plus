#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SINGLE_SCRIPT="$ROOT_DIR/train_sh/train_full_cmtflow_seed.sh"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/full_cmtflow_seed_sweep_4gpu}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
NAS_SEEDS="${NAS_SEEDS-44 45 47 50 56 57 58}"
SH_SEEDS="${SH_SEEDS-44 46 49 54}"

if [[ ! -f "$SINGLE_SCRIPT" ]]; then
  echo "Missing single-job script: $SINGLE_SCRIPT" >&2
  exit 1
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python was not found in PATH; activate the intended environment or set PYTHON_BIN." >&2
  exit 1
fi
if ! [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS_PER_GPU must be a positive integer: $JOBS_PER_GPU" >&2
  exit 1
fi

NAS_SEED_LIST=()
SH_SEED_LIST=()
if [[ -n "${NAS_SEEDS//[[:space:]]/}" ]]; then
  read -r -a NAS_SEED_LIST <<<"$NAS_SEEDS"
fi
if [[ -n "${SH_SEEDS//[[:space:]]/}" ]]; then
  read -r -a SH_SEED_LIST <<<"$SH_SEEDS"
fi

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

GPU_IDS=("$GPU0" "$GPU1" "$GPU2" "$GPU3")
gpu_count="${#GPU_IDS[@]}"
slot_count=$((gpu_count * JOBS_PER_GPU))
SLOT_QUEUES=()
for index in "${!jobs[@]}"; do
  slot=$((index % slot_count))
  SLOT_QUEUES[$slot]="${SLOT_QUEUES[$slot]:-}${SLOT_QUEUES[$slot]:+ }${jobs[$index]}"
done

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_ROOT/scheduler_logs"
fi

run_job() {
  local gpu="$1"
  local market="$2"
  local seed="$3"
  local run_name="${market}_full_42135_seed${seed}"
  local log_file="$OUTPUT_ROOT/scheduler_logs/${market}_seed${seed}_gpu${gpu}.log"

  echo "GPU ${gpu} starting market=${market} seed=${seed}; log=$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    MARKET="$market" \
    SEED="$seed" \
    GPU_ID="$gpu" \
    DRY_RUN=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_NAME="$run_name" \
    HEARTBEAT_SECONDS="$HEARTBEAT_SECONDS" \
    bash "$SINGLE_SCRIPT"
    return
  fi

  MARKET="$market" \
  SEED="$seed" \
  GPU_ID="$gpu" \
  DRY_RUN=0 \
  ALLOW_EXISTING_OUTPUT="$ALLOW_EXISTING_OUTPUT" \
  PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  RUN_NAME="$run_name" \
  HEARTBEAT_SECONDS="$HEARTBEAT_SECONDS" \
  bash "$SINGLE_SCRIPT" >"$log_file" 2>&1
}

run_queue() {
  local gpu="$1"
  local lane="$2"
  local queue_text="$3"
  local queue=()
  local job
  local market
  local seed
  local queue_status=0

  if [[ -n "$queue_text" ]]; then
    read -r -a queue <<<"$queue_text"
  fi
  echo "GPU ${gpu} lane ${lane} queue: ${queue[*]:-(empty)}"
  for job in "${queue[@]}"; do
    market="${job%%:*}"
    seed="${job#*:}"
    if ! run_job "$gpu" "$market" "$seed"; then
      echo "GPU ${gpu} task failed: market=${market} seed=${seed}" >&2
      queue_status=1
    fi
  done
  return "$queue_status"
}

write_test_summary() {
  local summary_file="$OUTPUT_ROOT/test_results_summary.txt"
  local index
  local job
  local market
  local seed
  local slot
  local gpu_index
  local gpu
  local log_file

  : >"$summary_file"
  for index in "${!jobs[@]}"; do
    job="${jobs[$index]}"
    market="${job%%:*}"
    seed="${job#*:}"
    slot=$((index % slot_count))
    gpu_index=$((slot % gpu_count))
    gpu="${GPU_IDS[$gpu_index]}"
    log_file="$OUTPUT_ROOT/scheduler_logs/${market}_seed${seed}_gpu${gpu}.log"
    {
      echo "===== market=${market} seed=${seed} gpu=${gpu} ====="
      if [[ -f "$log_file" ]]; then
        grep -E \
          "TEST REPORT|Controller eval exit_prob:|Switches[[:space:]]*:|Switch detail:|Total Ret[[:space:]]*:|Ann Ret[[:space:]]*:|Ann Vol[[:space:]]*:|Sharpe[[:space:]]*:|Max DD[[:space:]]*:" \
          "$log_file" || echo "(no test report found; inspect $log_file)"
      else
        echo "(missing scheduler log: $log_file)"
      fi
      echo
    } >>"$summary_file"
  done
  echo "Test summary: $summary_file"
}

echo "Full CMTFlow seed sweep"
echo "Schedule: Outer 4 -> Inner 2 -> Outer+Inner joint 1 -> Controller pretrain 3 -> Controller PG 5 -> Test"
echo "Output: $OUTPUT_ROOT"
echo "Concurrent jobs per GPU: $JOBS_PER_GPU"
echo "NAS seeds: ${NAS_SEED_LIST[*]:-(none)}"
echo "SH seeds: ${SH_SEED_LIST[*]:-(none)}"

pids=()
names=()
for ((slot = 0; slot < slot_count; slot++)); do
  gpu_index=$((slot % gpu_count))
  lane=$((slot / gpu_count))
  gpu="${GPU_IDS[$gpu_index]}"
  run_queue "$gpu" "$lane" "${SLOT_QUEUES[$slot]:-}" &
  pids+=("$!")
  names+=("gpu${gpu}-lane${lane}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]} completed."
  else
    echo "${names[$index]} failed; inspect its scheduler logs." >&2
    status=1
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  write_test_summary
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "All full CMTFlow seed tasks completed."
