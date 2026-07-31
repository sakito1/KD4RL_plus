#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_SCRIPT="$REPO_ROOT/train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh"
WORKER_SCRIPT="$REPO_ROOT/scripts/run_multi_market_multi_k_seed_worker.sh"

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/k15_then_k5_3gpu_seed50_61}"
RUN_PREFIX="${RUN_PREFIX:-e2e_k15_then_k5_seed50_61}"
GPU_IDS="${GPU_IDS:-0 1 2}"
SEEDS="${SEEDS:-50 51 52 53 54 55 56 57 58 59 60 61}"
MARKETS="${MARKETS:-nas sh}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_unique_values() {
  local label="$1"
  local value
  shift
  declare -A seen_values=()

  for value in "$@"; do
    if [[ -n "${seen_values[$value]+present}" ]]; then
      die "$label contains a duplicate value: $value"
    fi
    seen_values["$value"]=1
  done
}

read -r -a gpu_ids <<< "$GPU_IDS"
read -r -a seeds <<< "$SEEDS"
read -r -a markets <<< "$MARKETS"

(( ${#gpu_ids[@]} == 3 )) || die "GPU_IDS must contain exactly 3 GPU indices."
(( ${#seeds[@]} == 12 )) || die "SEEDS must contain exactly 12 seeds."
(( ${#markets[@]} > 0 )) || die "MARKETS must contain at least one market."

for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU_IDS contains a non-negative-integer violation: $gpu"
done
for seed in "${seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "SEEDS must contain non-negative integers: $seed"
done
for market in "${markets[@]}"; do
  case "$market" in
    nas|sh) ;;
    *) die "Unsupported market: $market (expected nas or sh)" ;;
  esac
done

require_unique_values "GPU_IDS" "${gpu_ids[@]}"
require_unique_values "SEEDS" "${seeds[@]}"
require_unique_values "MARKETS" "${markets[@]}"

[[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
[[ -f "$TRAINING_SCRIPT" ]] || die "E2E training script not found: $TRAINING_SCRIPT"
[[ -f "$WORKER_SCRIPT" ]] || die "Sweep worker script not found: $WORKER_SCRIPT"
command -v setsid >/dev/null 2>&1 || die "setsid is required for safe worker cleanup."

seed_shards=("" "" "")
for index in "${!seeds[@]}"; do
  slot=$((index % 3))
  if [[ -n "${seed_shards[$slot]}" ]]; then
    seed_shards[$slot]+=" ${seeds[$index]}"
  else
    seed_shards[$slot]="${seeds[$index]}"
  fi
done

phase_ks=(15 5)
total_runs=$((${#seeds[@]} * ${#markets[@]} * ${#phase_ks[@]}))
phase_runs=$((${#seeds[@]} * ${#markets[@]}))

echo "Three-GPU staged abk E2E K sweep"
echo "repo_root=$REPO_ROOT"
echo "training_script=$TRAINING_SCRIPT"
echo "output_root=$OUTPUT_ROOT"
echo "gpu_ids=$GPU_IDS"
echo "markets=$MARKETS"
echo "seeds=$SEEDS"
echo "gpu_workers=3 phases=2 markets=${#markets[@]} total_seed_market_runs=$total_runs"

for phase_index in "${!phase_ks[@]}"; do
  phase=$((phase_index + 1))
  k="${phase_ks[$phase_index]}"
  echo "phase=$phase k=$k seed_market_runs=$phase_runs"
  for gpu_slot in "${!gpu_ids[@]}"; do
    echo "phase=$phase k=$k gpu=${gpu_ids[$gpu_slot]} seeds=${seed_shards[$gpu_slot]} markets=$MARKETS"
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; no training processes were started."
  exit 0
fi

if [[ "$OUTPUT_ROOT" == /* ]]; then
  output_root_abs="$OUTPUT_ROOT"
else
  output_root_abs="$REPO_ROOT/$OUTPUT_ROOT"
fi
launcher_log_dir="$output_root_abs/launcher_logs"
mkdir -p "$launcher_log_dir"

pids=()
worker_labels=()

terminate_workers() {
  local signal="$1"
  local status="$2"
  local pid
  trap - INT TERM
  for pid in "${pids[@]:-}"; do
    kill "-$signal" -- "-$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap 'terminate_workers INT 130' INT
trap 'terminate_workers TERM 143' TERM

run_phase() {
  local phase="$1"
  local k="$2"
  local gpu_slot gpu shard worker_log worker_label index market
  local -a worker_args failures

  pids=()
  worker_labels=()
  failures=()

  echo
  echo "===== PHASE $phase START: K=$k ====="

  for gpu_slot in "${!gpu_ids[@]}"; do
    gpu="${gpu_ids[$gpu_slot]}"
    shard="${seed_shards[$gpu_slot]}"
    worker_log="$launcher_log_dir/k${k}_gpu_${gpu}.log"
    worker_label="k${k}_gpu_${gpu}"
    worker_args=(
      "$gpu"
      "$TRAINING_SCRIPT"
      "${RUN_PREFIX}_gpu${gpu}"
      "$PYTHON_BIN"
      "$OUTPUT_ROOT"
      "$shard"
      "$shard"
    )

    for market in "${markets[@]}"; do
      worker_args+=("$market" "$k" "$shard")
    done

    setsid bash "$WORKER_SCRIPT" "${worker_args[@]}" \
      > >(tee "$worker_log") 2>&1 &
    pids+=("$!")
    worker_labels+=("$worker_label")
  done

  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failures+=("${worker_labels[$index]}")
    fi
  done
  pids=()
  worker_labels=()

  if (( ${#failures[@]} > 0 )); then
    echo "Phase $phase (K=$k) failed workers: ${failures[*]}" >&2
    return 1
  fi

  echo "===== PHASE $phase DONE: K=$k ====="
}

# Hard barrier: all K=15 workers must finish successfully before K=5 starts.
run_phase 1 15
run_phase 2 5

trap - INT TERM
echo
echo "All K=15 and K=5 experiments completed successfully."
echo "Outputs: $output_root_abs"
