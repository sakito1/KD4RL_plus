#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_SCRIPT="$REPO_ROOT/train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh"
WORKER_SCRIPT="$REPO_ROOT/scripts/run_multi_market_multi_k_seed_worker.sh"

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/k_asset_selection_seed_sweep}"
RUN_PREFIX="${RUN_PREFIX:-e2e_asset_selection}"
GPU_IDS="${GPU_IDS:-0}"
MARKETS="${MARKETS:-nas sh}"
K_VALUES="${K_VALUES:-5 15}"
NAS_SEEDS="${NAS_SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56}"
SH_SEEDS="${SH_SEEDS:-83 84 85 86 87 88 89 90 91 92 93 94 95 96 97}"
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
read -r -a markets <<< "$MARKETS"
read -r -a k_values <<< "$K_VALUES"
read -r -a nas_seeds <<< "$NAS_SEEDS"
read -r -a sh_seeds <<< "$SH_SEEDS"

(( ${#gpu_ids[@]} > 0 )) || die "GPU_IDS must contain at least one GPU index."
(( ${#markets[@]} > 0 )) || die "MARKETS must contain at least one market."
(( ${#k_values[@]} > 0 )) || die "K_VALUES must contain at least one K."
(( ${#nas_seeds[@]} > 0 )) || die "NAS_SEEDS must contain at least one seed."
(( ${#sh_seeds[@]} > 0 )) || die "SH_SEEDS must contain at least one seed."

for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU_IDS contains a non-negative-integer violation: $gpu"
done

for market in "${markets[@]}"; do
  case "$market" in
    nas|sh) ;;
    *) die "Unsupported market: $market (expected nas or sh)" ;;
  esac
done

for k in "${k_values[@]}"; do
  [[ "$k" =~ ^[1-9][0-9]*$ ]] || die "K_VALUES contains a non-positive integer: $k"
done

for seed in "${nas_seeds[@]}" "${sh_seeds[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "Seed lists must contain non-negative integers: $seed"
done

require_unique_values "GPU_IDS" "${gpu_ids[@]}"
require_unique_values "MARKETS" "${markets[@]}"
require_unique_values "K_VALUES" "${k_values[@]}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  die "PYTHON_BIN is not executable: $PYTHON_BIN"
fi
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
  die "Final training script not found: $TRAINING_SCRIPT"
fi
if [[ ! -f "$WORKER_SCRIPT" ]]; then
  die "Sweep worker script not found: $WORKER_SCRIPT"
fi
if ! command -v setsid >/dev/null 2>&1; then
  die "setsid is required for safe worker process cleanup."
fi

job_markets=()
job_k_values=()
job_gpu_slots=()
job_gpu_ids=()
job_seed_strings=()
total_seed_runs=0

for market in "${markets[@]}"; do
  if [[ "$market" == "nas" ]]; then
    seed_string="$NAS_SEEDS"
    seed_count=${#nas_seeds[@]}
    max_assets=39
  else
    seed_string="$SH_SEEDS"
    seed_count=${#sh_seeds[@]}
    max_assets=53
  fi

  for k in "${k_values[@]}"; do
    (( k <= max_assets )) || die "K=$k exceeds the $market asset count ($max_assets)."
    job_index=${#job_markets[@]}
    gpu_slot=$((job_index % ${#gpu_ids[@]}))
    job_markets+=("$market")
    job_k_values+=("$k")
    job_gpu_slots+=("$gpu_slot")
    job_gpu_ids+=("${gpu_ids[$gpu_slot]}")
    job_seed_strings+=("$seed_string")
    total_seed_runs=$((total_seed_runs + seed_count))
  done
done

echo "Multi-market / multi-K / multi-seed end-to-end sweep"
echo "repo_root=$REPO_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "gpu_ids=$GPU_IDS"
echo "markets=$MARKETS"
echo "k_values=$K_VALUES"
active_worker_count=${#gpu_ids[@]}
if (( active_worker_count > ${#job_markets[@]} )); then
  active_worker_count=${#job_markets[@]}
fi
echo "gpu_workers=$active_worker_count configurations=${#job_markets[@]} total_seed_runs=$total_seed_runs"

for job_index in "${!job_markets[@]}"; do
  echo "job=$job_index gpu=${job_gpu_ids[$job_index]} market=${job_markets[$job_index]} k=${job_k_values[$job_index]} seeds=${job_seed_strings[$job_index]}"
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

for gpu_slot in "${!gpu_ids[@]}"; do
  has_jobs=0
  for job_index in "${!job_markets[@]}"; do
    if (( job_gpu_slots[job_index] == gpu_slot )); then
      has_jobs=1
      break
    fi
  done
  (( has_jobs == 1 )) || continue

  gpu="${gpu_ids[$gpu_slot]}"
  worker_log="$launcher_log_dir/gpu_${gpu}.log"
  worker_label="gpu_${gpu}"
  worker_args=(
    "$gpu"
    "$TRAINING_SCRIPT"
    "$RUN_PREFIX"
    "$PYTHON_BIN"
    "$OUTPUT_ROOT"
    "$NAS_SEEDS"
    "$SH_SEEDS"
  )

  for job_index in "${!job_markets[@]}"; do
    (( job_gpu_slots[job_index] == gpu_slot )) || continue
    worker_args+=(
      "${job_markets[$job_index]}"
      "${job_k_values[$job_index]}"
      "${job_seed_strings[$job_index]}"
    )
  done

  setsid bash "$WORKER_SCRIPT" "${worker_args[@]}" \
    > >(tee "$worker_log") 2>&1 &

  pids+=("$!")
  worker_labels+=("$worker_label")
done

failures=()
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failures+=("${worker_labels[$index]}")
  fi
done

trap - INT TERM

if (( ${#failures[@]} > 0 )); then
  echo "Failed GPU workers: ${failures[*]}" >&2
  exit 1
fi

echo "All configurations completed successfully."
echo "Outputs: $output_root_abs"
