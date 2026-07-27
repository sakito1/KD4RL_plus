#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_ROOT="${SOURCE_ROOT:-results/full_cmtflow_seed_sweep_4gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/full_cmtflow_test_controller_maxhold60}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0 1 2 3}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
JSON_PYTHON="${JSON_PYTHON:-$(command -v python3 || command -v python || true)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source result directory does not exist: $SOURCE_ROOT" >&2
  exit 1
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python was not found; activate the intended environment or set PYTHON_BIN." >&2
  exit 1
fi
if [[ -z "$JSON_PYTHON" ]]; then
  echo "Python is required to read seed command JSON files." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
SUMMARY_FILE="$OUTPUT_ROOT/test_results_summary.txt"
: >"$SUMMARY_FILE"

mapfile -d '' -t COMMAND_FILES < <(
  find "$SOURCE_ROOT" -type f -name 'seed_*_command.json' -print0 | sort -z
)
if [[ "${#COMMAND_FILES[@]}" -eq 0 ]]; then
  echo "No seed command JSON files found under: $SOURCE_ROOT" >&2
  exit 1
fi

read -r -a GPU_LIST <<<"$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU index." >&2
  exit 1
fi

ELIGIBLE_FILES=()
skipped=0
for command_json in "${COMMAND_FILES[@]}"; do
  market_root="$(dirname "$command_json")"
  market="$(basename "$market_root")"
  seed_name="$(basename "$command_json")"
  seed="${seed_name#seed_}"
  seed="${seed%_command.json}"
  checkpoint="$market_root/ppo/seed_${seed}/checkpoints/best_model.pth"

  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP seed=${seed} market=${market}: missing checkpoint $checkpoint"
    skipped=$((skipped + 1))
    continue
  fi
  ELIGIBLE_FILES+=("$command_json")
done

STATUS_DIR="$OUTPUT_ROOT/.controller_maxhold60_status_$$"
mkdir -p "$STATUS_DIR"
trap 'rm -rf "$STATUS_DIR"' EXIT

run_one() {
  local gpu="$1"
  local command_json="$2"
  local job_index="$3"
  local status_file="$STATUS_DIR/${job_index}.status"
  local summary_part="$STATUS_DIR/${job_index}.summary"
  local market_root
  local market
  local run_root
  local seed_name
  local seed
  local checkpoint
  local relative_run
  local test_run_root
  local log_file
  local index
  local argument
  local found_run_root
  local found_max_hold
  local found_eval_max_hold
  local command=()
  local rewritten=()

  market_root="$(dirname "$command_json")"
  market="$(basename "$market_root")"
  run_root="$(dirname "$market_root")"
  seed_name="$(basename "$command_json")"
  seed="${seed_name#seed_}"
  seed="${seed%_command.json}"
  checkpoint="$market_root/ppo/seed_${seed}/checkpoints/best_model.pth"
  command=()
  mapfile -d '' -t command < <(
    "$JSON_PYTHON" - "$command_json" <<'PY'
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
for argument in payload["command"]:
    sys.stdout.buffer.write(os.fsencode(str(argument)) + b"\0")
PY
  )
  if [[ "${#command[@]}" -eq 0 ]]; then
    echo "FAIL seed=${seed} market=${market}: invalid command JSON $command_json" >&2
    echo "fail" >"$status_file"
    return 1
  fi

  relative_run="${run_root#"$SOURCE_ROOT"/}"
  test_run_root="$OUTPUT_ROOT/$relative_run"
  log_file="$OUTPUT_ROOT/logs/${market}_seed${seed}.log"
  rewritten=("$PYTHON_BIN")
  found_run_root=0
  found_max_hold=0
  found_eval_max_hold=0

  for ((index = 1; index < ${#command[@]}; index++)); do
    argument="${command[$index]}"
    case "$argument" in
      --run_root)
        rewritten+=("--run_root" "$test_run_root")
        found_run_root=1
        index=$((index + 1))
        ;;
      --max_hold)
        rewritten+=("--max_hold" "30")
        found_max_hold=1
        index=$((index + 1))
        ;;
      --controller_eval_max_hold)
        rewritten+=("--controller_eval_max_hold" "60")
        found_eval_max_hold=1
        index=$((index + 1))
        ;;
      --test_only_checkpoint)
        index=$((index + 1))
        ;;
      --skip_test)
        ;;
      *)
        rewritten+=("$argument")
        ;;
    esac
  done

  if [[ "$found_run_root" == "0" ]]; then
    rewritten+=("--run_root" "$test_run_root")
  fi
  if [[ "$found_max_hold" == "0" ]]; then
    rewritten+=("--max_hold" "30")
  fi
  if [[ "$found_eval_max_hold" == "0" ]]; then
    rewritten+=("--controller_eval_max_hold" "60")
  fi
  rewritten+=("--test_only_checkpoint" "$checkpoint")

  echo "GPU ${gpu} RUN seed=${seed} market=${market} checkpoint=$checkpoint"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf -v command_line 'CUDA_VISIBLE_DEVICES=%q COMMAND:' "$gpu"
    for argument in "${rewritten[@]}"; do
      printf -v command_line '%s %q' "$command_line" "$argument"
    done
    echo "$command_line"
    echo "pass" >"$status_file"
    return 0
  fi

  mkdir -p "$test_run_root"
  if CUDA_VISIBLE_DEVICES="$gpu" "${rewritten[@]}" >"$log_file" 2>&1; then
    echo "GPU ${gpu} PASS seed=${seed} market=${market} log=$log_file"
    echo "pass" >"$status_file"
    {
      echo "===== seed=${seed} market=${market} ====="
      grep -E \
        "TEST REPORT|Controller eval exit_prob:|Switches[[:space:]]*:|Total Ret[[:space:]]*:|Ann Ret[[:space:]]*:|Sharpe[[:space:]]*:|Max DD[[:space:]]*:" \
        "$log_file" || echo "(test completed; inspect $log_file)"
      echo
    } >"$summary_part"
    return 0
  else
    echo "GPU ${gpu} FAIL seed=${seed} market=${market} log=$log_file" >&2
    echo "fail" >"$status_file"
    return 1
  fi
}

run_gpu_queue() {
  local gpu="$1"
  local slot="$2"
  local queue_labels=()
  local index
  local command_json
  local seed_name
  local seed

  for ((index = slot; index < ${#ELIGIBLE_FILES[@]}; index += ${#GPU_LIST[@]})); do
    command_json="${ELIGIBLE_FILES[$index]}"
    seed_name="$(basename "$command_json")"
    seed="${seed_name#seed_}"
    seed="${seed%_command.json}"
    queue_labels+=("$(basename "$(dirname "$command_json")"):${seed}")
  done
  echo "GPU ${gpu} queue: ${queue_labels[*]:-(empty)}"

  for ((index = slot; index < ${#ELIGIBLE_FILES[@]}; index += ${#GPU_LIST[@]})); do
    run_one "$gpu" "${ELIGIBLE_FILES[$index]}" "$index" || true
  done
}

pids=()
for slot in "${!GPU_LIST[@]}"; do
  run_gpu_queue "${GPU_LIST[$slot]}" "$slot" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

completed=0
failed=0
for index in "${!ELIGIBLE_FILES[@]}"; do
  if [[ -f "$STATUS_DIR/${index}.status" ]] &&
     [[ "$(<"$STATUS_DIR/${index}.status")" == "pass" ]]; then
    completed=$((completed + 1))
  else
    failed=$((failed + 1))
  fi
  if [[ -f "$STATUS_DIR/${index}.summary" ]]; then
    cat "$STATUS_DIR/${index}.summary" >>"$SUMMARY_FILE"
  fi
done

echo "SUMMARY completed=${completed} skipped=${skipped} failed=${failed}"
echo "Test summary: $SUMMARY_FILE"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
