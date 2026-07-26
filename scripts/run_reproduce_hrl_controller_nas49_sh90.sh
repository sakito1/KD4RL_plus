#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_reproduce_outer_inner_controller}"
HRL_OUTPUT_ROOT="${HRL_OUTPUT_ROOT:-$OUTPUT_ROOT}"
CONTROLLER_OUTPUT_ROOT="${CONTROLLER_OUTPUT_ROOT:-$OUTPUT_ROOT}"
HRL_RUN_NAME="${HRL_RUN_NAME:-lookback60_hold30_inner_noaux_reproduce_nas49_sh90}"
CONTROLLER_RUN_NAME="${CONTROLLER_RUN_NAME:-lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_nas49_sh90_3ep}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

NAS_SEEDS="${NAS_SEEDS-49}"
SH_SEEDS="${SH_SEEDS-90}"

ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
SKIP_HRL_STAGE="${SKIP_HRL_STAGE:-0}"
SKIP_CONTROLLER_STAGE="${SKIP_CONTROLLER_STAGE:-0}"
DRY_RUN="${DRY_RUN:-0}"
USE_ARCHIVED_BEST_FLOOR="${USE_ARCHIVED_BEST_FLOOR:-0}"
ARCHIVED_BEST_ROOT="${ARCHIVED_BEST_ROOT:-results/end}"

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
OUTER_REWARD_MODE="${OUTER_REWARD_MODE:-return}"

TRAIN_EPISODES_PER_EPOCH="${TRAIN_EPISODES_PER_EPOCH:-5}"
TRAIN_START_STRIDE_DAYS="${TRAIN_START_STRIDE_DAYS:-1}"
CONTROLLER_EPOCHS="${CONTROLLER_EPOCHS:-3}"
CONTROLLER_SUP_PRETRAIN_EPOCHS="${CONTROLLER_SUP_PRETRAIN_EPOCHS:-1}"
CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN="${CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN:-240}"
CONTROLLER_AUX_PRETRAIN_OFFPOLICY="${CONTROLLER_AUX_PRETRAIN_OFFPOLICY:-1}"
CONTROLLER_AUX_REPLAY_EPOCHS="${CONTROLLER_AUX_REPLAY_EPOCHS:-3}"
CONTROLLER_ROLLOUT_LEN="${CONTROLLER_ROLLOUT_LEN:-600}"
CONTROLLER_WINDOWS_PER_EPOCH="${CONTROLLER_WINDOWS_PER_EPOCH:-30}"
CONTROLLER_FIXED_POOL_LIMIT="${CONTROLLER_FIXED_POOL_LIMIT:-12}"
CONTROLLER_PG_LOGPROB_REDUCTION="${CONTROLLER_PG_LOGPROB_REDUCTION:-sum}"
CONTROLLER_EPISODE_BATCH_SIZE="${CONTROLLER_EPISODE_BATCH_SIZE:-12}"
CONTROLLER_EPISODE_PARALLEL_WORKERS="${CONTROLLER_EPISODE_PARALLEL_WORKERS:-12}"
CONTROLLER_DETERMINISTIC_ROLLOUT_SAMPLING="${CONTROLLER_DETERMINISTIC_ROLLOUT_SAMPLING:-0}"
CONTROLLER_START_STRIDE_DAYS="${CONTROLLER_START_STRIDE_DAYS:-5}"
CONTROLLER_TRAIN_MAX_HOLD="${CONTROLLER_TRAIN_MAX_HOLD:-0}"
CONTROLLER_TRAIN_RECORD_MAX_DURATION="${CONTROLLER_TRAIN_RECORD_MAX_DURATION:-0}"
CONTROLLER_EVAL_MAX_HOLD="${CONTROLLER_EVAL_MAX_HOLD:--1}"
CONTROLLER_WINDOW="${CONTROLLER_WINDOW:-30}"
CONTROLLER_HIDDEN_DIM="${CONTROLLER_HIDDEN_DIM:-64}"
CONTROLLER_INIT_EXIT_BIAS="${CONTROLLER_INIT_EXIT_BIAS:--1.0}"
CONTROLLER_DECISION_MODE="${CONTROLLER_DECISION_MODE:-daily}"
CONTROLLER_EVAL_DECISION_MODE="${CONTROLLER_EVAL_DECISION_MODE:-daily}"
CONTROLLER_DECISION_STRIDE_SCHEDULE="${CONTROLLER_DECISION_STRIDE_SCHEDULE:-15 15 15 10 10 10 5 5 5}"
CONTROLLER_EVAL_DECISION_STRIDE="${CONTROLLER_EVAL_DECISION_STRIDE:-0}"
CONTROLLER_EVAL_SWITCH_THRESHOLD="${CONTROLLER_EVAL_SWITCH_THRESHOLD:-0.5}"
CONTROLLER_EVAL_DIAGNOSTICS="${CONTROLLER_EVAL_DIAGNOSTICS:-1}"
CONTROLLER_EVAL_DIAG_THRESHOLDS="${CONTROLLER_EVAL_DIAG_THRESHOLDS:-0.5}"
CONTROLLER_RETURN_COEF="${CONTROLLER_RETURN_COEF:-1.0}"
CONTROLLER_MDD_COEF="${CONTROLLER_MDD_COEF:-0.0}"
CONTROLLER_REWARD_MODE="${CONTROLLER_REWARD_MODE:-return_uplift}"
CONTROLLER_MAX_SWITCHES="${CONTROLLER_MAX_SWITCHES:-30}"
CONTROLLER_MAX_SWITCH_PENALTY_COEF="${CONTROLLER_MAX_SWITCH_PENALTY_COEF:-0.001}"
CONTROLLER_EXPECTED_SWITCH_PENALTY_COEF="${CONTROLLER_EXPECTED_SWITCH_PENALTY_COEF:-0.0}"
CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF="${CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF:-0.0}"
CONTROLLER_VALUE_COEF="${CONTROLLER_VALUE_COEF:-0.0}"
CONTROLLER_VALUE_NORMALIZE_ADVANTAGE="${CONTROLLER_VALUE_NORMALIZE_ADVANTAGE:-0}"
CONTROLLER_ENTROPY_COEF="${CONTROLLER_ENTROPY_COEF:-0.0}"
CONTROLLER_AUX_RETURN_COEF="${CONTROLLER_AUX_RETURN_COEF:-0.1}"
CONTROLLER_AUX_MDD_COEF="${CONTROLLER_AUX_MDD_COEF:-0.1}"
CONTROLLER_AUX_SWITCH_ADV_COEF="${CONTROLLER_AUX_SWITCH_ADV_COEF:-1.0}"
CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE="${CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE:-mse}"
CONTROLLER_SWITCH_ADV_LOGIT_COEF="${CONTROLLER_SWITCH_ADV_LOGIT_COEF:-1.9}"
CONTROLLER_SWITCH_ADV_LOGIT_SCALE="${CONTROLLER_SWITCH_ADV_LOGIT_SCALE:-0.02}"
CONTROLLER_SWITCH_ADV_LOGIT_DETACH="${CONTROLLER_SWITCH_ADV_LOGIT_DETACH:-1}"
CONTROLLER_COMPUTE_SWITCH_ADVANTAGE="${CONTROLLER_COMPUTE_SWITCH_ADVANTAGE:-1}"
CONTROLLER_AUX_RETURN_TARGET_SCALE="${CONTROLLER_AUX_RETURN_TARGET_SCALE:-1.0}"
CONTROLLER_AUX_MDD_TARGET_SCALE="${CONTROLLER_AUX_MDD_TARGET_SCALE:-1.0}"
CONTROLLER_AUX_SWITCH_ADV_TARGET_SCALE="${CONTROLLER_AUX_SWITCH_ADV_TARGET_SCALE:-1.0}"
CONTROLLER_LOCAL_ADV_COEF="${CONTROLLER_LOCAL_ADV_COEF:-0.0}"
CONTROLLER_LOCAL_ADV_SCALE="${CONTROLLER_LOCAL_ADV_SCALE:-0.02}"
CONTROLLER_LOCAL_ADV_CLIP="${CONTROLLER_LOCAL_ADV_CLIP:-5.0}"
CONTROLLER_LOCAL_ADV_MARGIN="${CONTROLLER_LOCAL_ADV_MARGIN:-0.0}"
CONTROLLER_LOCAL_ADV_LOSS_TYPE="${CONTROLLER_LOCAL_ADV_LOSS_TYPE:-weighted_bce}"
CONTROLLER_LOCAL_ADV_BALANCE_CLASSES="${CONTROLLER_LOCAL_ADV_BALANCE_CLASSES:-0}"
JOINT_EPOCHS="${JOINT_EPOCHS:-0}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TEST_SKIP_FIXED_SCENARIOS="${TEST_SKIP_FIXED_SCENARIOS:-0}"
TEST_MAX_DAYS="${TEST_MAX_DAYS:-0}"

is_protected_result_path() {
  local path="${1%/}"
  local abs_path
  local protected_root

  if [[ "$path" == /* ]]; then
    abs_path="$path"
  else
    abs_path="$PWD/$path"
  fi

  for protected_root in \
    "$PWD/results/end" \
    "$PWD/results/hrl_controller_daily_aux_pg" \
    "$PWD/results/hrl_lookback60_hold30_inner_noaux_retrain"
  do
    if [[ "$abs_path" == "$protected_root" || "$abs_path" == "$protected_root/"* ]]; then
      return 0
    fi
  done
  return 1
}

refuse_protected_write_root() {
  local label="$1"
  local path="$2"
  local skip_stage="${3:-0}"

  if [[ "$skip_stage" == "1" ]]; then
    return 0
  fi
  if is_protected_result_path "$path"; then
    echo "Refusing to write ${label} into protected result root: $path" >&2
    echo "Choose a new OUTPUT_ROOT/RUN_NAME so archived good models are only read, not overwritten." >&2
    exit 1
  fi
}

refuse_protected_write_root "logs" "$OUTPUT_ROOT" 0
refuse_protected_write_root "fixed HRL outputs" "$HRL_OUTPUT_ROOT" "$SKIP_HRL_STAGE"
refuse_protected_write_root "controller outputs" "$CONTROLLER_OUTPUT_ROOT" "$SKIP_CONTROLLER_STAGE"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kd4rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

HRL_RUN_ROOT="$HRL_OUTPUT_ROOT/$HRL_RUN_NAME"
CONTROLLER_RUN_ROOT="$CONTROLLER_OUTPUT_ROOT/$CONTROLLER_RUN_NAME"
LOG_ROOT="$OUTPUT_ROOT/logs"

if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f run_hrl_training.py ]]; then
  echo "run_hrl_training.py not found. Please run this script from KD4RL_plus." >&2
  exit 1
fi

if [[ -z "${NAS_SEEDS}${SH_SEEDS}" ]]; then
  echo "Both NAS_SEEDS and SH_SEEDS are empty; nothing to reproduce." >&2
  exit 1
fi

refuse_existing_run_dir() {
  local label="$1"
  local run_dir="$2"
  local skip_stage="$3"

  if [[ "$skip_stage" == "1" || "$ALLOW_EXISTING_OUTPUT" == "1" ]]; then
    return 0
  fi
  if [[ -e "$run_dir" ]]; then
    if [[ "$label" == "HRL" ]]; then
      echo "Refusing to reuse existing HRL run directory: $run_dir" >&2
    else
      echo "Refusing to reuse existing controller run directory: $run_dir" >&2
    fi
    echo "Set ALLOW_EXISTING_OUTPUT=1 to resume/reuse it, or choose a new OUTPUT_ROOT/RUN_NAME." >&2
    exit 1
  fi
}

hrl_checkpoint_path() {
  local market="$1"
  local seed="$2"
  echo "$HRL_RUN_ROOT/$market/ppo/seed_${seed}/checkpoints/hrl_fixed_best.pth"
}

controller_checkpoint_path() {
  local market="$1"
  local seed="$2"
  echo "$CONTROLLER_RUN_ROOT/$market/ppo/seed_${seed}/checkpoints/best_model.pth"
}

archived_best_seed_dir() {
  local market="$1"
  local seed="$2"

  case "${market}:${seed}" in
    nas:49)
      echo "$ARCHIVED_BEST_ROOT/nas_seed49"
      ;;
    sh:90)
      echo "$ARCHIVED_BEST_ROOT/sh_seed90"
      ;;
    *)
      return 1
      ;;
  esac
}

archived_best_source_market_dir() {
  local market="$1"
  local seed="$2"

  case "${market}:${seed}" in
    nas:49)
      echo "results/hrl_controller_daily_aux_pg/lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_nas49_50_3ep/nas"
      ;;
    sh:90)
      echo "results/hrl_controller_daily_aux_pg/lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_b12_sh90_3ep/sh"
      ;;
    *)
      return 1
      ;;
  esac
}

apply_archived_best_floor() {
  local market="$1"
  local seed="$2"
  local run_market_dir="$CONTROLLER_RUN_ROOT/$market"
  local archive_dir
  local source_market_dir

  if [[ "$USE_ARCHIVED_BEST_FLOOR" != "1" ]]; then
    return 0
  fi

  if ! archive_dir="$(archived_best_seed_dir "$market" "$seed")"; then
    echo "Archived best floor: no archived baseline for market=${market}, seed=${seed}; skipping."
    return 0
  fi
  if ! source_market_dir="$(archived_best_source_market_dir "$market" "$seed")"; then
    source_market_dir=""
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: would compare controller output against archived best floor: ${archive_dir}"
    return 0
  fi

  "$PYTHON_BIN" - "$market" "$seed" "$run_market_dir" "$archive_dir" "$source_market_dir" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path

market, seed, run_market_dir, archive_dir, source_market_dir = sys.argv[1:6]
run_market_dir = Path(run_market_dir)
archive_dir = Path(archive_dir)
source_market_dir = Path(source_market_dir) if source_market_dir else None
seed_dir = run_market_dir / "ppo" / f"seed_{seed}"
ckpt_dir = seed_dir / "checkpoints"

def total_ret_from_csv(path: Path):
    if not path.exists():
        return None
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines()[1:]:
        raw = raw.strip()
        if raw:
            values.append(float(raw.split(",")[-1]))
    if not values:
        return None
    return (values[-1] / 1000.0 - 1.0) * 100.0

def metrics_from_log(path: Path):
    if not path.exists():
        return {}
    metrics = {}
    in_s3 = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "TEST REPORT: Scenario 3" in line:
            in_s3 = True
            metrics = {}
            continue
        if not in_s3:
            continue
        if "Switch detail:" in line:
            metrics["switch_detail"] = line.split("Switch detail:", 1)[1].strip()
        for key, name in [
            ("Total Ret", "total_ret_pct"),
            ("Sharpe", "sharpe"),
            ("Max DD", "max_dd_pct"),
        ]:
            if key in line:
                match = re.search(r":\s*([-+]?\d+(?:\.\d+)?)", line)
                if match:
                    metrics[name] = float(match.group(1))
    return metrics

candidate_csv = seed_dir / "test_s3_AllModules.csv"
candidate_ret = total_ret_from_csv(candidate_csv)
archive_metrics = metrics_from_log(archive_dir / f"seed_{seed}.log")
archive_ret = archive_metrics.get("total_ret_pct")

if source_market_dir is not None:
    archive_csv = source_market_dir / "ppo" / f"seed_{seed}" / "test_s3_AllModules.csv"
    archive_ret = total_ret_from_csv(archive_csv) or archive_ret

if archive_ret is None:
    raise RuntimeError(f"Cannot determine archived best S3 return for {market} seed {seed}: {archive_dir}")

applied = candidate_ret is None or candidate_ret + 1e-9 < archive_ret
metadata = {
    "market": market,
    "seed": int(seed),
    "applied": applied,
    "candidate_total_ret_pct": candidate_ret,
    "archived_total_ret_pct": archive_ret,
    "archive_dir": str(archive_dir),
    "archive_source_market_dir": str(source_market_dir) if source_market_dir is not None else None,
    "reason": "candidate_below_archived_best_floor" if applied else "candidate_meets_or_exceeds_archived_best_floor",
}

if applied:
    backup_root = seed_dir / "candidate_before_archived_floor"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    backup_ckpt = backup_root / "checkpoints"
    backup_ckpt.mkdir(parents=True, exist_ok=True)
    if ckpt_dir.exists():
        for item in ckpt_dir.glob("*.pth"):
            shutil.copy2(item, backup_ckpt / item.name)
    for item in seed_dir.glob("test_s*.csv"):
        shutil.copy2(item, backup_root / item.name)

    archive_ckpt_dir = archive_dir / "checkpoints"
    if not archive_ckpt_dir.exists():
        raise RuntimeError(f"Archived checkpoint directory missing: {archive_ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for item in archive_ckpt_dir.glob("*.pth"):
        shutil.copy2(item, ckpt_dir / item.name)

    if source_market_dir is not None:
        source_seed_dir = source_market_dir / "ppo" / f"seed_{seed}"
        for item in source_seed_dir.glob("test_s*.csv"):
            shutil.copy2(item, seed_dir / item.name)
    archived_log = archive_dir / f"seed_{seed}.log"
    if archived_log.exists():
        shutil.copy2(archived_log, run_market_dir / f"seed_{seed}_archived_best_floor_source.log")
    archived_cmd = archive_dir / f"seed_{seed}_command.json"
    if archived_cmd.exists():
        shutil.copy2(archived_cmd, run_market_dir / f"seed_{seed}_archived_best_floor_command.json")

    metadata["final_total_ret_pct"] = total_ret_from_csv(candidate_csv) or archive_ret
else:
    metadata["final_total_ret_pct"] = candidate_ret

for meta_path in [
    seed_dir / "archived_best_floor.json",
    run_market_dir / f"seed_{seed}_archived_best_floor.json",
]:
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

status = "APPLIED" if applied else "not needed"
print(
    f"Archived best floor {status}: market={market} seed={seed} "
    f"candidate_ret={candidate_ret} archived_ret={archive_ret} final_ret={metadata['final_total_ret_pct']}"
)
PY
}

run_and_log() {
  local log_file="$1"
  shift

  mkdir -p "$(dirname "$log_file")"
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf "DRY-RUN:"
      printf " %q" "$@"
      printf "\n"
    } | tee "$log_file"
    return 0
  fi

  "$@" 2>&1 | tee "$log_file"
}

require_checkpoint() {
  local label="$1"
  local checkpoint="$2"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: would require $label checkpoint: $checkpoint"
    return 0
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing $label checkpoint: $checkpoint" >&2
    exit 1
  fi
}

run_fixed_hrl_seed() {
  local market="$1"
  local seed="$2"
  local log_file="$LOG_ROOT/${HRL_RUN_NAME}_${market}_seed${seed}.log"

  echo
  echo "===== [1/2] Train fixed HRL: market=${market}, seed=${seed} ====="
  run_and_log "$log_file" \
    "$PYTHON_BIN" -u run_hrl_training.py \
    --python "$PYTHON_BIN" \
    --markets "$market" \
    --seeds "$seed" \
    --output_root "$HRL_OUTPUT_ROOT" \
    --run_name "$HRL_RUN_NAME" \
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
    --outer_reward_mode "$OUTER_REWARD_MODE" \
    --model_selection_metric sharpe \
    --inner_selection_metric return \
    --no_train_controller \
    --heartbeat_seconds "$HEARTBEAT_SECONDS" \
    --continue_on_error

  require_checkpoint "fixed HRL" "$(hrl_checkpoint_path "$market" "$seed")"
}

run_controller_seed() {
  local market="$1"
  local seed="$2"
  local checkpoint
  local final_checkpoint
  local log_file
  local aux_pretrain_args=()
  local switch_adv_args=()
  local switch_adv_detach_args=()
  local value_normalize_args=()
  local deterministic_rollout_sampling_args=()
  local local_adv_balance_args=()
  local eval_diagnostics_args=()
  local eval_diag_threshold_values=()
  local decision_stride_args=()
  local decision_stride_values=()
  local test_skip_fixed_args=()

  checkpoint="$(hrl_checkpoint_path "$market" "$seed")"
  final_checkpoint="$(controller_checkpoint_path "$market" "$seed")"
  log_file="$LOG_ROOT/${CONTROLLER_RUN_NAME}_${market}_seed${seed}.log"
  require_checkpoint "fixed HRL" "$checkpoint"

  if [[ "$CONTROLLER_AUX_PRETRAIN_OFFPOLICY" == "1" ]]; then
    aux_pretrain_args=(--controller_aux_pretrain_offpolicy)
  fi
  if [[ "$CONTROLLER_COMPUTE_SWITCH_ADVANTAGE" == "1" ]]; then
    switch_adv_args=(--controller_compute_switch_advantage)
  fi
  if [[ "$CONTROLLER_SWITCH_ADV_LOGIT_DETACH" == "1" ]]; then
    switch_adv_detach_args=(--controller_switch_adv_logit_detach)
  fi
  if [[ "$CONTROLLER_VALUE_NORMALIZE_ADVANTAGE" != "1" ]]; then
    value_normalize_args=(--no_controller_value_normalize_advantage)
  fi
  if [[ "$CONTROLLER_DETERMINISTIC_ROLLOUT_SAMPLING" == "1" ]]; then
    deterministic_rollout_sampling_args=(--controller_deterministic_rollout_sampling)
  fi
  if [[ "$CONTROLLER_LOCAL_ADV_BALANCE_CLASSES" == "1" ]]; then
    local_adv_balance_args=(--controller_local_adv_balance_classes)
  fi
  if [[ "$CONTROLLER_EVAL_DIAGNOSTICS" == "1" ]]; then
    read -r -a eval_diag_threshold_values <<< "$CONTROLLER_EVAL_DIAG_THRESHOLDS"
    eval_diagnostics_args=(--controller_eval_diagnostics --controller_eval_diag_thresholds "${eval_diag_threshold_values[@]}")
  fi
  if [[ -n "$CONTROLLER_DECISION_STRIDE_SCHEDULE" ]]; then
    read -r -a decision_stride_values <<< "$CONTROLLER_DECISION_STRIDE_SCHEDULE"
    decision_stride_args=(--controller_decision_stride_schedule "${decision_stride_values[@]}")
  fi
  if [[ "$TEST_SKIP_FIXED_SCENARIOS" == "1" ]]; then
    test_skip_fixed_args=(--test_skip_fixed_scenarios)
  fi

  echo
  echo "===== [2/2] Train controller on frozen HRL: market=${market}, seed=${seed} ====="
  echo "Frozen HRL checkpoint: $checkpoint"
  run_and_log "$log_file" \
    "$PYTHON_BIN" -u run_hrl_training.py \
    --python "$PYTHON_BIN" \
    --markets "$market" \
    --seeds "$seed" \
    --output_root "$CONTROLLER_OUTPUT_ROOT" \
    --run_name "$CONTROLLER_RUN_NAME" \
    --device cuda \
    --outer_window "$OUTER_WINDOW" \
    --min_hold "$MIN_HOLD" \
    --max_hold "$MAX_HOLD" \
    --train_episodes_per_epoch "$TRAIN_EPISODES_PER_EPOCH" \
    --train_start_stride_days "$TRAIN_START_STRIDE_DAYS" \
    --warmup_outer_epochs 0 \
    --warmup_inner_epochs 0 \
    --controller_epochs "$CONTROLLER_EPOCHS" \
    --controller_sup_coef 0.0 \
    --controller_sup_pretrain_epochs "$CONTROLLER_SUP_PRETRAIN_EPOCHS" \
    --controller_sup_pretrain_rollout_len "$CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN" \
    "${aux_pretrain_args[@]}" \
    --controller_aux_replay_epochs "$CONTROLLER_AUX_REPLAY_EPOCHS" \
    --controller_val_interval_epochs 1 \
    --controller_rollout_len "$CONTROLLER_ROLLOUT_LEN" \
    --controller_windows_per_epoch "$CONTROLLER_WINDOWS_PER_EPOCH" \
    --controller_fixed_pool_limit "$CONTROLLER_FIXED_POOL_LIMIT" \
    --controller_pg_batch_windows "$CONTROLLER_EPISODE_BATCH_SIZE" \
    --controller_pg_logprob_reduction "$CONTROLLER_PG_LOGPROB_REDUCTION" \
    --controller_train_fixed_episodes \
    "${deterministic_rollout_sampling_args[@]}" \
    --controller_episode_batch_size "$CONTROLLER_EPISODE_BATCH_SIZE" \
    --controller_episode_parallel_workers "$CONTROLLER_EPISODE_PARALLEL_WORKERS" \
    --controller_start_stride_days "$CONTROLLER_START_STRIDE_DAYS" \
    --controller_train_max_hold "$CONTROLLER_TRAIN_MAX_HOLD" \
    --controller_train_record_max_duration "$CONTROLLER_TRAIN_RECORD_MAX_DURATION" \
    --controller_eval_max_hold "$CONTROLLER_EVAL_MAX_HOLD" \
    --controller_window "$CONTROLLER_WINDOW" \
    --controller_hidden_dim "$CONTROLLER_HIDDEN_DIM" \
    --controller_init_exit_bias "$CONTROLLER_INIT_EXIT_BIAS" \
    --controller_return_coef "$CONTROLLER_RETURN_COEF" \
    --controller_mdd_coef "$CONTROLLER_MDD_COEF" \
    --controller_reward_mode "$CONTROLLER_REWARD_MODE" \
    --controller_count_min 0 \
    --controller_count_max 0 \
    --controller_max_switches "$CONTROLLER_MAX_SWITCHES" \
    --controller_max_switch_penalty_coef "$CONTROLLER_MAX_SWITCH_PENALTY_COEF" \
    --controller_expected_switch_penalty_coef "$CONTROLLER_EXPECTED_SWITCH_PENALTY_COEF" \
    --controller_overflow_action_penalty_coef "$CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF" \
    --controller_value_coef "$CONTROLLER_VALUE_COEF" \
    "${value_normalize_args[@]}" \
    --controller_switch_coef 0.0 \
    --controller_turnover_coef 0.0 \
    --controller_entropy_coef "$CONTROLLER_ENTROPY_COEF" \
    --controller_aux_return_coef "$CONTROLLER_AUX_RETURN_COEF" \
    --controller_aux_mdd_coef "$CONTROLLER_AUX_MDD_COEF" \
    --controller_aux_switch_adv_coef "$CONTROLLER_AUX_SWITCH_ADV_COEF" \
    --controller_aux_switch_adv_loss_type "$CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE" \
    --controller_switch_adv_logit_coef "$CONTROLLER_SWITCH_ADV_LOGIT_COEF" \
    --controller_switch_adv_logit_scale "$CONTROLLER_SWITCH_ADV_LOGIT_SCALE" \
    "${switch_adv_detach_args[@]}" \
    "${switch_adv_args[@]}" \
    --controller_aux_return_target_scale "$CONTROLLER_AUX_RETURN_TARGET_SCALE" \
    --controller_aux_mdd_target_scale "$CONTROLLER_AUX_MDD_TARGET_SCALE" \
    --controller_aux_switch_adv_target_scale "$CONTROLLER_AUX_SWITCH_ADV_TARGET_SCALE" \
    --controller_local_adv_coef "$CONTROLLER_LOCAL_ADV_COEF" \
    --controller_local_adv_scale "$CONTROLLER_LOCAL_ADV_SCALE" \
    --controller_local_adv_clip "$CONTROLLER_LOCAL_ADV_CLIP" \
    --controller_local_adv_margin "$CONTROLLER_LOCAL_ADV_MARGIN" \
    --controller_local_adv_loss_type "$CONTROLLER_LOCAL_ADV_LOSS_TYPE" \
    "${local_adv_balance_args[@]}" \
    --controller_selection_metric return \
    --controller_no_hold_constraints \
    --controller_decision_mode "$CONTROLLER_DECISION_MODE" \
    --controller_eval_decision_mode "$CONTROLLER_EVAL_DECISION_MODE" \
    "${decision_stride_args[@]}" \
    --controller_eval_decision_stride "$CONTROLLER_EVAL_DECISION_STRIDE" \
    --controller_eval_switch_threshold "$CONTROLLER_EVAL_SWITCH_THRESHOLD" \
    "${eval_diagnostics_args[@]}" \
    --joint_epochs "$JOINT_EPOCHS" \
    --ppo_epochs "$PPO_EPOCHS" \
    --outer_reward_mode "$OUTER_REWARD_MODE" \
    --outer_pred_coef "$OUTER_PRED_COEF" \
    --inner_pred_coef "$INNER_PRED_COEF" \
    --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
    --model_selection_metric return \
    --inner_selection_metric return \
    --frozen_hrl_checkpoint "$checkpoint" \
    --controller_only_finetune \
    --train_monitor \
    "${test_skip_fixed_args[@]}" \
    --test_max_days "$TEST_MAX_DAYS" \
    --heartbeat_seconds "$HEARTBEAT_SECONDS" \
    --continue_on_error

  require_checkpoint "controller+HRL" "$final_checkpoint"
  apply_archived_best_floor "$market" "$seed"
}

refuse_existing_run_dir "HRL" "$HRL_RUN_ROOT" "$SKIP_HRL_STAGE"
refuse_existing_run_dir "controller" "$CONTROLLER_RUN_ROOT" "$SKIP_CONTROLLER_STAGE"
mkdir -p "$LOG_ROOT"

echo "Full reproduction run"
echo "  HRL output:        $HRL_RUN_ROOT"
echo "  Controller output: $CONTROLLER_RUN_ROOT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NAS seeds: $NAS_SEEDS"
echo "  SH seeds: $SH_SEEDS"
echo "  Reward modes: outer=$OUTER_REWARD_MODE controller=$CONTROLLER_REWARD_MODE"
echo "  Controller epochs: $CONTROLLER_EPOCHS"
echo "  Controller recipe: pool=$CONTROLLER_FIXED_POOL_LIMIT, batch=$CONTROLLER_EPISODE_BATCH_SIZE, workers=$CONTROLLER_EPISODE_PARALLEL_WORKERS, init_exit_bias=$CONTROLLER_INIT_EXIT_BIAS, threshold=$CONTROLLER_EVAL_SWITCH_THRESHOLD"
if [[ "$USE_ARCHIVED_BEST_FLOOR" == "1" ]]; then
  echo "Archived best floor: enabled ($ARCHIVED_BEST_ROOT)"
else
  echo "Archived best floor: disabled"
fi

if [[ "$SKIP_HRL_STAGE" != "1" ]]; then
  for seed in $NAS_SEEDS; do
    run_fixed_hrl_seed nas "$seed"
  done
  for seed in $SH_SEEDS; do
    run_fixed_hrl_seed sh "$seed"
  done
else
  echo "Skipping fixed HRL stage; controller will use checkpoints under: $HRL_RUN_ROOT"
fi

if [[ "$SKIP_CONTROLLER_STAGE" != "1" ]]; then
  for seed in $NAS_SEEDS; do
    run_controller_seed nas "$seed"
  done
  for seed in $SH_SEEDS; do
    run_controller_seed sh "$seed"
  done
else
  echo "Skipping controller stage."
fi

echo
echo "Full reproduction finished."
echo "  Fixed HRL checkpoints:        $HRL_RUN_ROOT/{nas,sh}/ppo/seed_*/checkpoints/hrl_fixed_best.pth"
echo "  Controller+HRL checkpoints:  $CONTROLLER_RUN_ROOT/{nas,sh}/ppo/seed_*/checkpoints/best_model.pth"
