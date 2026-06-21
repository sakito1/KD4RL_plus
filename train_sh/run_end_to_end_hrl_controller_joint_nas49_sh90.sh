#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/end_to_end_hrl_controller_joint_nas49_sh90}"
RUN_NAME="${RUN_NAME:-lookback60_hold30_e2e_hrl_controller_joint_nas49_sh90}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

NAS_SEEDS="${NAS_SEEDS:-49}"
SH_SEEDS="${SH_SEEDS:-90}"

OUTER_WINDOW="${OUTER_WINDOW:-60}"
MIN_HOLD="${MIN_HOLD:-30}"
MAX_HOLD="${MAX_HOLD:-30}"
TRAIN_EPISODES_PER_EPOCH="${TRAIN_EPISODES_PER_EPOCH:-5}"
TRAIN_START_STRIDE_DAYS="${TRAIN_START_STRIDE_DAYS:-1}"

WARMUP_OUTER_EPOCHS="${WARMUP_OUTER_EPOCHS:-2}"
WARMUP_INNER_EPOCHS="${WARMUP_INNER_EPOCHS:-2}"
INNER_SEGMENTS_PER_EPISODE="${INNER_SEGMENTS_PER_EPISODE:-20}"
INNER_EPISODE_LEN="${INNER_EPISODE_LEN:-$((MAX_HOLD * INNER_SEGMENTS_PER_EPISODE))}"
INNER_START_STRIDE_DAYS="${INNER_START_STRIDE_DAYS:-1}"
INNER_TRAIN_EPISODES_PER_EPOCH="${INNER_TRAIN_EPISODES_PER_EPOCH:-30}"
INNER_EPISODE_BATCH_SIZE="${INNER_EPISODE_BATCH_SIZE:-12}"
INNER_EPISODE_PARALLEL_WORKERS="${INNER_EPISODE_PARALLEL_WORKERS:-12}"
INNER_ROLLOUT_UPDATE_STEPS="${INNER_ROLLOUT_UPDATE_STEPS:-$INNER_EPISODE_LEN}"
INNER_PPO_EPOCHS="${INNER_PPO_EPOCHS:-1}"

CONTROLLER_EPOCHS="${CONTROLLER_EPOCHS:-3}"
CONTROLLER_JOINT_EPOCHS="${CONTROLLER_JOINT_EPOCHS:-1}"
CONTROLLER_SUP_PRETRAIN_EPOCHS="${CONTROLLER_SUP_PRETRAIN_EPOCHS:-1}"
CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN="${CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN:-240}"
CONTROLLER_AUX_PRETRAIN_OFFPOLICY="${CONTROLLER_AUX_PRETRAIN_OFFPOLICY:-1}"
CONTROLLER_AUX_REPLAY_EPOCHS="${CONTROLLER_AUX_REPLAY_EPOCHS:-3}"
CONTROLLER_ROLLOUT_LEN="${CONTROLLER_ROLLOUT_LEN:-600}"
CONTROLLER_WINDOWS_PER_EPOCH="${CONTROLLER_WINDOWS_PER_EPOCH:-30}"
CONTROLLER_FIXED_POOL_LIMIT="${CONTROLLER_FIXED_POOL_LIMIT:-12}"
CONTROLLER_SKIP_VAL="${CONTROLLER_SKIP_VAL:-0}"
CONTROLLER_PG_LOGPROB_REDUCTION="${CONTROLLER_PG_LOGPROB_REDUCTION:-sum}"
CONTROLLER_EPISODE_BATCH_SIZE="${CONTROLLER_EPISODE_BATCH_SIZE:-12}"
CONTROLLER_EPISODE_PARALLEL_WORKERS="${CONTROLLER_EPISODE_PARALLEL_WORKERS:-12}"
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
CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE="${CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE:-weighted_bce}"
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

JOINT_EPOCHS="${JOINT_EPOCHS:-1}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.001}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
OUTER_PRED_COEF="${OUTER_PRED_COEF:-0.1}"
INNER_PRED_COEF="${INNER_PRED_COEF:-0.05}"
INNER_PRED_TARGET_SCALE="${INNER_PRED_TARGET_SCALE:-10}"
TEST_SKIP_FIXED_SCENARIOS="${TEST_SKIP_FIXED_SCENARIOS:-0}"
TEST_MAX_DAYS="${TEST_MAX_DAYS:-0}"

case "$OUTPUT_ROOT" in
  results/end|results/end/*|results/hrl_lookback60_hold30_inner_noaux_retrain|results/hrl_controller_daily_aux_pg)
    echo "Refusing to write end-to-end training outputs into protected result root: $OUTPUT_ROOT" >&2
    exit 1
    ;;
esac

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

CONTROLLER_SKIP_VAL_ARGS=()
if [[ "$CONTROLLER_SKIP_VAL" == "1" ]]; then
  CONTROLLER_SKIP_VAL_ARGS=(--controller_skip_val)
fi

CONTROLLER_VALUE_NORMALIZE_ARGS=()
if [[ "$CONTROLLER_VALUE_NORMALIZE_ADVANTAGE" != "1" ]]; then
  CONTROLLER_VALUE_NORMALIZE_ARGS=(--no_controller_value_normalize_advantage)
fi

CONTROLLER_AUX_PRETRAIN_OFFPOLICY_ARGS=()
if [[ "$CONTROLLER_AUX_PRETRAIN_OFFPOLICY" == "1" ]]; then
  CONTROLLER_AUX_PRETRAIN_OFFPOLICY_ARGS=(--controller_aux_pretrain_offpolicy)
fi

CONTROLLER_COMPUTE_SWITCH_ADVANTAGE_ARGS=()
if [[ "$CONTROLLER_COMPUTE_SWITCH_ADVANTAGE" == "1" ]]; then
  CONTROLLER_COMPUTE_SWITCH_ADVANTAGE_ARGS=(--controller_compute_switch_advantage)
fi

CONTROLLER_SWITCH_ADV_LOGIT_DETACH_ARGS=()
if [[ "$CONTROLLER_SWITCH_ADV_LOGIT_DETACH" == "1" ]]; then
  CONTROLLER_SWITCH_ADV_LOGIT_DETACH_ARGS=(--controller_switch_adv_logit_detach)
fi

CONTROLLER_LOCAL_ADV_BALANCE_ARGS=()
if [[ "$CONTROLLER_LOCAL_ADV_BALANCE_CLASSES" == "1" ]]; then
  CONTROLLER_LOCAL_ADV_BALANCE_ARGS=(--controller_local_adv_balance_classes)
fi

CONTROLLER_EVAL_DIAGNOSTICS_ARGS=()
if [[ "$CONTROLLER_EVAL_DIAGNOSTICS" == "1" ]]; then
  CONTROLLER_EVAL_DIAGNOSTICS_ARGS=(--controller_eval_diagnostics --controller_eval_diag_thresholds $CONTROLLER_EVAL_DIAG_THRESHOLDS)
fi

TEST_SKIP_FIXED_SCENARIOS_ARGS=()
if [[ "$TEST_SKIP_FIXED_SCENARIOS" == "1" ]]; then
  TEST_SKIP_FIXED_SCENARIOS_ARGS=(--test_skip_fixed_scenarios)
fi

echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SH seeds: $SH_SEEDS"
echo "NAS seeds: $NAS_SEEDS"
echo "Schedule: HRL warmup -> fixed HRL joint -> controller PG -> controller-active HRL joint"
echo "Final controller joint epochs: $CONTROLLER_JOINT_EPOCHS"

run_market() {
  local market="$1"
  local seeds="$2"
  local log_file="$OUTPUT_ROOT/logs/${RUN_NAME}_${market}.log"

  if [[ -z "${seeds// }" ]]; then
    echo "Skipping ${market}: empty seed list"
    return 0
  fi

  echo
  echo "===== End-to-end HRL/controller joint run: market=${market}, seeds=${seeds} ====="

  "$PYTHON_BIN" -u run_hrl_training.py \
    --markets "$market" \
    --seeds $seeds \
    --output_root "$OUTPUT_ROOT" \
    --run_name "$RUN_NAME" \
    --device cuda \
    --outer_window "$OUTER_WINDOW" \
    --min_hold "$MIN_HOLD" \
    --max_hold "$MAX_HOLD" \
    --train_episodes_per_epoch "$TRAIN_EPISODES_PER_EPOCH" \
    --train_start_stride_days "$TRAIN_START_STRIDE_DAYS" \
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
    --controller_epochs "$CONTROLLER_EPOCHS" \
    --controller_sup_coef 0.0 \
    --controller_sup_pretrain_epochs "$CONTROLLER_SUP_PRETRAIN_EPOCHS" \
    --controller_sup_pretrain_rollout_len "$CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN" \
    "${CONTROLLER_AUX_PRETRAIN_OFFPOLICY_ARGS[@]}" \
    --controller_aux_replay_epochs "$CONTROLLER_AUX_REPLAY_EPOCHS" \
    --controller_val_interval_epochs 1 \
    "${CONTROLLER_SKIP_VAL_ARGS[@]}" \
    --controller_rollout_len "$CONTROLLER_ROLLOUT_LEN" \
    --controller_windows_per_epoch "$CONTROLLER_WINDOWS_PER_EPOCH" \
    --controller_fixed_pool_limit "$CONTROLLER_FIXED_POOL_LIMIT" \
    --controller_pg_batch_windows "$CONTROLLER_EPISODE_BATCH_SIZE" \
    --controller_pg_logprob_reduction "$CONTROLLER_PG_LOGPROB_REDUCTION" \
    --controller_train_fixed_episodes \
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
    --controller_mdd_coef 0.0 \
    --controller_count_min 0 \
    --controller_count_max 0 \
    --controller_max_switches "$CONTROLLER_MAX_SWITCHES" \
    --controller_max_switch_penalty_coef "$CONTROLLER_MAX_SWITCH_PENALTY_COEF" \
    --controller_expected_switch_penalty_coef "$CONTROLLER_EXPECTED_SWITCH_PENALTY_COEF" \
    --controller_overflow_action_penalty_coef "$CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF" \
    --controller_value_coef "$CONTROLLER_VALUE_COEF" \
    "${CONTROLLER_VALUE_NORMALIZE_ARGS[@]}" \
    --controller_switch_coef 0.0 \
    --controller_turnover_coef 0.0 \
    --controller_entropy_coef "$CONTROLLER_ENTROPY_COEF" \
    --controller_aux_return_coef "$CONTROLLER_AUX_RETURN_COEF" \
    --controller_aux_mdd_coef "$CONTROLLER_AUX_MDD_COEF" \
    --controller_aux_switch_adv_coef "$CONTROLLER_AUX_SWITCH_ADV_COEF" \
    --controller_aux_switch_adv_loss_type "$CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE" \
    --controller_switch_adv_logit_coef "$CONTROLLER_SWITCH_ADV_LOGIT_COEF" \
    --controller_switch_adv_logit_scale "$CONTROLLER_SWITCH_ADV_LOGIT_SCALE" \
    "${CONTROLLER_SWITCH_ADV_LOGIT_DETACH_ARGS[@]}" \
    "${CONTROLLER_COMPUTE_SWITCH_ADVANTAGE_ARGS[@]}" \
    --controller_aux_return_target_scale "$CONTROLLER_AUX_RETURN_TARGET_SCALE" \
    --controller_aux_mdd_target_scale "$CONTROLLER_AUX_MDD_TARGET_SCALE" \
    --controller_aux_switch_adv_target_scale "$CONTROLLER_AUX_SWITCH_ADV_TARGET_SCALE" \
    --controller_local_adv_coef "$CONTROLLER_LOCAL_ADV_COEF" \
    --controller_local_adv_scale "$CONTROLLER_LOCAL_ADV_SCALE" \
    --controller_local_adv_clip "$CONTROLLER_LOCAL_ADV_CLIP" \
    --controller_local_adv_margin "$CONTROLLER_LOCAL_ADV_MARGIN" \
    --controller_local_adv_loss_type "$CONTROLLER_LOCAL_ADV_LOSS_TYPE" \
    "${CONTROLLER_LOCAL_ADV_BALANCE_ARGS[@]}" \
    --controller_selection_metric return \
    --controller_no_hold_constraints \
    --controller_decision_mode "$CONTROLLER_DECISION_MODE" \
    --controller_eval_decision_mode "$CONTROLLER_EVAL_DECISION_MODE" \
    --controller_decision_stride_schedule $CONTROLLER_DECISION_STRIDE_SCHEDULE \
    --controller_eval_decision_stride "$CONTROLLER_EVAL_DECISION_STRIDE" \
    --controller_eval_switch_threshold "$CONTROLLER_EVAL_SWITCH_THRESHOLD" \
    "${CONTROLLER_EVAL_DIAGNOSTICS_ARGS[@]}" \
    --joint_epochs "$JOINT_EPOCHS" \
    --joint_single_full_episode \
    --joint_lr_mult "$JOINT_LR_MULT" \
    --controller_joint_epochs "$CONTROLLER_JOINT_EPOCHS" \
    --end_to_end_controller_joint \
    --ppo_epochs "$PPO_EPOCHS" \
    --outer_pred_coef "$OUTER_PRED_COEF" \
    --inner_pred_coef "$INNER_PRED_COEF" \
    --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
    --model_selection_metric sharpe \
    --inner_selection_metric return \
    --train_monitor \
    "${TEST_SKIP_FIXED_SCENARIOS_ARGS[@]}" \
    --test_max_days "$TEST_MAX_DAYS" \
    --heartbeat_seconds "$HEARTBEAT_SECONDS" \
    --continue_on_error \
    2>&1 | tee "$log_file"
}

run_market sh "$SH_SEEDS"
run_market nas "$NAS_SEEDS"

echo "End-to-end HRL/controller joint runs finished: $OUTPUT_ROOT/$RUN_NAME"
