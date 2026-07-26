#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-results/hrl_lookback60_hold30_inner_noaux_retrain/lookback60_hold30_inner_noaux_retrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_controller_daily_aux_pg}"
RUN_NAME="${RUN_NAME:-lookback60_hold30_daily_pg_trainfree_fullpg_swadvlogit19_pen1e3_aux1_pre1r3_pool12_b12_sh90_3ep}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

# NAS_SEEDS="${NAS_SEEDS-49 50}"
# SH_SEEDS="${SH_SEEDS-90 83}"
NAS_SEEDS="${NAS_SEEDS-}"
SH_SEEDS="${SH_SEEDS-90}"
OUTER_WINDOW="${OUTER_WINDOW:-60}"
MIN_HOLD="${MIN_HOLD:-30}"
MAX_HOLD="${MAX_HOLD:-30}"
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
CONTROLLER_SKIP_VAL="${CONTROLLER_SKIP_VAL:-0}"
TEST_SKIP_FIXED_SCENARIOS="${TEST_SKIP_FIXED_SCENARIOS:-0}"
TEST_MAX_DAYS="${TEST_MAX_DAYS:-0}"
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
CONTROLLER_MDD_COEF="${CONTROLLER_MDD_COEF:-0.0}"
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

CONTROLLER_ONLY="${CONTROLLER_ONLY:-1}"
JOINT_EPOCHS="${JOINT_EPOCHS:-1}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.0001}"
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

if [[ "$CONTROLLER_ONLY" == "1" ]]; then
  RUN_MODE="controller_only"
  FINETUNE_MODE_ARGS=(--controller_only_finetune)
  EFFECTIVE_JOINT_EPOCHS=0
else
  RUN_MODE="controller_first_joint"
  FINETUNE_MODE_ARGS=(--controller_first_joint_finetune)
  EFFECTIVE_JOINT_EPOCHS="$JOINT_EPOCHS"
fi

echo "Run name: $RUN_NAME"
echo "Source root: $SOURCE_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NAS seeds: $NAS_SEEDS"
echo "SH seeds: $SH_SEEDS"
echo "Training mode: mode=$RUN_MODE, joint_epochs=$EFFECTIVE_JOINT_EPOCHS, joint_lr_mult=$JOINT_LR_MULT"
if [[ "$CONTROLLER_ONLY" == "1" ]]; then
  echo "CONTROLLER-ONLY finetune enabled"
fi
echo "Controller: shared emb hidden=$CONTROLLER_HIDDEN_DIM, window=$CONTROLLER_WINDOW, init_exit_bias=$CONTROLLER_INIT_EXIT_BIAS"
echo "Controller dataset: episode_len=$CONTROLLER_ROLLOUT_LEN, offsets=$CONTROLLER_WINDOWS_PER_EPOCH, start_stride_days=$CONTROLLER_START_STRIDE_DAYS, batch=$CONTROLLER_EPISODE_BATCH_SIZE, workers=$CONTROLLER_EPISODE_PARALLEL_WORKERS"
echo "Aux warmup: epochs=$CONTROLLER_SUP_PRETRAIN_EPOCHS, offpolicy=$CONTROLLER_AUX_PRETRAIN_OFFPOLICY, replay_epochs=$CONTROLLER_AUX_REPLAY_EPOCHS, rollout_len=$CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN, return_coef=$CONTROLLER_AUX_RETURN_COEF, mdd_coef=$CONTROLLER_AUX_MDD_COEF, switch_adv_coef=$CONTROLLER_AUX_SWITCH_ADV_COEF, switch_adv_loss=$CONTROLLER_AUX_SWITCH_ADV_LOSS_TYPE, switch_adv_mining=$CONTROLLER_COMPUTE_SWITCH_ADVANTAGE, switch_adv_logit_coef=$CONTROLLER_SWITCH_ADV_LOGIT_COEF, switch_adv_logit_scale=$CONTROLLER_SWITCH_ADV_LOGIT_SCALE, switch_adv_logit_detach=$CONTROLLER_SWITCH_ADV_LOGIT_DETACH"
echo "Local advantage: coef=$CONTROLLER_LOCAL_ADV_COEF, scale=$CONTROLLER_LOCAL_ADV_SCALE, clip=$CONTROLLER_LOCAL_ADV_CLIP, margin=$CONTROLLER_LOCAL_ADV_MARGIN, loss=$CONTROLLER_LOCAL_ADV_LOSS_TYPE, balance_classes=$CONTROLLER_LOCAL_ADV_BALANCE_CLASSES"
echo "PG curriculum: decision_mode=$CONTROLLER_DECISION_MODE, eval_mode=$CONTROLLER_EVAL_DECISION_MODE, decision_stride_schedule=$CONTROLLER_DECISION_STRIDE_SCHEDULE, eval_stride=$CONTROLLER_EVAL_DECISION_STRIDE, logprob_reduction=$CONTROLLER_PG_LOGPROB_REDUCTION, value_coef=$CONTROLLER_VALUE_COEF, entropy_coef=$CONTROLLER_ENTROPY_COEF"
echo "Controller eval: eval_threshold=$CONTROLLER_EVAL_SWITCH_THRESHOLD, diagnostics=$CONTROLLER_EVAL_DIAGNOSTICS, diag_thresholds=$CONTROLLER_EVAL_DIAG_THRESHOLDS"
echo "Constraints: no min-hold, eval forced max_hold=$MAX_HOLD switch, controller_train_max_hold=$CONTROLLER_TRAIN_MAX_HOLD, train_record_max_duration=$CONTROLLER_TRAIN_RECORD_MAX_DURATION, controller_eval_max_hold=$CONTROLLER_EVAL_MAX_HOLD, max switches=$CONTROLLER_MAX_SWITCHES, max_switch_penalty=$CONTROLLER_MAX_SWITCH_PENALTY_COEF, expected_switch_penalty=$CONTROLLER_EXPECTED_SWITCH_PENALTY_COEF, overflow_action_penalty=$CONTROLLER_OVERFLOW_ACTION_PENALTY_COEF, fixed_pool_limit=$CONTROLLER_FIXED_POOL_LIMIT, skip_fixed_scenarios=$TEST_SKIP_FIXED_SCENARIOS, test_max_days=$TEST_MAX_DAYS"

CONTROLLER_SKIP_VAL_ARGS=()
if [[ "$CONTROLLER_SKIP_VAL" == "1" ]]; then
  CONTROLLER_SKIP_VAL_ARGS=(--controller_skip_val)
fi

TEST_SKIP_FIXED_SCENARIOS_ARGS=()
if [[ "$TEST_SKIP_FIXED_SCENARIOS" == "1" ]]; then
  TEST_SKIP_FIXED_SCENARIOS_ARGS=(--test_skip_fixed_scenarios)
fi

CONTROLLER_VALUE_NORMALIZE_ARGS=()
if [[ "$CONTROLLER_VALUE_NORMALIZE_ADVANTAGE" != "1" ]]; then
  CONTROLLER_VALUE_NORMALIZE_ARGS=(--no_controller_value_normalize_advantage)
fi

CONTROLLER_LOCAL_ADV_BALANCE_ARGS=()
if [[ "$CONTROLLER_LOCAL_ADV_BALANCE_CLASSES" == "1" ]]; then
  CONTROLLER_LOCAL_ADV_BALANCE_ARGS=(--controller_local_adv_balance_classes)
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

CONTROLLER_EVAL_DIAGNOSTICS_ARGS=()
if [[ "$CONTROLLER_EVAL_DIAGNOSTICS" == "1" ]]; then
  CONTROLLER_EVAL_DIAGNOSTICS_ARGS=(--controller_eval_diagnostics --controller_eval_diag_thresholds $CONTROLLER_EVAL_DIAG_THRESHOLDS)
fi

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
  echo "===== Daily aux-PG controller run: market=${market}, seed=${seed} ====="
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
    --controller_mdd_coef "$CONTROLLER_MDD_COEF" \
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
    --joint_epochs "$EFFECTIVE_JOINT_EPOCHS" \
    --joint_lr_mult "$JOINT_LR_MULT" \
    --ppo_epochs "$PPO_EPOCHS" \
    --outer_pred_coef "$OUTER_PRED_COEF" \
    --inner_pred_coef "$INNER_PRED_COEF" \
    --inner_pred_target_scale "$INNER_PRED_TARGET_SCALE" \
    --model_selection_metric return \
    --inner_selection_metric return \
    --frozen_hrl_checkpoint "$checkpoint" \
    "${FINETUNE_MODE_ARGS[@]}" \
    --train_monitor \
    "${TEST_SKIP_FIXED_SCENARIOS_ARGS[@]}" \
    --test_max_days "$TEST_MAX_DAYS" \
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

echo "Daily aux-PG controller runs finished: $OUTPUT_ROOT/$RUN_NAME"
