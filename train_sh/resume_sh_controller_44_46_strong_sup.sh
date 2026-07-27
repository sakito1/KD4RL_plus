#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sh_controller_strong_sup_resume}"
SOURCE_CHECKPOINT_44="${SOURCE_CHECKPOINT_44:-results/sh_full_42135_seed44/sh/ppo/seed_44/checkpoints/hrl_fixed_best.pth}"
SOURCE_CHECKPOINT_46="${SOURCE_CHECKPOINT_46:-results/full_cmtflow_seed_sweep_4gpu/sh_full_42135_seed46/sh/ppo/seed_46/checkpoints/hrl_fixed_best.pth}"
GPU_44="${GPU_44:-0}"
GPU_46="${GPU_46:-1}"
CONTROLLER_EPOCHS="${CONTROLLER_EPOCHS:-5}"
CONTROLLER_SUP_COEF="${CONTROLLER_SUP_COEF:-0.10}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

if ! [[ "$CONTROLLER_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTROLLER_EPOCHS must be a positive integer: $CONTROLLER_EPOCHS" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 1
fi

declare -A CHECKPOINTS=(
  [44]="$SOURCE_CHECKPOINT_44"
  [46]="$SOURCE_CHECKPOINT_46"
)
declare -A GPUS=(
  [44]="$GPU_44"
  [46]="$GPU_46"
)

for seed in 44 46; do
  checkpoint="${CHECKPOINTS[$seed]}"
  run_name="sh_controller_strong_sup010_seed${seed}"
  run_dir="$OUTPUT_ROOT/$run_name/sh/ppo/seed_${seed}"
  if [[ "$DRY_RUN" != "1" && ! -f "$checkpoint" ]]; then
    echo "Missing frozen Outer+Inner checkpoint for seed $seed: $checkpoint" >&2
    exit 1
  fi
  if [[ "$DRY_RUN" != "1" && "$ALLOW_EXISTING_OUTPUT" != "1" && -e "$run_dir" ]]; then
    echo "Refusing to reuse existing output for seed $seed: $run_dir" >&2
    echo "Set ALLOW_EXISTING_OUTPUT=1 or choose another OUTPUT_ROOT." >&2
    exit 1
  fi
done

build_command() {
  local seed="$1"
  local checkpoint="$2"
  local run_name="sh_controller_strong_sup010_seed${seed}"
  COMMAND=(
    "$PYTHON_BIN" -u run_hrl_training.py
    --markets sh
    --seeds "$seed"
    --output_root "$OUTPUT_ROOT"
    --run_name "$run_name"
    --device cuda
    --frozen_hrl_checkpoint "$checkpoint"
    --controller_only_finetune
    --trade_num 5
    --outer_window 60
    --min_hold 30
    --max_hold 60
    --train_episodes_per_epoch 1
    --train_start_stride_days 1
    --warmup_outer_epochs 0
    --warmup_inner_epochs 0
    --joint_epochs 0
    --controller_epochs "$CONTROLLER_EPOCHS"
    --controller_sup_coef "$CONTROLLER_SUP_COEF"
    --controller_sup_pretrain_epochs 3
    --controller_sup_pretrain_rollout_len 300
    --controller_aux_replay_epochs 30
    --controller_guidance_risk_threshold 0.05
    --controller_guidance_advantage_threshold 0.05
    --controller_guidance_pretrain_coef 1.0
    --controller_rollout_len 300
    --controller_windows_per_epoch 12
    --controller_fixed_pool_limit 12
    --controller_pg_batch_windows 12
    --controller_pg_logprob_reduction mean
    --controller_train_fixed_episodes
    --controller_episode_batch_size 12
    --controller_episode_parallel_workers 6
    --controller_start_stride_days 5
    --controller_train_max_hold 60
    --controller_eval_max_hold 60
    --controller_window 30
    --controller_hidden_dim 64
    --controller_init_exit_bias 0.0
    --controller_return_coef 1.0
    --controller_downside_coef 0.0
    --controller_mdd_coef 0.0
    --controller_reward_mode return_uplift
    --controller_selection_metric return
    --controller_no_hold_constraints
    --controller_decision_mode daily
    --controller_eval_decision_mode daily
    --controller_eval_switch_threshold 0.5
    --controller_eval_diagnostics
    --controller_eval_diag_thresholds 0.3 0.4 0.5 0.6 0.7
    --controller_max_switches 0
    --controller_max_switch_penalty_coef 0.0
    --controller_switch_rate_penalty_coef 0.01
    --controller_switch_rate_min 0.05
    --controller_switch_rate_max 0.15
    --controller_switch_rate_margin 0.1
    --controller_entropy_coef 0.001
    --controller_aux_return_coef 0.0
    --controller_aux_mdd_coef 0.01
    --controller_aux_mdd_target_scale 20.0
    --controller_aux_switch_adv_coef 0.01
    --controller_aux_switch_adv_target_scale 20.0
    --controller_aux_switch_adv_loss_type mse
    --controller_compute_switch_advantage
    --controller_use_switch_supervision
    --controller_val_interval_epochs 1
    --val_interval 1
    --model_selection_metric return
    --inner_selection_metric return
    --ppo_epochs 1
    --train_monitor
    --test_skip_fixed_scenarios
    --heartbeat_seconds "$HEARTBEAT_SECONDS"
  )
}

run_seed() {
  local seed="$1"
  local gpu="${GPUS[$seed]}"
  local checkpoint="${CHECKPOINTS[$seed]}"
  local run_name="sh_controller_strong_sup010_seed${seed}"
  local log_file="$OUTPUT_ROOT/logs/${run_name}.log"

  build_command "$seed" "$checkpoint"
  echo "seed=$seed GPU=$gpu checkpoint=$checkpoint"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q COMMAND:' "$gpu"
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    return 0
  fi

  mkdir -p "$OUTPUT_ROOT/logs"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-sh-controller-strong-sup-${seed}}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "${COMMAND[@]}" 2>&1 | tee "$log_file"
  )
}

if [[ "$DRY_RUN" == "1" ]]; then
  run_seed 44
  run_seed 46
  exit 0
fi

echo "SH Controller strong-supervision resume"
echo "PG switch supervision coefficient: $CONTROLLER_SUP_COEF"
echo "Controller horizon: 60 trading days"
echo "Outer+Inner checkpoints remain frozen"

run_seed 44 &
pid_44=$!
run_seed 46 &
pid_46=$!

status=0
if ! wait "$pid_44"; then
  echo "seed 44 failed; inspect $OUTPUT_ROOT/logs/sh_controller_strong_sup010_seed44.log" >&2
  status=1
fi
if ! wait "$pid_46"; then
  echo "seed 46 failed; inspect $OUTPUT_ROOT/logs/sh_controller_strong_sup010_seed46.log" >&2
  status=1
fi

summary_file="$OUTPUT_ROOT/test_results_summary.txt"
: >"$summary_file"
for seed in 44 46; do
  log_file="$OUTPUT_ROOT/logs/sh_controller_strong_sup010_seed${seed}.log"
  {
    echo "===== SH seed=$seed ====="
    if [[ -f "$log_file" ]]; then
      grep -E \
        "CTRL-AUX|CTRL-PG|VAL controller_pg|Controller eval exit_prob:|TEST REPORT|Switches[[:space:]]*:|Switch detail:|Total Ret[[:space:]]*:|Ann Ret[[:space:]]*:|Sharpe[[:space:]]*:|Max DD[[:space:]]*:" \
        "$log_file" || echo "(no matching statistics; inspect $log_file)"
    else
      echo "(missing log: $log_file)"
    fi
    echo
  } >>"$summary_file"
done

echo "Summary: $summary_file"
exit "$status"
