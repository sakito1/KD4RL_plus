#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_controller_pg}"
RUN_NAME="${RUN_NAME:-min10_max40_pg_countband_return_select}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.01}"

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

echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $PYTHON_BIN"
echo "Checkpoint selection: fixed_hrl=return controller=return"
echo "Joint LR multiplier: $JOINT_LR_MULT (base lr 1e-3 -> joint lr 1e-5)"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets sh \
  --seeds 42 43 44 45 46 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --warmup_inner_epochs 5 \
  --controller_epochs 10 \
  --controller_val_interval_epochs 1 \
  --controller_mdd_coef 5.0 \
  --controller_return_coef 0.2 \
  --controller_pg_batch_windows 4 \
  --controller_windows_per_epoch 5 \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_sh.log"

  
"$PYTHON_BIN" -u run_hrl_training.py \
  --markets nas \
  --seeds 42 43 44 45 46 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --warmup_inner_epochs 5 \
  --controller_epochs 10 \
  --controller_val_interval_epochs 1 \
  --controller_mdd_coef 5.0 \
  --controller_return_coef 0.2 \
  --controller_pg_batch_windows 4 \
  --controller_windows_per_epoch 5 \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_nas.log"

echo "All requested return-selection HRL runs finished: $OUTPUT_ROOT/$RUN_NAME"
