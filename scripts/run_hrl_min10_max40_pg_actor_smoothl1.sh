#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_controller_pg}"
RUN_NAME="${RUN_NAME:-min10_max40_distributional_controller}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.001}"
OUTER_PRED_COEF="${OUTER_PRED_COEF:-0.1}"
INNER_PRED_COEF="${INNER_PRED_COEF:-0.0}"

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
echo "Joint LR multiplier: $JOINT_LR_MULT (base lr 1e-3 -> joint lr 1e-6)"
echo "Actor SmoothL1 supervision: outer=$OUTER_PRED_COEF inner=$INNER_PRED_COEF"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets sh \
  --seeds 75 76 77 78 79 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --outer_pred_coef "$OUTER_PRED_COEF" \
  --inner_pred_coef "$INNER_PRED_COEF" \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_sh.log"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets nas \
  --seeds 42 43 44 45 46 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --outer_pred_coef "$OUTER_PRED_COEF" \
  --inner_pred_coef "$INNER_PRED_COEF" \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_nas.log"



echo "All requested HRL runs finished: $OUTPUT_ROOT/$RUN_NAME"
