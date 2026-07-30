#!/usr/bin/env bash
set -euo pipefail

if (( $# < 10 || ($# - 7) % 3 != 0 )); then
  echo "Internal worker received an invalid argument list." >&2
  exit 2
fi

gpu="$1"
training_script="$2"
run_prefix="$3"
python_bin="$4"
output_root="$5"
nas_seeds="$6"
sh_seeds="$7"
shift 7

while (( $# > 0 )); do
  market="$1"
  k="$2"
  seed_string="$3"
  shift 3

  run_name="${run_prefix}_k${k}_${market}"

  echo
  echo "===== START gpu=$gpu market=$market k=$k seeds=$seed_string ====="
  REPRODUCE_BEST_MODE=0 \
  CUDA_VISIBLE_DEVICES="$gpu" \
  GPU_ID="$gpu" \
  MARKETS="$market" \
  TRADE_NUM="$k" \
  NAS_SEEDS="$nas_seeds" \
  SH_SEEDS="$sh_seeds" \
  OUTPUT_ROOT="$output_root" \
  RUN_NAME="$run_name" \
  PYTHON_BIN="$python_bin" \
    bash "$training_script"
  echo "===== DONE gpu=$gpu market=$market k=$k ====="
done
