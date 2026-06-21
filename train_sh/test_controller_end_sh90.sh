#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

COMMAND_JSON="/home/tongwenxuan/KD4RL_plus/results/end/sh_seed90/seed_90_command.json"
CHECKPOINT="/home/tongwenxuan/KD4RL_plus/results/end/sh_seed90/checkpoints/best_model.pth"
RUN_ROOT="${RUN_ROOT:-/home/tongwenxuan/KD4RL_plus/results/end_replay/sh_seed90}"

if [[ ! -f "$COMMAND_JSON" ]]; then
  echo "Missing command json: $COMMAND_JSON" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

mapfile -t cmd < <(jq -r '.command[]' "$COMMAND_JSON")
if [[ -n "${PYTHON_BIN:-}" ]]; then
  cmd[0]="$PYTHON_BIN"
fi

for i in "${!cmd[@]}"; do
  if [[ "${cmd[$i]}" == "--run_root" ]]; then
    cmd[$((i + 1))]="$RUN_ROOT"
  fi
done

mkdir -p "$RUN_ROOT"
cmd+=(--test_only_checkpoint "$CHECKPOINT")
exec "${cmd[@]}"
