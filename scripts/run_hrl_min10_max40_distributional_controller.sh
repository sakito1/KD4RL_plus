#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-min10_max40_distributional_controller}"

exec "$SCRIPT_DIR/run_hrl_min10_max40_pg_actor_smoothl1.sh" "$@"
