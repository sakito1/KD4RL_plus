# Multi-Market, Multi-K, Multi-Seed Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Provide one command that runs the final CMTFlow recipe over configurable markets, K values, and market-specific seed lists using any operator-supplied GPU list.

**Architecture:** Extend the existing final training shell entry point with backward-compatible `TRADE_NUM` and `MARKETS` environment controls. Add a launcher that creates the `MARKETS × K_VALUES` configuration matrix, assigns it round-robin across `GPU_IDS`, and runs one sequential worker per GPU.

**Tech Stack:** Bash, Python `unittest`/`subprocess`, existing `run_hrl_training.py`.

---

## File Structure

- Modify `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`: accept K and market controls, derive the repository root portably, and forward K to Python.
- Create `scripts/run_multi_market_multi_k_seed_sweep.sh`: validate configuration, print a dry run, launch one worker per GPU, and propagate failures.
- Create `tests/test_multi_market_multi_k_seed_sweep.py`: exercise both shell scripts without starting model training.

### Task 1: Add K and Market Controls to the Final Training Entry Point

**Files:**
- Modify: `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`
- Create: `tests/test_multi_market_multi_k_seed_sweep.py`

- [ ] **Step 1: Write the failing compatibility test**

Add a test that creates an executable fake Python command in `tmp_path`, invokes the
training shell script with:

```python
env.update({
    "PYTHON_BIN": str(fake_python),
    "OUTPUT_ROOT": str(tmp_path / "outputs"),
    "RUN_NAME": "compat",
    "MARKETS": "nas",
    "NAS_SEEDS": "42 43",
    "SH_SEEDS": "90",
    "TRADE_NUM": "5",
    "CUDA_VISIBLE_DEVICES": "0",
})
```

Assert that the captured command contains:

```text
--markets nas
--seeds 42 43
--trade_num 5
```

and does not contain `--markets sh`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_multi_market_multi_k_seed_sweep.py::test_final_training_script_forwards_k_and_filters_market -q
```

Expected: FAIL because `TRADE_NUM` and `MARKETS` are not yet supported.

- [ ] **Step 3: Implement the minimum compatibility changes**

In the final training script:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRADE_NUM="${TRADE_NUM:-10}"
MARKETS="${MARKETS:-sh nas}"
```

Forward:

```bash
--trade_num "$TRADE_NUM"
```

Replace the two unconditional `run_market` calls with:

```bash
for market in $MARKETS; do
  case "$market" in
    nas) run_market nas "$NAS_SEEDS" ;;
    sh) run_market sh "$SH_SEEDS" ;;
    *)
      echo "Unsupported market in MARKETS: $market" >&2
      exit 2
      ;;
  esac
done
```

- [ ] **Step 4: Run the compatibility test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Commit the compatibility change**

```bash
git add train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh \
  tests/test_multi_market_multi_k_seed_sweep.py
git commit -m "feat: parameterize final training market and K"
```

### Task 2: Add the GPU-Round-Robin Launcher

**Files:**
- Create: `scripts/run_multi_market_multi_k_seed_sweep.sh`
- Modify: `tests/test_multi_market_multi_k_seed_sweep.py`

- [ ] **Step 1: Write the failing launcher dry-run test**

Invoke the not-yet-existing launcher with `DRY_RUN=1`, `GPU_IDS="2 5"`,
`MARKETS="nas sh"`, and `K_VALUES="5 15"`. Assert that output contains:

```text
job=0 gpu=2 market=nas k=5
job=1 gpu=5 market=nas k=15
job=2 gpu=2 market=sh k=5
job=3 gpu=5 market=sh k=15
```

Also assert that the output reports two GPU workers and the correct seed-run
count. A second dry-run test verifies the default four configurations contain
sixty seed runs on the default one-GPU setup.

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_multi_market_multi_k_seed_sweep.py::test_launcher_dry_run_round_robins_configs_across_available_gpus -q
```

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Create a strict Bash script with:

```bash
set -euo pipefail
```

Use environment-overridable defaults for:

```bash
PYTHON_BIN
OUTPUT_ROOT
GPU_IDS
MARKETS
K_VALUES
NAS_SEEDS
SH_SEEDS
DRY_RUN
```

Validate that seed strings and GPU/K lists contain integers and markets are
`nas` or `sh`. Build the Cartesian product and assign each configuration to a
GPU round-robin. In dry-run mode, print records and exit.

For a real run, start one background subshell per GPU. Each subshell processes
its assigned configurations sequentially and exports:

```bash
CUDA_VISIBLE_DEVICES=<assigned GPU>
MARKETS=<assigned market>
TRADE_NUM=<assigned K>
NAS_SEEDS=<configured NAS seeds>
SH_SEEDS=<configured SH seeds>
RUN_NAME=k<k>_<market>
OUTPUT_ROOT=<shared root>
PYTHON_BIN=<configured interpreter>
```

and executes:

```bash
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

Redirect each worker through `tee` to:

```text
<OUTPUT_ROOT>/launcher_logs/gpu_<gpu>.log
```

Store PIDs, wait for all four, list failed workers, and return nonzero if any
worker fails.

- [ ] **Step 4: Run the launcher test and verify GREEN**

Run the command from Step 2.

Expected: `1 passed`.

- [ ] **Step 5: Verify shell syntax**

Run:

```bash
bash -n scripts/run_multi_market_multi_k_seed_sweep.sh
bash -n train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

Expected: both commands exit `0` without output.

- [ ] **Step 6: Commit the launcher**

```bash
git add scripts/run_multi_market_multi_k_seed_sweep.sh \
  tests/test_multi_market_multi_k_seed_sweep.py
git commit -m "feat: schedule multi-market multi-K seed sweeps"
```

### Task 3: Full Verification and Operator Handoff

**Files:**
- Verify only.

- [ ] **Step 1: Run the focused test file**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_multi_market_multi_k_seed_sweep.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run related shell-entry tests**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_run_hrl_training_command.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect the final dry run**

```bash
DRY_RUN=1 bash scripts/run_multi_market_multi_k_seed_sweep.sh
```

Expected: four configurations and sixty total seed runs, with no Python
training process.

- [ ] **Step 4: Confirm the working tree scope**

```bash
git status --short
git diff --check HEAD^
```

Expected: no unintended files from this implementation and no whitespace
errors in its commits.
