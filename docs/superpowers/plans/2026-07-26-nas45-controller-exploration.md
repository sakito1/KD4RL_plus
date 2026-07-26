# NAS45 Controller Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add one safe, dry-runnable shell entry point for probing and comparing Controller-only training recipes from the fixed NAS seed 45 Outer+Inner checkpoint.

**Architecture:** A single shell script maps `MODE=probe|pg_only|sup_only|sup_pg` to explicit `run_hrl_training.py` arguments. It never edits or overwrites the source checkpoint, and each mode receives a distinct run name. A focused pytest executes dry runs and verifies checkpoint selection, frozen-module mode, supervision/PG toggles, and the approved hyperparameters.

**Tech Stack:** Bash, Python CLI, pytest.

---

### Task 1: Specify the shell interface with a failing dry-run test

**Files:**
- Create: `tests/test_explore_controller_from_nas45_outer_inner_script.py`
- Create: `train_sh/explore_controller_from_nas45_outer_inner.sh`

- [ ] **Step 1: Write a failing pytest**

The test runs all four modes with `DRY_RUN=1` and `/bin/echo`, then asserts:

```python
assert "--frozen_hrl_checkpoint" in output
assert "hrl_fixed_best.pth" in output
assert "--controller_only_finetune" in output
assert "--trade_num 5" in output
assert "--controller_rollout_len 300" in output
assert "--controller_eval_max_hold 30" in output
```

It additionally verifies mode-specific behavior:

```python
assert "--controller_guidance_probe_only" in probe_output
assert "--controller_sup_pretrain_epochs 0" in pg_only_output
assert "--controller_pretrain_only" in sup_only_output
assert "--controller_sup_pretrain_epochs 1" in sup_pg_output
assert "--controller_aux_replay_epochs 30" in sup_pg_output
assert "--controller_epochs 3" in sup_pg_output
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest -q tests/test_explore_controller_from_nas45_outer_inner_script.py
```

Expected: FAIL because `train_sh/explore_controller_from_nas45_outer_inner.sh` does not exist.

### Task 2: Implement the four-mode Controller-only launcher

**Files:**
- Create: `train_sh/explore_controller_from_nas45_outer_inner.sh`

- [ ] **Step 1: Add input and checkpoint guards**

The script must:

```bash
set -euo pipefail
MODE="${MODE:-sup_pg}"
CONTROLLER_SEED="${CONTROLLER_SEED:-45}"
DRY_RUN="${DRY_RUN:-0}"
```

It accepts only `probe`, `pg_only`, `sup_only`, or `sup_pg`, checks that the frozen checkpoint exists, and refuses an existing mode/seed output directory unless `ALLOW_EXISTING_OUTPUT=1`.

- [ ] **Step 2: Add common approved Controller arguments**

All modes use:

```text
--markets nas
--trade_num 5
--controller_only_finetune
--controller_rollout_len 300
--controller_train_max_hold 30
--controller_eval_max_hold 30
--controller_decision_mode daily
--controller_eval_decision_mode daily
--controller_init_exit_bias 0.0
--controller_reward_mode return_uplift
--controller_selection_metric return
--controller_guidance_risk_threshold 0.05
--controller_guidance_advantage_threshold 0.05
--controller_aux_mdd_target_scale 20.0
--controller_aux_switch_adv_target_scale 20.0
```

Outer and Inner epochs are zero and `--controller_only_finetune` keeps them frozen.

- [ ] **Step 3: Add mode-specific arguments**

Use these exact mappings:

```text
probe:    guidance probe only; no training
pg_only: pretrain=0, PG=3, switch supervision disabled
sup_only: pretrain=1, replay=30, PG path skipped with pretrain-only
sup_pg:   pretrain=1, replay=30, PG=3
```

Both supervised modes pass `--controller_use_switch_supervision`,
`--controller_sup_coef 0.1`, and the two 5% guidance thresholds.

- [ ] **Step 4: Add dry-run output**

When `DRY_RUN=1`, print the fully escaped command without launching Python. Otherwise create the output root and pipe stdout/stderr through `tee` to a mode/seed-specific log.

- [ ] **Step 5: Run the focused test**

Run:

```bash
pytest -q tests/test_explore_controller_from_nas45_outer_inner_script.py
```

Expected: PASS.

### Task 3: Verify syntax and all four generated commands

**Files:**
- Verify: `train_sh/explore_controller_from_nas45_outer_inner.sh`
- Verify: `tests/test_explore_controller_from_nas45_outer_inner_script.py`

- [ ] **Step 1: Check Bash syntax**

Run:

```bash
bash -n train_sh/explore_controller_from_nas45_outer_inner.sh
```

Expected: exit code 0.

- [ ] **Step 2: Inspect the recommended command**

Run:

```bash
DRY_RUN=1 MODE=sup_pg bash train_sh/explore_controller_from_nas45_outer_inner.sh
```

Expected: one Controller-only NAS command using the seed 45 frozen checkpoint, one supervised data collection, 30 replay updates, and three PG epochs.

- [ ] **Step 3: Run related CLI tests**

Run:

```bash
pytest -q \
  tests/test_explore_controller_from_nas45_outer_inner_script.py \
  tests/test_run_hrl_training_command.py
```

Expected: PASS.

No Git commit is performed because the user requested local changes only.
