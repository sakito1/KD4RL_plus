# SH Full CMTFlow Risk-10/Advantage-0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a four-GPU SH training sweep for seeds 44, 46, 49, and 54 that trains all three modules from scratch and applies `(Risk >= 10% and Advantage > 0%) or Advantage >= 10%`.

**Architecture:** A dedicated single-seed script owns the exact training command, while a dedicated scheduler assigns the four default seeds to four GPU queues. A dry-run integration test treats the rendered commands as the public interface and prevents the new experiment from changing existing scripts.

**Tech Stack:** Bash, Python `pytest`, existing `run_hrl_training.py` CLI.

---

### Task 1: Add the failing dry-run contract tests

**Files:**
- Create: `tests/test_sh_full_cmtflow_risk10_adv0_script.py`

- [ ] **Step 1: Write a test for the single-seed command**

Create a test that runs the new single-seed script with `DRY_RUN=1` and asserts:

```python
assert "--markets sh" in output
assert "--warmup_outer_epochs 4" in output
assert "--warmup_inner_epochs 2" in output
assert "--joint_epochs 1" in output
assert "--controller_sup_pretrain_epochs 3" in output
assert "--controller_epochs 5" in output
assert "--controller_guidance_risk_threshold 0.10" in output
assert "--controller_guidance_risk_min_advantage_threshold 0.00" in output
assert "--controller_guidance_advantage_threshold 0.10" in output
assert "--frozen_hrl_checkpoint" not in output
assert "--controller_only_finetune" not in output
assert "--no_train_controller" not in output
```

- [ ] **Step 2: Write a test for the four-GPU scheduler**

Run the scheduler with `DRY_RUN=1`, explicit GPU IDs, and `/bin/echo`. Assert
each seed is started once, each GPU has one lane, all rendered run names include
`risk10_adv0_or_adv10`, and no NAS job is emitted.

- [ ] **Step 3: Verify RED**

Run:

```bash
pytest -q tests/test_sh_full_cmtflow_risk10_adv0_script.py
```

Expected: failures because the two new scripts do not exist.

### Task 2: Add the dedicated single-seed training script

**Files:**
- Create: `train_sh/train_sh_full_cmtflow_risk10_adv0_seed.sh`
- Test: `tests/test_sh_full_cmtflow_risk10_adv0_script.py`

- [ ] **Step 1: Implement the command**

Copy the approved training schedule and non-guidance hyperparameters from
`train_sh/train_full_cmtflow_seed.sh`, restrict the market to SH, and pass:

```bash
--controller_guidance_risk_threshold 0.10
--controller_guidance_risk_min_advantage_threshold 0.00
--controller_guidance_advantage_threshold 0.10
```

Use the default output root
`results/sh_full_cmtflow_risk10_adv0_or_adv10_4gpu` and run name
`sh_full_42135_risk10_adv0_or_adv10_seed${SEED}`. Refuse an existing run
directory unless `ALLOW_EXISTING_OUTPUT=1`.

- [ ] **Step 2: Run the focused single-seed test**

Run:

```bash
pytest -q tests/test_sh_full_cmtflow_risk10_adv0_script.py -k single
```

Expected: the single-seed test passes.

### Task 3: Add the four-GPU scheduler

**Files:**
- Create: `train_sh/run_sh_full_cmtflow_risk10_adv0_4gpu.sh`
- Test: `tests/test_sh_full_cmtflow_risk10_adv0_script.py`

- [ ] **Step 1: Implement scheduling and summaries**

Default `SH_SEEDS` to `44 46 49 54`, default GPUs to `0 1 2 3`, and default
`JOBS_PER_GPU` to `1`. Queue jobs round-robin, wait for every queue, propagate
failures, and create `test_results_summary.txt` from scheduler logs after real
runs.

- [ ] **Step 2: Run the complete focused test**

Run:

```bash
pytest -q tests/test_sh_full_cmtflow_risk10_adv0_script.py
```

Expected: all focused tests pass.

### Task 4: Verify shell syntax and relevant regression coverage

**Files:**
- Verify: `train_sh/train_sh_full_cmtflow_risk10_adv0_seed.sh`
- Verify: `train_sh/run_sh_full_cmtflow_risk10_adv0_4gpu.sh`

- [ ] **Step 1: Check Bash syntax**

Run:

```bash
bash -n train_sh/train_sh_full_cmtflow_risk10_adv0_seed.sh
bash -n train_sh/run_sh_full_cmtflow_risk10_adv0_4gpu.sh
```

Expected: both commands exit 0 without output.

- [ ] **Step 2: Run focused and adjacent tests**

Run:

```bash
pytest -q \
  tests/test_sh_full_cmtflow_risk10_adv0_script.py \
  tests/test_full_cmtflow_seed_sweep_4gpu_script.py \
  tests/test_controller_guidance.py \
  tests/test_run_hrl_training_command.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check
git status --short
```

Confirm the implementation only adds the dedicated plan, scripts, and test; all
pre-existing modified files remain otherwise untouched.
