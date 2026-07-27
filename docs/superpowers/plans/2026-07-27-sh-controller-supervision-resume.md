# SH Controller Strong-Supervision Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a directly runnable two-GPU script that resumes only the Controller for SH seeds 44 and 46 with a 60-day Controller horizon and PG switch-supervision coefficient 0.10.

**Architecture:** A single Bash scheduler owns the two seed-specific checkpoint defaults and launches one isolated job per GPU. Each job invokes the existing `run_hrl_training.py` controller-only path, freezes Outer+Inner through `--frozen_hrl_checkpoint`, and writes an independent log and result directory.

**Tech Stack:** Bash, Python CLI, pytest

---

### Task 1: Define the script contract with a failing dry-run test

**Files:**
- Create: `tests/test_resume_sh_controller_strong_supervision_script.py`
- Create: `train_sh/resume_sh_controller_44_46_strong_sup.sh`

- [ ] **Step 1: Write the failing test**

Create a pytest that runs the absent script with `DRY_RUN=1` and asserts:

```python
assert "--markets sh" in output
assert "--seeds 44" in output
assert "--seeds 46" in output
assert "--controller_only_finetune" in output
assert output.count("--controller_sup_coef 0.10") == 2
assert output.count("--max_hold 60") == 2
assert output.count("--controller_train_max_hold 60") == 2
assert output.count("--controller_eval_max_hold 60") == 2
assert output.count("--frozen_hrl_checkpoint") == 2
assert "--warmup_outer_epochs 0" in output
assert "--warmup_inner_epochs 0" in output
assert "--joint_epochs 0" in output
assert "--skip_test" not in output
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_resume_sh_controller_strong_supervision_script.py -q
```

Expected: FAIL because `train_sh/resume_sh_controller_44_46_strong_sup.sh` does not exist.

### Task 2: Implement the two-seed resume scheduler

**Files:**
- Create: `train_sh/resume_sh_controller_44_46_strong_sup.sh`

- [ ] **Step 1: Implement environment configuration**

Support:

```bash
GPU_44="${GPU_44:-0}"
GPU_46="${GPU_46:-1}"
CONTROLLER_EPOCHS="${CONTROLLER_EPOCHS:-5}"
CONTROLLER_SUP_COEF="${CONTROLLER_SUP_COEF:-0.10}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/sh_controller_strong_sup_resume}"
```

Provide separate `SOURCE_CHECKPOINT_44` and `SOURCE_CHECKPOINT_46` defaults and allow both to be overridden.

- [ ] **Step 2: Implement one controller-only command**

For each seed, construct a Bash array containing:

```bash
--frozen_hrl_checkpoint "$checkpoint"
--controller_only_finetune
--warmup_outer_epochs 0
--warmup_inner_epochs 0
--joint_epochs 0
--max_hold 60
--controller_train_max_hold 60
--controller_eval_max_hold 60
--controller_sup_coef "$CONTROLLER_SUP_COEF"
--controller_guidance_pretrain_coef 1.0
--controller_aux_mdd_coef 0.01
--controller_aux_switch_adv_coef 0.01
```

Retain final testing by omitting `--skip_test`.

- [ ] **Step 3: Add validation and isolated execution**

In normal mode, require executable Python and both checkpoint files before launching either task. Run seed 44 and 46 in parallel, write separate logs, wait for both PIDs, and return nonzero if either fails. In dry-run mode, print both commands without requiring checkpoint files.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_resume_sh_controller_strong_supervision_script.py -q
```

Expected: PASS.

### Task 3: Regression and shell validation

**Files:**
- Verify: `train_sh/resume_sh_controller_44_46_strong_sup.sh`
- Verify: `tests/test_resume_sh_controller_strong_supervision_script.py`

- [ ] **Step 1: Validate Bash syntax**

Run:

```bash
bash -n train_sh/resume_sh_controller_44_46_strong_sup.sh
```

Expected: exit code 0.

- [ ] **Step 2: Inspect generated commands**

Run:

```bash
DRY_RUN=1 PYTHON_BIN=/bin/echo \
  bash train_sh/resume_sh_controller_44_46_strong_sup.sh
```

Expected: two commands, seeds 44 and 46, GPUs 0 and 1, Controller-only training, 60-day horizon, PG supervision 0.10, and no `--skip_test`.

- [ ] **Step 3: Run focused existing regression**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_controller_seed_sweep_3gpu_script.py \
  tests/test_full_cmtflow_controller_maxhold60_script.py -q
```

Expected: existing controller script tests remain green.

No commit or push is performed.
