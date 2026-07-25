# Staged Seed Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-tdd while implementing each behavior. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a staged CMTFlow seed-screening flow: train Outer first, then freeze Outer and train Controller without Inner, then freeze Outer+Controller and train Inner, with per-stage ranking against DeepTrader TR and DeepAries Sharpe.

**Architecture:** Reuse the existing `run_hrl_training.py` child/parent flow and trainer checkpoint format. Add only the missing hooks: controller PG can bypass Inner, an Inner-only-from-frozen-controller finetune mode, a controller-without-inner test scenario, a ranking utility, and a shell wrapper that runs one stage at a time.

**Tech Stack:** Python, PyTorch, pandas/numpy, unittest, bash.

---

### Task 1: Ranking Utility

**Files:**
- Create: `scripts/staged_seed_screening.py`
- Test: `tests/test_staged_seed_screening.py`

- [x] Write failing tests for DeepTrader TR / DeepAries Sharpe target extraction.
- [x] Write failing tests for stage ranking rows and pass flags.
- [ ] Implement curve metric computation and stage scan.
- [ ] Run the ranking tests.

### Task 2: Controller PG Without Inner

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `run_hrl_training.py`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_run_hrl_training_command.py`

- [x] Write failing tests for `controller_pg_disable_inner` command forwarding.
- [x] Write failing tests showing controller PG execution can bypass Inner.
- [ ] Implement `--controller_pg_disable_inner` and use `base_used` as execution weights in both baseline and controlled paths.
- [ ] Run focused tests.

### Task 3: Inner-Only From Frozen Controller

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `run_hrl_training.py`
- Test: `tests/test_run_hrl_training_command.py`

- [x] Write failing tests for `--inner_only_finetune` command forwarding.
- [ ] Implement trainer entry that freezes Outer+Controller and updates Inner with controller-active segments.
- [ ] Run command tests.

### Task 4: Stage Script

**Files:**
- Create: `train_sh/run_staged_seed_screening_sharpe_cr.sh`
- Test: `tests/test_staged_seed_screening_script.py`

- [x] Write failing echo tests for outer/controller/inner stages.
- [ ] Implement stage wrapper with isolated output roots and explicit checkpoint-root inputs for later stages.
- [ ] Run script tests.

### Verification

- [ ] Run targeted unittest files.
- [ ] Run bash syntax checks for new/modified scripts.
