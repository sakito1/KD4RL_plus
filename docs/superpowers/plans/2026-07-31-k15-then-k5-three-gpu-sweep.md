# K=15 Then K=5 Three-GPU Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add one shell launcher that runs seeds 50--61 on both markets across three GPUs, finishing K=15 before starting K=5.

**Architecture:** A standalone Bash launcher shards twelve seeds across three fixed GPU workers. Each worker invokes the existing abk E2E shell entry for both markets; a phase-level wait provides the K=15-to-K=5 barrier.

**Tech Stack:** Bash, pytest, existing E2E training shell entry.

---

### Task 1: Specify the dry-run contract

**Files:**
- Create: `tests/test_k15_then_k5_three_gpu_sweep.py`

- [ ] Write a test invoking the launcher with `DRY_RUN=1`.
- [ ] Assert the resolved repository/E2E paths point to the current checkout.
- [ ] Assert phase 1 is K=15 and phase 2 is K=5.
- [ ] Assert GPU seed shards are `50 53 56 59`, `51 54 57 60`, and `52 55 58 61` in both phases.
- [ ] Assert each phase reports both `nas sh` markets and 24 runs.
- [ ] Run the focused test and confirm it fails because the launcher is missing.

### Task 2: Implement the staged launcher

**Files:**
- Create: `scripts/run_k15_then_k5_3gpu_seed50_61.sh`

- [ ] Add strict Bash mode and resolve `REPO_ROOT` from `BASH_SOURCE`.
- [ ] Add environment-overridable defaults for three GPU IDs, seeds 50--61, both markets, Python, output root, run prefix, and dry-run.
- [ ] Validate exactly three unique GPU IDs and twelve unique integer seeds.
- [ ] Shard seeds round-robin across GPUs.
- [ ] Print a complete dry-run manifest without starting processes.
- [ ] Implement a phase function that starts three process-group-isolated workers, waits for all, and returns nonzero on any failure.
- [ ] Invoke the phase function for K=15, then K=5.
- [ ] Run the focused test and confirm it passes.

### Task 3: Verify and hand off

**Files:**
- Verify only.

- [ ] Run `bash -n scripts/run_k15_then_k5_3gpu_seed50_61.sh`.
- [ ] Run `DRY_RUN=1 bash scripts/run_k15_then_k5_3gpu_seed50_61.sh`.
- [ ] Run the new focused pytest file.
- [ ] Run existing multi-K sweep and E2E shell-entry tests.
- [ ] Run `git diff --check` and inspect status for unintended changes.

