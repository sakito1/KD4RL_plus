# Controller Advantage Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Align Controller advantage inputs with the Inner-adjusted execution weights used by its target and train the advantage head as magnitude-weighted sign classification.

**Architecture:** Extend `MonitorAC.decision_stats` with optional hold/switch execution weights and normalized remaining horizon. Store those immutable inputs in Controller replay records, then compute weighted BCE from the existing counterfactual advantage target.

**Tech Stack:** PyTorch, pytest.

---

### Task 1: Specify aligned Controller inputs

**Files:**
- Modify: `tests/test_controller_dual_branch.py`
- Modify: `Components/PPO_model.py`

- [ ] Add a failing test that passes different `hold_exec_weights`,
  `switch_exec_weights`, and `remaining_horizon` values to `decision_stats`.
- [ ] Run the focused test and confirm failure due to unsupported arguments.
- [ ] Extend the advantage feature construction while leaving the risk feature unchanged.
- [ ] Run the focused test and confirm it passes.

### Task 2: Preserve aligned inputs in replay records

**Files:**
- Modify: `tests/test_controller_counterfactual_pg.py`
- Modify: `Train/PPO_train.py`

- [ ] Add a failing test asserting `_detach_controller_record` stores both execution
  branches and normalized remaining horizon.
- [ ] Run the test and confirm the new fields are absent.
- [ ] Pass the precomputed execution weights through fixed-window and PG collection,
  then consume them in `_controller_episode_terms`.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Replace advantage magnitude regression

**Files:**
- Modify: `tests/test_controller_counterfactual_pg.py`
- Modify: `Train/PPO_train.py`
- Modify: `train_sh/explore_controller_from_nas45_outer_inner.sh`

- [ ] Add a failing numerical test for sign BCE weighted by normalized
  `abs(switch_advantage)`.
- [ ] Run it and confirm the existing Smooth-L1 path fails the expectation.
- [ ] Implement the weighted classification loss and select it in the NAS45 launcher.
- [ ] Run the focused loss and launcher tests.

### Task 4: Regression verification

**Files:**
- Verify: `Components/PPO_model.py`
- Verify: `Train/PPO_train.py`
- Verify: `train_sh/explore_controller_from_nas45_outer_inner.sh`

- [ ] Run:

```bash
pytest -q \
  tests/test_controller_dual_branch.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_explore_controller_from_nas45_outer_inner_script.py
```

- [ ] Run `bash -n train_sh/explore_controller_from_nas45_outer_inner.sh`.
- [ ] Inspect the dry-run command and report the new run command without launching
  training.

No Git commit is performed because the user requested local changes only.
