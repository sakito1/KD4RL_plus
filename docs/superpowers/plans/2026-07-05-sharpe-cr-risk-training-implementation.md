# Sharpe/CR Risk Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-tdd while implementing each behavior. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add the minimum-intrusion reward modes needed for the Sharpe/CR risk-enhanced seed sweep.

**Architecture:** Keep the existing end-to-end training flow intact. Add opt-in reward modes at the existing controller reward and outer segment aggregation boundaries, then expose them through `run_hrl_training.py` and a small seed-sweep wrapper around the current e2e script.

**Tech Stack:** Python, PyTorch, unittest/pytest, bash training scripts.

---

### Task 1: Controller Relative CR Reward

**Files:**
- Modify: `Train/controller_pg.py`
- Test: `tests/test_controller_counterfactual_pg.py`

- [x] Add tests showing `controller_reward(..., reward_mode="return_uplift")` keeps current behavior.
- [x] Add tests showing `controller_reward(..., reward_mode="relative_cr")` computes `CR(controlled) - CR(baseline) - normalized_switch_penalty`.
- [x] Implement `CounterfactualStats.calmar_ratio` and `controller_reward` mode dispatch.
- [x] Run `python -m unittest discover -s tests -p 'test_controller_counterfactual_pg.py'`.

### Task 2: Outer Segment Sharpe Reward

**Files:**
- Modify: `agent/PPO_agent.py`
- Test: `tests/test_run_hrl_training_command.py`

- [x] Add a focused buffer test showing `outer_reward_mode="segment_sharpe"` uses segment daily returns to compute annualized Sharpe.
- [x] Implement opt-in `outer_reward_mode` on `HRL_Buffer`, preserving default return aggregation.
- [x] Run the focused test.

### Task 3: Argument Plumbing

**Files:**
- Modify: `run_hrl_training.py`
- Test: `tests/test_run_hrl_training_command.py`

- [x] Add parser args `--outer_reward_mode` and `--controller_reward_mode`.
- [x] Include both args in child command construction and runtime config.
- [x] Ensure `HRL_Buffer` receives `outer_reward_mode`.
- [x] Pass `controller_reward_mode` into controller reward calls.
- [x] Run `python -m unittest discover -s tests -p 'test_run_hrl_training_command.py'`.

### Task 4: Risk Seed Sweep Script

**Files:**
- Create: `train_sh/run_end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep.sh`
- Test: create or extend script echo tests.

- [x] Reuse the current e2e script through a thin wrapper.
- [x] Use isolated output root and expanded seeds.
- [x] Set `--outer_reward_mode segment_sharpe`, `--controller_reward_mode relative_cr`, and return-based selection metrics.
- [x] Preserve protected output checks through the delegated e2e script.
- [x] Run script echo test with `PYTHON_BIN=/bin/echo`.

### Verification

- [x] Run targeted tests for controller reward, command plumbing, and scripts.
- [x] Re-run old e2e script echo test to confirm old flow is unchanged.
