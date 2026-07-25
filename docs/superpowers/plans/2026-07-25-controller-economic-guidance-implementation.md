# Controller Economic Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Replace the exploratory Top-20 probe with train-time 5% economic-event guidance using market-relative risk, masked event-onset labels, balanced BCE, and stage-specific loss weights.

**Architecture:** The environment computes a dense market-relative risk target alongside the existing frozen-path returns. Pure helpers in `Train/controller_guidance.py` convert chronological risk/advantage targets into onset labels, masks, and class-balancing weights. The trainer annotates collected decision segments and reuses the existing supervised policy-logit path, adding the label term to auxiliary pretraining with coefficient 1.0 and to PG with coefficient 0.1.

**Tech Stack:** Python, PyTorch, pandas, pytest.

---

### Task 1: Market-relative risk target

**Files:**
- Modify: `env/PPO_env.py`
- Modify: `tests/test_actor_score_supervision.py`

- [ ] Add a failing test with a two-asset path where the held asset loses relative wealth against the equal-weight pool while the market itself remains positive.
- [ ] Run `pytest -q tests/test_actor_score_supervision.py -k relative_market` and confirm failure because the relative-risk helper is absent.
- [ ] Add `_future_portfolio_return_and_relative_market_drawdown(weights, start_day, horizon)` using equal-weight buy-and-hold cumulative asset paths.
- [ ] Change `controller_hold_mdd_target` construction to use the new relative target while preserving the existing return target.
- [ ] Run the focused test and the full `tests/test_actor_score_supervision.py`.

### Task 2: Economic event labels and balanced weights

**Files:**
- Modify: `Train/controller_guidance.py`
- Modify: `tests/test_controller_guidance.py`

- [ ] Add failing tests for the exact rule `(risk >= 0.05 and advantage > 0) or advantage >= 0.05`.
- [ ] Add failing tests showing that only the first date of a consecutive trigger run is positive and later dates are masked.
- [ ] Add a failing test showing that separate decision segments reset event continuity.
- [ ] Add a failing test verifying positive and negative samples receive equal total weight.
- [ ] Run the focused tests and confirm each fails for missing economic-label behavior.
- [ ] Implement `build_economic_guidance_labels` and `balanced_guidance_weights` as pure tensor helpers.
- [ ] Run `pytest -q tests/test_controller_guidance.py`.

### Task 3: Annotate Controller decision segments

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `tests/test_controller_counterfactual_pg.py`

- [ ] Add a failing trainer test that supplies two segments and checks `sup_label`/`sup_weight`, including masked continuation dates.
- [ ] Run the focused test and confirm failure because records are not annotated.
- [ ] Implement a trainer helper that applies economic labels per segment, computes class weights over the full episode, and writes `sup_label` and `sup_weight` tensors.
- [ ] Split fixed auxiliary records at each forced revision so event continuity resets after the portfolio changes.
- [ ] Apply annotation before PG and auxiliary window results are returned.
- [ ] Run the focused test and the full Controller PG test module.

### Task 4: Stage-specific loss composition

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `tests/test_controller_counterfactual_pg.py`

- [ ] Add failing tests that auxiliary pretraining includes `1.0 * label + 0.1 * risk + 1.0 * advantage`.
- [ ] Add failing tests that PG includes `0.1 * label + 0.1 * risk + 1.0 * advantage - 0.01 * entropy`.
- [ ] Confirm focused failures are caused by the missing auxiliary label term and missing stage-specific coefficient.
- [ ] Extend auxiliary replay aggregation to carry supervised losses and apply `controller_guidance_pretrain_coef`.
- [ ] Keep PG supervised aggregation on the existing policy logit and apply `controller_sup_coef=0.1`.
- [ ] Extend diagnostics and Controller auxiliary logs with raw and weighted label loss.
- [ ] Run the focused tests and full Controller PG tests.

### Task 5: Configuration and command forwarding

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `utils/config.py`
- Modify: `utils/config_Nas.py`
- Modify: `utils/config_SH.py`
- Modify: `train_sh/sh_77.sh`
- Modify: `tests/test_run_hrl_training_command.py`

- [ ] Add failing command/config tests for risk threshold `0.05`, advantage threshold `0.05`, pretrain coefficient `1.0`, PG coefficient `0.1`, risk coefficient `0.1`, advantage coefficient `1.0`, entropy coefficient `0.01`, and 300-day rollout/pretrain windows.
- [ ] Run focused tests and confirm the new arguments are missing.
- [ ] Add CLI parsing, child forwarding, runtime configuration, and metadata for the guidance settings.
- [ ] Update default configs and `train_sh/sh_77.sh`; do not add a new launcher.
- [ ] Remove Top-20 guidance from the active training path while retaining no compatibility branch for old labels.
- [ ] Run command/config tests.

### Task 6: Verification

**Files:**
- Verify all modified files.

- [ ] Run `python -m py_compile env/PPO_env.py Train/controller_guidance.py Train/PPO_train.py run_hrl_training.py`.
- [ ] Run `pytest -q tests/test_controller_guidance.py tests/test_actor_score_supervision.py tests/test_controller_counterfactual_pg.py tests/test_controller_dual_branch.py tests/test_run_hrl_training_command.py`.
- [ ] Inspect `git diff --check` and `git diff --stat`.
- [ ] Confirm no training, validation, test, checkpoint, or result artifact was modified.
- [ ] Do not commit or push; report the exact verification output and the command the user can run.

