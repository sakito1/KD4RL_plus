# Downside Controller NAS50 SH91 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a controller tuning path for NAS seed 50 and SH seed 91 that rewards return improvement mainly through avoided downside risk, keeps policy-gradient reward dominant, and limits switches only through penalties above 30 switches.

**Architecture:** Extend controller counterfactual statistics with cumulative downside loss and add a `relative_downside_mdd` reward mode. Wire the new downside coefficient through `run_hrl_training.py`, keep switch limits soft by setting `controller_hard_max_switches=0` in the new script, and create a dedicated shell script for NAS-50/SH-91 tuning.

**Tech Stack:** Python, PyTorch controller PG training, Bash training scripts, unittest.

---

### Task 1: Add Downside-Aware Controller Reward

**Files:**
- Modify: `Train/controller_pg.py`
- Modify: `Train/PPO_train.py`
- Modify: `run_hrl_training.py`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_run_hrl_training_command.py`

- [ ] Write a failing test that `relative_downside_mdd` combines relative return uplift, relative downside-loss improvement, relative MDD improvement, and normalized switch overflow.
- [ ] Write a failing test that `_counterfactual_stats` records cumulative downside loss from negative daily log returns.
- [ ] Write a failing test that `--controller_downside_coef` and `--controller_reward_mode relative_downside_mdd` are forwarded to child commands.
- [ ] Implement `CounterfactualStats.downside_loss`, compute it from portfolio history, add `downside_coef`, and pass it into `controller_reward`.
- [ ] Run `python -m unittest discover -s tests -p 'test_controller_counterfactual_pg.py'`.
- [ ] Run `python -m unittest discover -s tests -p 'test_run_hrl_training_command.py'`.

### Task 2: Add NAS50/SH91 Controller Tuning Script

**Files:**
- Create: `train_sh/run_controller_downside_uplift_nas50_sh91.sh`
- Modify: `train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh`
- Test: `tests/test_controller_daily_aux_pg_script.py`

- [ ] Write a failing script test that the new script defaults to `NAS_SEEDS=50`, `SH_SEEDS=91`, `CONTROLLER_HARD_MAX_SWITCHES=0`, `CONTROLLER_MAX_SWITCHES=30`, `CONTROLLER_MAX_SWITCH_PENALTY_COEF=0.1`, `CONTROLLER_REWARD_MODE=relative_downside_mdd`, `CONTROLLER_RETURN_COEF=1.0`, `CONTROLLER_DOWNSIDE_COEF=0.5`, `CONTROLLER_MDD_COEF=0.3`, and zero separate switch losses so the policy-gradient reward remains dominant.
- [ ] Add `CONTROLLER_DOWNSIDE_COEF` plumbing in the shared controller script.
- [ ] Add the dedicated wrapper script that execs the shared script with NAS-50/SH-91 defaults.
- [ ] Run `python -m unittest discover -s tests -p 'test_controller_daily_aux_pg_script.py'`.
- [ ] Run `bash -n train_sh/run_controller_downside_uplift_nas50_sh91.sh train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh`.

### Task 3: Final Verification

**Files:**
- Verify only.

- [ ] Run targeted unit tests for controller reward, command forwarding, and scripts.
- [ ] Run `python -m py_compile Train/controller_pg.py Train/PPO_train.py run_hrl_training.py`.
- [ ] Print the final command for the user.
