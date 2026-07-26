# Controller Switch-Rate Band Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Constrain the fraction of free Controller decisions whose probability exceeds 0.5 to approximately 5%--15%, without suppressing all learned switches.

**Architecture:** Compute a differentiable approximation of the hard threshold rate from free-decision probabilities only. Add a two-sided hinge penalty to both supervised replay and policy-gradient updates, expose its parameters through the existing CLI/config flow, and log the soft rate and weighted penalty.

**Tech Stack:** Python, PyTorch, pytest, Bash.

---

### Task 1: Define and test the rate-band loss

**Files:**
- Modify: `Train/PPO_train.py`
- Test: `tests/test_controller_counterfactual_pg.py`

- [ ] Add tests showing that the loss is zero inside 5%--15%, positive above and below the band, and differentiable with the correct gradient direction.
- [ ] Run the focused tests and verify that they fail because the helper does not exist.
- [ ] Add `_controller_switch_rate_band_loss()` using a sigmoid approximation around the 0.5 decision threshold and a two-sided linear hinge.
- [ ] Re-run the focused tests and verify that they pass.

### Task 2: Connect the loss to Controller updates

**Files:**
- Modify: `Train/PPO_train.py`
- Test: `tests/test_controller_counterfactual_pg.py`

- [ ] Add a failing test showing that `_controller_episode_terms()` returns a switch-rate loss based only on decision-record probabilities.
- [ ] Carry the switch-rate loss through supervised replay aggregation and `_update_controller_aux_batch()`.
- [ ] Carry the same loss through `_update_controller_pg_batch()`.
- [ ] Add `switch_rate`, `switch_rate_loss`, and `switch_rate_weighted_loss` diagnostics.
- [ ] Re-run the focused Controller tests.

### Task 3: Expose and enable the approved configuration

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `utils/config.py`
- Modify: `utils/config_Nas.py`
- Modify: `utils/config_SH.py`
- Modify: `train_sh/explore_controller_from_nas45_outer_inner.sh`
- Test: `tests/test_explore_controller_from_nas45_outer_inner_script.py`
- Test: `tests/test_run_hrl_training_command.py`

- [ ] Add CLI/config fields for coefficient `5.0`, lower bound `0.05`, upper bound `0.15`, threshold `0.5`, and temperature `0.05`.
- [ ] Validate `0 <= min <= max <= 1` and positive temperature when constructing runtime configuration.
- [ ] Enable the rate band in the NAS seed-45 exploration script and reduce the entropy coefficient from `0.01` to `0.001`.
- [ ] Extend dry-run command tests to verify all arguments.

### Task 4: Verify behavior and regressions

**Files:**
- Verify only; no Git commit.

- [ ] Run the Controller unit tests.
- [ ] Run the command/config tests.
- [ ] Run the broader related test subset.
- [ ] Confirm the dry-run command contains the approved 5%--15% band and no Outer/Inner training changes.

