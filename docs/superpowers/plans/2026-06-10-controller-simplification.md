# Controller Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Simplify the HRL controller to a distributional alpha-state comparator trained by relative-return PG with max-switch overflow penalty, while keeping outer actor score supervision and minimally changing HRL training.

**Architecture:** Keep the existing `MonitorAC` API so `HRL_PPO_Agent` and `HRL_Trainer` call sites stay mostly unchanged. Replace its internals with hold/switch embedding pooling, Gaussian switch-advantage probability, and a dynamic threshold. Simplify controller reward to return uplift minus max-switch overflow penalty, and select inner checkpoints by validation return.

**Tech Stack:** PyTorch, existing unittest-style tests, existing HRL training scripts.

---

### Task 1: Controller Reward Tests

**Files:**
- Modify: `tests/test_controller_counterfactual_pg.py`
- Modify: `Train/controller_pg.py`

- [ ] Write tests for return-uplift-only reward and max-switch overflow penalty.
- [ ] Run the test to confirm it fails on old behavior.
- [ ] Implement simplified reward helper.
- [ ] Re-run the test.

### Task 2: Distributional Controller Tests

**Files:**
- Create: `tests/test_distributional_alpha_controller.py`
- Modify: `Components/PPO_model.py`

- [ ] Write tests that `MonitorAC` computes `p_adv`, `tau`, and Bernoulli logits from alpha-state embeddings and portfolio weights.
- [ ] Run the test to confirm old `MonitorAC` lacks diagnostics/probability behavior.
- [ ] Replace `MonitorAC` internals without changing public call signatures.
- [ ] Re-run the test.

### Task 3: Training Wiring

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `run_hrl_training.py`

- [ ] Use the simplified controller reward in PG windows with `max_allowed_switches = rollout_len // min_hold` by default.
- [ ] Log return uplift, actual switches, max allowed switches, and overflow.
- [ ] Select warmup inner checkpoints by validation return.
- [ ] Set default `joint_lr_mult` to `0.001` for `1e-6` joint LR when base LR is `1e-3`.

### Task 4: Verification

**Files:**
- Run tests and py_compile.

- [ ] `python tests/test_controller_counterfactual_pg.py`
- [ ] `python tests/test_distributional_alpha_controller.py`
- [ ] `python tests/test_actor_score_supervision.py`
- [ ] `python -m py_compile Components/PPO_model.py Train/controller_pg.py Train/PPO_train.py run_hrl_training.py`

