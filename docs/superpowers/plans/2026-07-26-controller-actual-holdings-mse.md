# Controller Actual-Holdings MSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Train the Controller with MSE-supervised risk and advantage signals that
are aligned to the pre-decision actual drifted portfolio, using Controller-sampled
supervision trajectories and one non-duplicative `sup_pg` shell command.

**Architecture:** Keep the dual-branch `MonitorAC` parameter shapes and the real
Inner Actor execution path unchanged. The Controller's supervision view uses the
pre-decision `weights_drift` on the Hold side and the candidate after deterministic
Inner execution on the Switch side; environment execution may still run the daily
Inner Actor after a Hold action. Collect one 12-window Controller rollout batch,
then replay it 30 times before three PG epochs.

**Tech Stack:** Python 3.10, PyTorch, Bash, pytest.

---

### Task 1: Align Risk target with the actual drifted portfolio

**Files:**
- Modify: `env/PPO_env.py:505-519`
- Test: `tests/test_actor_score_supervision.py`

- [ ] **Step 1: Write a failing environment test**

Create a test in which `prev_weights` and `prev_base_weight` differ, call
`env.step(...)`, and assert that `controller_hold_return_target` and
`controller_hold_mdd_target` equal:

```python
r_past = env.ratio[:, env.day - 1]
actual_drift = env._normalize(env.prev_weights_before_step * r_past)
expected_return, expected_mdd = (
    env._future_portfolio_return_and_relative_market_drawdown(
        actual_drift,
        env.day_before_step,
        env.max_hold - env.t_held_before_step,
    )
)
```

Also compute the active-base target and assert it differs, proving the test catches
the old `prev_base_weight` behavior.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_actor_score_supervision.py \
  -k controller_aux_targets_use_actual_drifted_holdings
```

Expected: FAIL because the environment currently constructs
`current_holdings_drift` from `prev_base_weight`.

- [ ] **Step 3: Implement the minimal target correction**

In `PPO_Env.step`, replace:

```python
current_holdings_drift = self._normalize(self.prev_base_weight * r_past)
```

with:

```python
current_holdings_drift = self._normalize(self.prev_weights * r_past)
```

Do not change the remaining-horizon or relative-market drawdown definitions.

- [ ] **Step 4: Run the target test and existing environment tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_actor_score_supervision.py
```

Expected: PASS.

### Task 2: Align Advantage features and target with pre-decision holdings

**Files:**
- Modify: `Train/PPO_train.py:891-927`
- Modify: `Train/PPO_train.py:1310-1465`
- Modify: `Train/PPO_train.py:1660-1740`
- Modify: `agent/PPO_agent.py:410-520`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_controller_dual_branch.py`

- [ ] **Step 1: Write failing Advantage target tests**

Replace the old Inner-Hold comparison assertion with a test using:

```python
obs = {"weights_drift": torch.tensor([[0.50, 0.50]])}
switch_exec = torch.tensor([[0.20, 0.80]])
advantage = trainer._controller_actual_holdings_switch_advantage(
    env,
    obs,
    switch_exec,
)
```

The expected value must be:

```python
actual = env._normalize(obs["weights_drift"].flatten())
switch = env._normalize(switch_exec.flatten())
hold_ret, _ = env._future_portfolio_return_and_max_drawdown(actual, env.day, 20)
switch_ret, _ = env._future_portfolio_return_and_max_drawdown(switch, env.day, 20)
switch_cost = torch.sum(torch.abs(switch - actual)) * env.transaction_cost_pct
expected = switch_ret - hold_ret - switch_cost
```

Add a record-capture assertion proving `hold_exec_weights` stored for Controller
replay equals `obs["weights_drift"]`, not a new Inner Hold preview.

- [ ] **Step 2: Run the Advantage tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_controller_counterfactual_pg.py \
  -k 'actual_holdings_switch_advantage or controller_record_uses_actual_holdings'
```

Expected: FAIL because the helper currently compares `hold_exec` with
`switch_exec`.

- [ ] **Step 3: Implement an explicit actual-holdings helper**

Replace `_controller_inner_adjusted_switch_advantage` with:

```python
def _controller_actual_holdings_switch_advantage(self, env, obs, switch_exec):
    actual = obs["weights_drift"].detach().view(-1)
    actual = env._normalize(actual)
    switch = env._normalize(switch_exec.detach().view(-1))
    max_hold = int(getattr(env, "max_hold", self.cfg.max_hold))
    horizon = max(1, max_hold - int(getattr(env, "t_held", 0)))
    start_day = int(getattr(env, "day", 0))
    hold_return, _ = env._future_portfolio_return_and_max_drawdown(
        actual, start_day, horizon
    )
    switch_return, _ = env._future_portfolio_return_and_max_drawdown(
        switch, start_day, horizon
    )
    switch_cost = torch.sum(torch.abs(switch - actual))
    return (
        switch_return
        - hold_return
        - switch_cost * float(getattr(env, "transaction_cost_pct", 0.0))
    )
```

Use the same normalization fallback already present when an environment lacks
`_normalize`.

- [ ] **Step 4: Separate Controller supervision weights from actual execution**

In Controller rollout collection:

```python
actual_hold = weights_drift.detach()
inner_hold_exec = self._deterministic_inner_exec(
    obs, obs["base_drift"].detach(), weights_drift
)
switch_exec = self._controller_exec_weights(...)
```

Then:

- pass `actual_hold` as `hold_exec_weights` to `MonitorAC.decision_stats`;
- compute the Advantage target from `actual_hold` and `switch_exec`;
- store `actual_hold` in the replay record;
- continue using `inner_hold_exec` as the real environment execution when the
  sampled action is Hold.

Apply the same Controller feature/target alignment in auxiliary collection.

- [ ] **Step 5: Align inference features without disabling daily Inner execution**

In `HRL_PPO_Agent.get_action`, set:

```python
controller_hold_exec = weight_drift.detach()
controller_switch_exec = self._preview_inner_exec(
    obs,
    act_out,
    weight_drift,
    force_inner_zero=force_inner_zero,
)
```

Continue running the normal Inner execution path after the Controller selects its
base. This changes only Controller features, not portfolio execution.

- [ ] **Step 6: Run Advantage and dual-branch tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_dual_branch.py
```

Expected: PASS.

### Task 3: Replace Controller Huber regression with scaled MSE

**Files:**
- Modify: `Train/PPO_train.py:1970-2005`
- Modify: `run_hrl_training.py:480-490`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_run_hrl_training_command.py`

- [ ] **Step 1: Change regression tests to require MSE**

For Risk and Advantage test records, assert:

```python
expected_risk = F.mse_loss(risk_pred, raw_risk * 20.0)
expected_adv = F.mse_loss(adv_pred, raw_advantage * 20.0)
```

Use values with absolute error below one so MSE and Smooth-L1 are observably
different.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_controller_counterfactual_pg.py \
  -k 'risk_mse or advantage_mse'
```

Expected: FAIL because both branches currently call `F.smooth_l1_loss`.

- [ ] **Step 3: Implement direct MSE**

Replace only the Risk and Advantage loss calls with:

```python
F.mse_loss(stats["hold_risk_pred"].view(-1), target_mdd.view(-1))
F.mse_loss(pred_switch_adv, target_switch_adv)
```

Retain target scales of `20.0`. Leave Outer/Inner supervision losses unchanged.

- [ ] **Step 4: Make the CLI describe the actual loss**

Change `--controller_aux_switch_adv_loss_type` to accept/default to `mse`, and
pass `mse` from current training scripts. The training implementation remains a
single direct MSE path; the argument exists only for configuration reporting and
rejects obsolete values.

- [ ] **Step 5: Run regression and command tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_controller_counterfactual_pg.py \
  tests/test_run_hrl_training_command.py
```

Expected: PASS.

### Task 4: Make supervision collection on-policy and simplify the shell script

**Files:**
- Modify: `train_sh/explore_controller_from_nas45_outer_inner.sh`
- Modify: `tests/test_explore_controller_from_nas45_outer_inner_script.py`

- [ ] **Step 1: Replace multi-mode tests with one formal-command test**

Run the script with:

```python
env.update({
    "DRY_RUN": "1",
    "PYTHON_BIN": "/bin/echo",
    "OUTPUT_ROOT": "results/test_nas45_controller_exploration",
})
```

Assert the output contains:

```text
--controller_sup_pretrain_epochs 1
--controller_aux_replay_epochs 30
--controller_epochs 3
--controller_use_switch_supervision
--controller_aux_switch_adv_loss_type mse
```

Assert it does not contain:

```text
--controller_aux_pretrain_offpolicy
--controller_pretrain_only
--controller_guidance_probe_only
```

Read the shell source and assert it contains no `MODE`, `case "$MODE"`, `probe`,
`pg_only`, or `sup_only`.

- [ ] **Step 2: Run the script test and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_explore_controller_from_nas45_outer_inner_script.py
```

Expected: FAIL because the script still contains multiple modes and explicitly
enables off-policy auxiliary collection.

- [ ] **Step 3: Simplify the script to one `sup_pg` command**

Remove `MODE`, the mode `case`, `MODE_ARGS`, and repeated temporary coefficient
assignments. Write the final values once in `CMD`:

```bash
--controller_epochs 3
--controller_sup_coef 0.1
--controller_sup_pretrain_epochs 1
--controller_aux_replay_epochs 30
--controller_aux_mdd_coef 0.1
--controller_aux_switch_adv_coef 0.1
--controller_aux_switch_adv_loss_type mse
--controller_use_switch_supervision
```

Do not pass `--controller_aux_pretrain_offpolicy`; this makes
`_run_controller_aux_pretrain_windows` select Controller rollouts.

Keep `CONTROLLER_SEED`, `GPU_ID`, `DRY_RUN`, `ALLOW_EXISTING_OUTPUT`,
`PYTHON_BIN`, `OUTPUT_ROOT`, `SOURCE_CHECKPOINT`, `RUN_NAME`, and
`HEARTBEAT_SECONDS` as environment-overridable values.

- [ ] **Step 4: Verify shell syntax and dry-run output**

Run:

```bash
bash -n train_sh/explore_controller_from_nas45_outer_inner.sh
DRY_RUN=1 PYTHON_BIN=/bin/echo \
  bash train_sh/explore_controller_from_nas45_outer_inner.sh
```

Expected: syntax succeeds and exactly one formal `sup_pg` command is printed.

### Task 5: Full verification

**Files:**
- Verify only; do not commit.

- [ ] **Step 1: Run focused tests**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_actor_score_supervision.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_dual_branch.py \
  tests/test_explore_controller_from_nas45_outer_inner_script.py \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py
```

Expected: all pass.

- [ ] **Step 2: Run all collectable repository tests**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  --ignore=tests/test_final_model_module_coordination.py \
  --ignore=tests/test_paper_experiments.py \
  --ignore=tests/test_paper_figure_readability.py \
  --ignore=tests/test_population_level_mechanism_analysis.py
```

Expected: all pass. The four ignored tests require the already-missing
`paper_experiments` package and are unrelated to Controller training.

- [ ] **Step 3: Inspect the final command**

Confirm the dry-run contains one 12-window, 300-day Controller-sampled
supervision collection, 30 auxiliary replay updates, three PG epochs, target
scales of 20, MSE reporting, daily decisions, and 30-day forced switching.

No Git commit is created because the user explicitly requested local changes
only.
