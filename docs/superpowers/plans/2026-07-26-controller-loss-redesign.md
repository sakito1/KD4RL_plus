# Controller Loss Redesign Implementation Plan

> **Scope:** Modify the current local workspace only. Do not create commits or
> change the saved Outer+Inner checkpoint.

**Goal:** Make the Controller learn interpretable risk and candidate-advantage
signals while keeping switch decisions sparse (5%--15%) and retaining the
trajectory-level PG objective.

**Architecture:** Keep the existing dual-branch `MonitorAC` tensor shapes.
Replace classification-style advantage supervision with continuous Huber
regression, use ordinary BCE for the economic switch label, aggregate PG
log-probability over all free daily decisions, and replace the soft expected-rate
band with a batch-level Top-Tail logit margin. Remove Controller dropout so
rollout collection and replay evaluate the same policy distribution.

**Tech stack:** Python, PyTorch, pytest, Bash.

---

### Task 1: Add behavior tests for the approved losses

**Files:**
- Modify: `tests/test_controller_counterfactual_pg.py`
- Modify: `tests/test_controller_dual_branch.py`
- Modify: `tests/test_explore_controller_from_nas45_outer_inner_script.py`

1. Replace old soft-rate tests with tests asserting that Top-Tail loss:
   - is zero when the top 5% exceed `+margin` and logits after 15% stay below
     `-margin`;
   - gives gradients to violating upper and lower tails;
   - uses logits rather than temperature-smoothed probabilities.
2. Add a two-segment example proving mean PG reduction is the mean over all
   daily free decisions rather than the mean of segment means.
3. Add an imbalanced economic-label example proving valid samples receive unit
   weight (ordinary BCE).
4. Add a model test proving the Controller contains no active dropout.
5. Update the exploration-script assertions to require `smooth_l1`, target
   scales of `20`, and PG auxiliary/rate coefficients of `0.1`.
6. Run the targeted tests and confirm failures correspond to the old behavior.

### Task 2: Implement the simplified Controller losses

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `Components/PPO_model.py`

1. Replace `_controller_switch_rate_band_loss` with a Top-Tail margin helper
   operating on policy logits.
2. Collect policy logits from every free decision, compute the rate constraint
   once over the full multi-window batch, and apply:
   - coefficient `1.0` during auxiliary pretraining;
   - configured coefficient `0.1` during PG.
3. Restrict advantage auxiliary loss to continuous Smooth-L1/Huber regression.
4. Replace balanced economic-label weights with unit weights for every valid
   label.
5. Change mean log-prob reduction to the mean of every daily free-decision
   log-probability.
6. Replace Controller dropout layers with `Identity`; this preserves checkpoint
   parameter shapes because dropout has no trainable state.

### Task 3: Align command-line configuration

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `train_sh/explore_controller_from_nas45_outer_inner.sh`
- Modify dependent command/script tests only where the removed options are
  asserted.

1. Expose `controller_switch_rate_margin` and stop forwarding the obsolete
   threshold/temperature controls.
2. Configure risk and advantage target scales to `20`.
3. Configure risk, advantage, switch BCE, and Top-Tail coefficients to `0.1`
   during PG; auxiliary pretraining uses the existing
   `controller_guidance_pretrain_coef=1.0` for all four terms.
4. Keep rollout length, fixed training windows, checkpoint paths, seed, and
   Outer+Inner loading unchanged.

### Task 4: Verify the implementation

**Files:**
- Verify only; no commits.

1. Run the focused Controller, CLI, and shell-script tests.
2. Run the broader existing Controller test file to catch tuple/diagnostic
   regressions.
3. Execute the exploration script in command-print/dry-run mode and inspect the
   resolved arguments.
4. Report the modified files, test evidence, and the exact training command.
