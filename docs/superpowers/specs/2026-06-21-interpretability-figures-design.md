# Interpretability Figures Design

## Goal

Create paper-ready interpretability figures and conclusion text for the final archived Controller+HRL models in `results/end`.

The output should support two claims:

1. The inner actor is useful because, under the same fixed 30-day outer schedule, enabling inner weight adjustment improves returns and Sharpe over the no-inner variant.
2. The controller is useful because it replaces a rigid fixed rebalance schedule with state-dependent switching. The explanation should show when it switches and which controller signals move before those switches.

## Scope

Use the two archived final runs:

- SH seed 90: `results/end/sh_seed90/checkpoints/best_model.pth`
- NAS seed 49: `results/end/nas_seed49/checkpoints/best_model.pth`

Use existing test metrics from the archived logs for the main ablation summary:

- Scenario 1: fixed 30-day HRL with inner, no controller.
- Scenario 2: fixed 30-day outer, no controller and no inner.
- Scenario 3: full model with controller and inner.

Add one reproducible probe script that reruns test episodes with the archived checkpoints and records per-day controller diagnostics. The probe should not retrain models.

## Figures

### Figure 1: Module Ablation Summary

Grouped bars for SH and NAS:

- Total return
- Sharpe
- Maximum drawdown
- Switch count or free switch count

This figure establishes module effectiveness. Inner effectiveness is read from Scenario 1 versus Scenario 2. Controller effectiveness is read from Scenario 3 versus Scenario 1.

### Figure 2: Equity Curve Comparison

Line plot per market comparing:

- Fixed HRL with inner, no controller
- No inner, no controller
- Full Controller+HRL

Use test cumulative portfolio value normalized to 1 at the start. If archived CSVs are missing from `results/end`, regenerate them with the archived checkpoints.

### Figure 3: Controller Decision Timeline

For each market, plot the full-model test equity curve with controller switch days overlaid. Under it, plot daily `exit_prob`.

This should show that switches are sparse state-dependent events rather than a fixed calendar rhythm.

### Figure 4: Controller Signal Relationships

Scatter or binned plots:

- `exit_prob` versus current segment drawdown
- `exit_prob` versus current segment return
- `exit_prob` versus predicted switch advantage
- Optional: `exit_prob` versus hold duration

Use correlation annotations. The claim is qualitative: the controller reacts to portfolio state and candidate-switch advantage, not only elapsed time.

### Figure 5: Switch Event Study

For free controller switches, align days around each switch and plot mean cumulative return from `-10` to `+20` trading days. Include confidence bands if enough events exist.

This figure explains what the controller is doing around switch points: it should reveal whether switches tend to happen after local deterioration and whether post-switch behavior stabilizes or improves.

### Optional Figure 6: Lightweight Sensitivity

Evaluate `exit_prob` sensitivity on recorded controller states by perturbing one interpretable input at a time:

- Increase segment drawdown.
- Reduce segment return.
- Increase predicted switch advantage if directly available through the learned head.

Only include this figure if it produces stable, easy-to-explain behavior. Otherwise keep it as appendix data.

## Data Collection

The probe script should create an output directory under `results/end/interpretability/` with:

- `ablation_metrics.csv`
- `controller_trace_<market>.csv`
- `switch_event_<market>.csv`
- PNG or PDF figures suitable for paper insertion.
- `paper_conclusions_zh.md` containing concise Chinese interpretation text.

`controller_trace_<market>.csv` should include at least:

- date
- portfolio value
- daily return
- action: hold or switch
- free or forced switch flag
- hold duration
- `exit_prob`
- `base_exit_logit`
- `switch_advantage_pred`
- `hold_return_pred`
- `hold_risk_pred`
- current segment return
- current segment drawdown
- turnover between current hold portfolio and candidate switch portfolio
- overlap between current hold portfolio and candidate switch portfolio

## Implementation Notes

Prefer adding a standalone script rather than changing training behavior. The script can reuse `run_hrl_training.py`, `PPO_Env`, `HRL_Networks`, `HRL_PPO_Agent`, and `HRL_Trainer` setup code where practical, but it should keep the experiment isolated.

If importing the full trainer setup is too brittle, parse archived logs for Figure 1 and implement a focused evaluation loop modeled on `HRL_Trainer.run_episode` for Figures 2 to 5.

The script should run on CPU by default, with an optional `--device` flag. It must not require CUDA for figure generation.

## Testing And Verification

Verification should include:

- The script runs for both archived markets without training.
- Output CSVs contain nonempty controller traces.
- Figure 1 reproduces the archived metrics:
  - SH Scenario 1 total return 158.99%, Scenario 2 147.05%, Scenario 3 204.99%.
  - NAS Scenario 1 total return 227.43%, Scenario 2 220.42%, Scenario 3 265.53%.
- The controller traces contain nonzero switch events.
- Generated figures are valid image files and visually inspectable.

## Paper Conclusion Shape

The Chinese conclusion should be concise and claim-bounded:

- Inner actor: improves performance by optimizing weights inside the outer-selected base portfolio.
- Controller: improves performance mainly by learning dynamic rebalance timing, increasing useful free switches compared with fixed 30-day rebalancing.
- Mechanism: controller exit probability is tied to current segment quality, drawdown/return state, candidate-switch difference, and predicted switch advantage.
- Caveat: these are model-level interpretability probes on held-out test periods, not causal proof of market mechanism.
