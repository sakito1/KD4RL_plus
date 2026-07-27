# Controller Adaptive-Timing Statistical Validation Design

Date: 2026-07-27

## Objective

Evaluate whether the learned Controller makes economically meaningful adaptive
`switch`/`hold` decisions without repeating the existing learned-versus-fixed-day
comparison.

The analysis must answer two questions:

1. Does the Controller choose the better action at free decision points?
2. Does the Controller's exit propensity rank states in which switching has
   greater counterfactual value?

The analysis is explanatory. It does not retrain the model and does not claim
that the Controller is optimal.

## Scope

Use the same paper-selected scope as the Inner Actor validation:

- Nasdaq-100 experimental universe, seed 49.
- CSI-300 (`sh`) experimental universe, seed 90.
- One selected checkpoint per market.
- Full test period.
- Existing counterfactual action traces only; no new multi-day evaluation is
  required.

Primary input traces:

```text
paper_experiments_outputs/paper_experiments_final/_cache/
  counterfactual_horizon30/
    nas_seed49_full_controller_horizon30_actions.csv
    sh_seed90_full_controller_horizon30_actions.csv
```

The Nasdaq trace contains 1,334 free decisions and 231 free switches. The
CSI-300 trace contains 1,220 free decisions and 92 free switches. Every free
decision has complete hold and switch curves through 30 days.

## Exclusions

- Do not repeat comparisons against fixed 5/10/20/30/60-day policies.
- Do not select only favorable case windows for the primary statistics.
- Do not treat `exit_prob` as a calibrated probability that switching will be
  profitable. It is a policy propensity and will be evaluated primarily as a
  ranking signal.
- Do not claim cross-seed or training-run uncertainty.
- Do not claim a statistically supported timing effect when the relevant
  confidence interval includes zero or the corrected test fails.

## Data Validation

Before computing statistics:

1. Parse dates and sort them.
2. Keep rows with `decision_type == "free_decision"`.
3. Verify unique decision dates and steps.
4. Verify finite `exit_prob`, `duration_before_decision`, and action flags.
5. Verify that `is_switch` and `is_free_switch` agree on free decisions.
6. Parse `hold_curve_30` and `switch_curve_30`.
7. Verify both curves start at one, have the same usable length, and cover the
   event-specific horizon.
8. Record invalid and excluded row counts in the manifest and report.

Failure to satisfy the curve or action invariants must stop the analysis for the
affected market rather than silently dropping a material number of rows.

## Event-Adaptive Horizon

For free decision \(t\), define:

\[
H_t = \max(1, 30-d_t),
\]

where \(d_t\) is `duration_before_decision`.

This compares immediate switching with continuing the current portfolio until
the end of its original 30-day holding window. The evaluation length therefore
varies by decision and is not a fixed-day policy comparison.

For robustness only, derive the same quantities at 5, 10, 20, and 30 days from
the already stored counterfactual curves. These fixed horizons are sensitivity
checks, not strategy baselines.

## Counterfactual Outcomes

Let \(W^{switch}_{t,h}\) and \(W^{hold}_{t,h}\) be the frozen counterfactual
wealth paths stored in the trace.

### Return advantage

Because adaptive horizons have different lengths, use daily log-return
advantage as the primary return outcome:

\[
A^R_t =
\frac{
\log W^{switch}_{t,H_t} -
\log W^{hold}_{t,H_t}
}{H_t}.
\]

Report it in basis points per day. Also report the cumulative
switch-minus-hold return over \(H_t\) as a descriptive effect size.

### Drawdown advantage

\[
A^{MDD}_t =
MDD(W^{hold}_{t,0:H_t}) -
MDD(W^{switch}_{t,0:H_t}).
\]

A positive value means switching reduces counterfactual maximum drawdown.

### Actual decision value

Let \(a_t=1\) for `switch` and \(a_t=0\) for `hold`.

\[
V^R_t = (2a_t-1)A^R_t,
\]

\[
V^{MDD}_t = (2a_t-1)A^{MDD}_t.
\]

Positive decision value means the Controller selected the better of the two
frozen counterfactual actions for that outcome.

The primary timing hypothesis is:

\[
H_0: E[V^R_t] \le 0.
\]

## Analysis 1: Full Free-Decision Value

This is the primary mechanism analysis and must use every valid free decision,
not only actual switches.

For each market, report:

- number of free decisions and free switches;
- empirical switch rate;
- mean, median, and standard deviation of \(H_t\);
- mean and median \(V^R_t\);
- positive-return-decision ratio;
- mean and median \(V^{MDD}_t\);
- positive-MDD-decision ratio;
- Newey-West mean test;
- circular block-bootstrap 95% confidence intervals.

The full-decision statistic is primary because switch-only statistics condition
on the policy's selected events and cannot evaluate whether hold actions were
correct.

## Analysis 2: Switch/Hold Decomposition

Split all free decisions by actual action.

For actual switches, a correct return decision has \(A^R_t>0\). For actual
holds, a correct return decision has \(A^R_t<0\).

Report per action group:

- sample size;
- mean counterfactual switch advantage;
- block-bootstrap 95% confidence interval;
- favorable-action ratio;
- mean counterfactual MDD advantage;
- favorable-MDD-action ratio.

Actual-switch statistics are secondary, conditional event evidence. They must
not be presented as an unbiased estimate of the value of all Controller
decisions.

## Analysis 3: Exit-Probability Ranking

Evaluate whether `exit_prob` ranks states with larger switch advantage.

Report:

1. Spearman correlation between `exit_prob` and \(A^R_t\).
2. Spearman correlation between `policy_logit` and \(A^R_t\).
3. Exit-probability quintile means for \(A^R_t\), \(A^{MDD}_t\), and actual
   switch rate.
4. Q5-minus-Q1 return-advantage difference with a block-bootstrap interval.
5. AUROC for the label \(\mathbb{1}(A^R_t>0)\).
6. AUROC block-bootstrap interval.
7. Balanced accuracy and Matthews correlation coefficient for the actual
   action versus the sign-optimal counterfactual action.

Do not use Brier score or reliability calibration language because
`exit_prob` is not trained as a probability of positive future advantage.

## Analysis 4: Matched Action-Permutation Placebo

Test whether the actual timing is better than an action sequence with the same
switch intensity and broadly similar decision opportunities.

Construct strata within each market using:

- holding-duration tercile;
- trailing market-volatility tercile.

Within each stratum, permute the observed action labels. This preserves the
number of switches within duration and volatility conditions while breaking the
link between each decision state and its chosen action.

For each of 5,000 permutations, compute mean \(V^R_t\) and mean
\(V^{MDD}_t\). Use the finite Monte Carlo correction:

\[
p =
\frac{1+\#(\bar V^{perm}\ge \bar V^{actual})}{B+1}.
\]

Report:

- observed decision value;
- placebo mean;
- placebo 95% range;
- observed percentile;
- one-sided permutation p-value;
- invalid-stratum and invariant counts.

This is the strongest test of timing specificity because it controls switch
frequency, holding duration, volatility regime, market, and the complete set of
counterfactual outcomes.

## Analysis 5: Dynamic Holding Behavior

Describe the learned non-fixed timing policy without comparing it with a fixed
schedule.

Report:

- completed holding-spell count;
- holding-duration mean, standard deviation, and 10/25/50/75/90 percentiles;
- empirical switch hazard by holding duration;
- switch rate by duration tercile.

Fit an explanatory discrete-time logistic model:

\[
\Pr(a_t=1) =
\operatorname{logit}^{-1}
\left[
\beta_0 + f(d_t) +
\beta_1\,preReturn_t +
\beta_2\,preDrawdown_t +
\beta_3\,marketVol_t
\right].
\]

Use observable state proxies derivable without future data:

- current holding duration;
- trailing 20-day portfolio return;
- trailing 20-day portfolio drawdown;
- trailing 20-day equal-weight volatility of the same cleaned experimental
  asset universe, computed from project adjusted-close returns through the
  decision date.

Report odds ratios and time-dependence-robust intervals. Treat this model as
behavioral explanation, not causal identification.

## Analysis 6: State-Conditional Decision Value

For holding duration, prior portfolio return, prior drawdown, and market
volatility terciles, report:

- free-decision count;
- switch rate;
- mean exit probability;
- mean \(V^R_t\);
- positive-return-decision ratio;
- mean \(V^{MDD}_t\).

A state-dependent switch response is considered economically interpretable only
when higher switching activity is accompanied by improved decision value.
Switch-rate differences alone show sensitivity, not decision quality.

## Statistical Inference

The counterfactual windows overlap, so decision observations are not
independent.

Primary inference:

- circular block bootstrap;
- block length 30 consecutive free-decision rows;
- 10,000 repetitions;
- paired resampling of all values belonging to the same decision;
- Newey-West HAC with lag 5 as a secondary mean test.

Robustness:

- block lengths 20, 40, and 60;
- pre-existing 5/10/20/30-day counterfactual horizons;
- Benjamini-Hochberg correction within each hypothesis family.

The report must show both raw and adjusted p-values where multiple outcomes or
horizons are tested.

## Claim Rules

### Supported

A claim is supported only when its primary effect has the expected sign, its
primary block-bootstrap confidence interval excludes the null, and any
applicable corrected p-value is below 0.05.

### Descriptive

Use descriptive language when the point estimate has the expected sign but the
interval includes the null, or when only subgroup patterns support the claim.

### Not supported

Use `NOT SUPPORTED` when the point estimate is in the wrong direction or both
the interval and test fail.

The final report will classify:

- overall action value;
- switch timing specificity;
- exit-probability ranking;
- drawdown timing value;
- state-dependent behavior.

## Outputs

Write all outputs under:

```text
reproduced_outputs/controller_adaptive_timing_statistical_validation/
```

Required tables:

```text
tables/adaptive_horizon_decision_value.csv
tables/switch_hold_decomposition.csv
tables/exit_probability_ranking.csv
tables/exit_probability_quintiles.csv
tables/matched_action_permutation.csv
tables/holding_duration_distribution.csv
tables/holding_duration_hazard.csv
tables/controller_state_model.csv
tables/controller_state_conditional_value.csv
tables/horizon_robustness.csv
```

Required figures:

```text
figures/decision_value_forest_nas.png
figures/decision_value_forest_sh.png
figures/action_permutation_nas.png
figures/action_permutation_sh.png
figures/exit_probability_ranking_nas.png
figures/exit_probability_ranking_sh.png
figures/holding_hazard_nas.png
figures/holding_hazard_sh.png
```

Save PDF versions of all figures.

Required documentation:

```text
CONTROLLER_ADAPTIVE_TIMING_STATISTICAL_VALIDATION.md
metadata/run_manifest.json
```

The manifest records input paths and hashes, checkpoint identities when
available, analysis arguments, random seeds, row-validation counts, and the code
commit.

## Implementation Boundaries

Create a standalone analysis module under `paper_experiments/` rather than
mixing inferential statistics into the existing figure-generation pipeline.
Reuse tested statistical helpers from the Inner–Outer analysis where their
semantics match.

Do not overwrite the user's existing Controller figures or CSVs under
`paper_experiments_outputs/paper_experiments_final/`.

## Testing

Tests must cover:

- adaptive-horizon indexing;
- daily log-return advantage;
- drawdown advantage;
- action-value sign for switch and hold;
- all-free-decision inclusion;
- action-stratified decomposition;
- matched permutation invariants;
- finite Monte Carlo p-value;
- AUROC and degenerate-label handling;
- holding-spell and hazard construction;
- block-bootstrap pairing;
- deterministic outputs for fixed random seeds;
- manifest and required-artifact creation.

Synthetic tests must include cases where:

- switch is always better;
- hold is always better;
- the policy is perfectly aligned;
- the policy is random;
- exit probability is perfectly ranked;
- exit probability has no variation;
- adaptive horizons differ across decisions.
