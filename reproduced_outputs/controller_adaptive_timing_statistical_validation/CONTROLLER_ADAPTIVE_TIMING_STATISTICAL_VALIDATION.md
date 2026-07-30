# Controller Adaptive-Timing Statistical Validation

This analysis uses every learned free switch/hold decision and excludes fixed-day policy comparisons.

Intervals quantify time-series uncertainty within the test period for one selected checkpoint. They do not measure cross-seed training uncertainty.

## Claim audit

- **DESCRIPTIVE — NASDAQ-100 chosen-action return value:** 0.240 bp/day (95% block CI -2.688, 3.298; adjusted HAC p=0.8357).
- **NOT SUPPORTED — NASDAQ-100 drawdown decision value:** -0.030 pp of MDD reduction (95% block CI -0.228, 0.164 pp; adjusted HAC p=0.6846).
- **DESCRIPTIVE — NASDAQ-100 action decomposition:** switch dates have 0.232 bp/day switch-minus-hold advantage; hold dates have -0.241 bp/day. The latter contributes positively to chosen-action value only when it is negative, because hold was selected.
- **DESCRIPTIVE — NASDAQ-100 exit-probability ranking:** Spearman rho=0.040 (95% block CI -0.076, 0.149); Q5−Q1=-0.349 bp/day.
- **NOT SUPPORTED — NASDAQ-100 matched timing placebo:** observed 0.240 versus placebo mean 0.536 bp/day (adjusted permutation p=0.962).
- **NOT SUPPORTED — NASDAQ-100 matched drawdown timing placebo:** observed MDD value -0.030 pp versus placebo mean -0.023 pp (adjusted permutation p=0.6471).
- **SUPPORTED — CSI-300 chosen-action return value:** 5.831 bp/day (95% block CI 1.213, 10.430; adjusted HAC p=0.0009978).
- **SUPPORTED — CSI-300 drawdown decision value:** 0.371 pp of MDD reduction (95% block CI 0.158, 0.595 pp; adjusted HAC p=2.311e-05).
- **DESCRIPTIVE — CSI-300 action decomposition:** switch dates have 0.460 bp/day switch-minus-hold advantage; hold dates have -6.269 bp/day. The latter contributes positively to chosen-action value only when it is negative, because hold was selected.
- **SUPPORTED — CSI-300 exit-probability ranking:** Spearman rho=0.141 (95% block CI 0.018, 0.252); Q5−Q1=13.835 bp/day.
- **DESCRIPTIVE — CSI-300 matched timing placebo:** observed 5.831 versus placebo mean 5.260 bp/day (adjusted permutation p=0.1128).
- **SUPPORTED — CSI-300 matched drawdown timing placebo:** observed MDD value 0.371 pp versus placebo mean 0.305 pp (adjusted permutation p=0.03159).

State models and conditional-state tables are explanatory descriptions, not causal evidence. A positive action-value estimate means the realized action beat its same-date unchosen counterfactual over the adaptive remaining horizon.
