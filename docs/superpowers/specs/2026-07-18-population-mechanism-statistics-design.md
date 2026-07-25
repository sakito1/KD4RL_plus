# Population-Level Mechanism Statistics Design

## Scope

Analyze only the two selected final-model deterministic replay traces
(Nasdaq-100 seed 49 and CSI-300 seed 90). Do not retrain models and do not
aggregate training seeds. Produce CSV evidence and one Chinese Markdown report;
do not generate figures or edit the paper LaTeX in this pass.

## Controller analysis

Use every `free_decision` row with a complete 20-trading-day counterfactual
curve. Define return uplift as `switch_future_return_20 -
hold_future_return_20`, and drawdown improvement as
`hold_future_mdd_20 - switch_future_mdd_20`, so positive values favor Switch.
Create deterministic equal-count probability quintiles within each market.
Report quintile means, Switch win rates, drawdown improvements, Spearman
probability--uplift correlation, and Q5--Q1 uplift. Use a 20-day circular
moving-block bootstrap on the dense decision calendar for 95% intervals.

## Inner Actor analysis

At each test date, restrict the cross-section to the ten assets with positive
base weight. Define tilt as execution weight minus base weight. Compute future
log return from date t to t+h for h=1 and h=5, and demean it within that day's
active support. Report daily Pearson IC, Spearman Rank IC, positive-day ratios,
and 20-day block-bootstrap intervals for their time-series means. Rank the ten
active assets into five deterministic equal-count tilt groups and report the
date-balanced mean future relative return for each group plus Q5--Q1 spread.

## Holding-duration analysis

Construct actual completed holding segments between consecutive revision
dates. A segment's termination type is the next revision's `free_switch` or
`forced_switch` label. Exclude the final right-censored segment from duration
summary statistics but report it separately. Report mean, median, 25th and
75th percentiles, and free/forced termination proportions.

## Output

Write detailed event and summary CSV files under
`paper_experiments_outputs/population_level_mechanism_analysis/`. The Chinese
Markdown report contains one compact paper-candidate table and supporting
quintile tables, with explicit caveats where intervals cross zero or patterns
are non-monotonic.
