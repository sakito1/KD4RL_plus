# Controller Decision Statistics Design

## Goal

Evaluate whether each free Controller decision selected the better Hold or Switch
alternative, and supplement that one-step decision analysis with a descriptive
comparison over the realized segment from each free switch to the next revision.
The analysis uses the final selected model traces only and does not retrain or pool
training seeds.

## Counterfactual Alternatives

At every free decision, the existing evaluation trace contains two deterministic
counterfactual curves generated from the same state and future price path:

- **Hold:** retain the drifted active base and apply the deterministic Inner Actor.
- **Switch:** replace the active base with the Outer Actor candidate and apply the
  same deterministic Inner Actor.

Both alternatives include their initial transaction cost relative to the same
pre-decision holding. Their future weights are then frozen, so later Controller
or Inner-Actor decisions do not contaminate the comparison.

## Primary: One-Step Chosen-Action Advantage

Because the Controller makes another decision on the next trading day, the primary
decision horizon is one trading day. Let `delta = log(V_switch[1]) -
log(V_hold[1])`. The chosen-action advantage is `delta` for a Switch decision and
`-delta` for a Hold decision. Positive values mean the action actually chosen by
the Controller beat its unchosen counterfactual.

For each market, report Hold and Switch counts, mean and median advantage, positive
ratio, balanced hit rate, overall chosen-action advantage, and a 95% moving-block
bootstrap confidence interval. Bootstrap resampling uses 30-trading-day blocks to
retain time dependence.

## Secondary: Realized-Segment Switch Comparison

For every actual free switch, define the realized segment endpoint as the next
revision of any kind or the test-set end. Compare the frozen Switch and Hold curves
over that identical segment length. Report segment length, terminal return
difference, maximum-drawdown difference, their mean/median/positive ratios, and
block-bootstrap confidence intervals.

Because many segments are only one or two days, do not compute segment Sharpe,
Sortino, or annualized volatility. The report must disclose the duration
distribution and the fraction of one-day segments.

## Outputs

Create `paper_experiments_outputs/controller_decision_statistics/` containing:

- `decision_level_events.csv`
- `decision_level_summary.csv`
- `segment_level_events.csv`
- `segment_level_summary.csv`
- `CONTROLLER_DECISION_STATISTICS.md`

The Markdown report distinguishes evidence, limitations, and safe paper claims. It
must not claim cross-seed statistical significance or generalize beyond the final
selected policies and their test periods.

## Validation

Focused tests cover Hold/Switch sign orientation, transaction-cost-inclusive curve
use, next-revision segment boundaries, maximum drawdown, and deterministic bootstrap
output. Production results are checked for complete market coverage, finite summary
statistics, event-count consistency, and reproducibility under a fixed bootstrap
seed.
