# Fixed-Path Transaction-Cost Sensitivity Design

## Objective

Measure how the paper-selected CMTFlow trajectories for Nasdaq-100 seed 49 and
CSI-300 seed 90 degrade as proportional transaction costs increase, without
changing any Controller, Manager, or Trader decisions.

This experiment is a mechanical cost-sensitivity replay. It does not estimate
how the policies would behave if transaction cost were changed in the
environment state, during inference, or during training.

## Inputs

- Paper-selected full-controller action traces:
  - Nasdaq-100 seed 49.
  - CSI-300 seed 90.
- Daily executed portfolio weights from each trace.
- Adjusted closing prices for the exact trace asset universe.
- Requested transaction-cost rates:
  - 0.010% (`0.00010`).
  - 0.015% (`0.00015`).
  - 0.020% (`0.00020`).
  - 0.050% (`0.00050`).
- Reference transaction-cost rate:
  - 0.005% (`0.00005`), matching the selected models' original environment.

## Replay Definition

For each decision date \(t\), drift the previous executed target weights with
the realized price relatives from \(t-1\) to \(t\):

\[
\widetilde w_{i,t}
=
\frac{w_{i,t-1}(P_{i,t}/P_{i,t-1})}
{\sum_j w_{j,t-1}(P_{j,t}/P_{j,t-1})}.
\]

Compute full \(L_1\) turnover:

\[
TO_t=\sum_i|w_{i,t}-\widetilde w_{i,t}|.
\]

For cost rate \(c\), compute the net one-day growth from \(t\) to the next
available trading date:

\[
G_{t+1}^{(c)}
=
(1-cTO_t)
\sum_i w_{i,t}\frac{P_{i,t+1}}{P_{i,t}}.
\]

The executed weight path, switch dates, asset support, and all policy outputs
remain identical across cost rates. Only the cost multiplier changes.

The initial trace row has no preceding trace target from which to reconstruct
turnover and is excluded consistently from all cost-rate comparisons. Each
market therefore uses one common aligned replay sample across all rates.

## Outputs

Write results under:

`reproduced_outputs/fixed_path_transaction_cost_sensitivity/`

Required artifacts:

- `tables/transaction_cost_sensitivity.csv`
  - market, seed, cost rate and cost percentage;
  - replay days;
  - total return;
  - Sharpe ratio;
  - maximum drawdown;
  - Calmar ratio;
  - mean daily turnover;
  - cumulative charged cost rate;
  - changes in TR, SR, MDD, and CR relative to the 0.005% reference.
- `tables/<market>_daily_replay.csv`
  - common dates, gross return, turnover, and the net return/wealth path for
    every cost rate.
- `TRANSACTION_COST_SENSITIVITY.md`
  - compact paper-facing table;
  - interpretation of monotonic degradation and market differences;
  - explicit fixed-path limitation.
- `metadata/run_manifest.json`
  - input paths and SHA-256 hashes;
  - cost rates;
  - selected checkpoint identities;
  - replay command and code commit.

No figure is required for this first pass.

## Metrics

Use the same path definitions as the existing paper experiment utilities:

- \(TR=V_T/V_0-1\).
- \(SR=\operatorname{mean}(r_t)/\operatorname{std}(r_t)\sqrt{252}\), with zero
  risk-free rate.
- \(MDD\) as maximum peak-to-trough wealth loss.
- \(CR=AR/MDD\), where \(AR=252\operatorname{mean}(r_t)\).

Metrics must be recomputed from each replayed net-return path rather than
adjusted algebraically from the published summary metrics.

## Validation

Automated tests must verify:

1. Turnover is computed against price-drifted previous executed weights.
2. Zero turnover produces identical results for every cost rate.
3. Higher cost never increases the same day's net growth when turnover is
   positive.
4. The 0.005% replay matches the independently reconstructed fixed-path formula.
5. Every cost rate uses identical dates, weights, gross returns, and turnover.
6. Output metrics are finite and wealth remains positive.
7. Total return is monotonically non-increasing as cost rises for each market.

## Interpretation Boundary

This experiment supports statements about the fee sensitivity of the already
observed trading path. It cannot support statements about:

- policy adaptation to a new cost feature;
- changes in Controller switch decisions;
- model performance after retraining under a new cost;
- optimality of the selected cost level.

Those questions require a separate inference-rerun or retraining experiment.
