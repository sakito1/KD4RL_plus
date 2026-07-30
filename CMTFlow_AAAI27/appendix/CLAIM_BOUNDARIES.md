# Appendix Claim Boundaries

## B.1 Transaction costs

The cost sweep holds the recorded Controller, Manager, and Trader actions fixed
and mechanically changes the proportional fee. It shows that the complete
executed CMTFlow path remains economically robust through the tested costs. It
does not isolate standalone Trader alpha and does not retrain the policy at each
fee.

## B.2 Fixed holding-window sensitivity

The dense baseline changes only the Controller schedule. For each market it
evaluates all integer holding windows from 1 through 60 while retaining the
paper-selected Manager, Trader, checkpoint, and test split. The 0.01% values
are mechanical turnover-based fee replays of evaluation paths recorded at
0.005%; no retraining occurs.

The defensible conclusion is that the learned Controller attains
high-percentile risk-return performance without ex-post selection of a constant
holding period. A few fixed windows outperform it on individual metrics, so
the analysis must not be described as universal dominance. The experiment
tests timing-schedule sensitivity and does not isolate Manager or Trader alpha.

## C.1 Controller cases

The four cases are selected explanatory examples, not estimates of average
Controller performance. Probabilities are the recorded Controller outputs.
The 20/30-day differences are calculated from the endpoints of the packaged
frozen hold/candidate wealth curves. This curve-endpoint rule is authoritative
for the plotted values.

## C.2 Controller statistics

NASDAQ-100 has a descriptive mean chosen-action value of 0.240 bp/day and an
MDD value of -0.030 pp; the corresponding HAC confidence intervals include
zero. CSI-300 has a mean chosen-action value of 5.831 bp/day and an MDD value of
0.371 pp, supported by the reported HAC analysis. The table must not be
described as statistically significant in both markets.

## C.3 Trader statistics

Trader refinements have non-trivial Active Share and reduce ex-ante volatility
relative to random within-support tilts in the reported placebo test. The
frozen-path direct-return analysis does not support a standalone Trader alpha
claim. The defensible conclusion is risk-aware, non-random refinement combined
with cost robustness of the full executed strategy.
