# Inner–Outer Statistical Validation

## Material Passport

- Verification Status: ANALYZED
- Scope: paper-selected checkpoint, one checkpoint per market
- Inference: Newey-West HAC and paired circular block bootstrap
- Limitation: intervals describe test-period uncertainty, not training-seed uncertainty

## Claim-level results

### NAS configuration refinement

- Mean Active Share: 1.0714% (>1% on 50.3% of days).
- Mean ex-ante volatility change: -0.0267%; 95% CI [-0.0617%, 0.0012%] — **NOT SUPPORTED**.

### SH configuration refinement

- Mean Active Share: 1.6189% (>1% on 74.3% of days).
- Mean ex-ante volatility change: -0.0884%; 95% CI [-0.1170%, -0.0608%] — **SUPPORTED**.

### NAS frozen-path direct effect

- Fair net alpha: -0.049 bp/day; 95% CI [-0.174, 0.077], p=0.4690 — **NOT SUPPORTED**.

### SH frozen-path direct effect

- Fair net alpha: -0.214 bp/day; 95% CI [-0.431, -0.019], p=0.0762 — **NOT SUPPORTED**.

### NAS closed-loop contribution

- Total-return difference: 28.02%; 95% CI [-7.59%, 94.49%] — **NOT SUPPORTED**.
- MDD difference: -2.62%; 95% CI [-4.66%, 1.99%] — **NOT SUPPORTED**.

### SH closed-loop contribution

- Total-return difference: 13.97%; 95% CI [-52.09%, 89.15%] — **NOT SUPPORTED**.
- MDD difference: -0.24%; 95% CI [-5.83%, 3.44%] — **NOT SUPPORTED**.

## Risk-refinement placebo

### NAS

- Actual mean ex-ante volatility change: -0.0267%; random-tilt mean: 0.0020%.
- One-sided risk-reduction permutation p=0.0002.
- Alpha-direction permutation p=0.7692; this is not used to claim standalone alpha.

### SH

- Actual mean ex-ante volatility change: -0.0884%; random-tilt mean: 0.0054%.
- One-sided risk-reduction permutation p=0.0002.
- Alpha-direction permutation p=0.9818; this is not used to claim standalone alpha.

## Interpretation boundary

A non-significant frozen-path alpha is reported as a null result. A positive closed-loop difference is interpreted as a system-level complementary contribution, not as proof of standalone daily alpha.

Placebo results are available in `tables/placebo_analysis.csv`.
