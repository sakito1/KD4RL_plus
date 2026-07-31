# Controller Probability–Advantage Resonance Figure Design

## Summary

Create two English, appendix-ready figures—one for NASDAQ-100 and one for
CSI-300—that show how the Controller's Base and Adv probabilities evolve with
the realized counterfactual advantage of switching from the current portfolio
to the candidate portfolio. The figures are explanatory case windows, not
full-test-set statistical evidence.

## Data and scope

- Repository: `KD4RL_plus`.
- Markets and seeds: NASDAQ-100 seed 49; CSI-300 seed 90.
- Decision sample: free Controller decision dates only
  (`decision_type == "free_decision"`).
- Controller trace:
  `reproduced_outputs/controller_gate_adv_case_analysis/controller_gate_adv_trace_{market}.csv`.
- Frozen counterfactuals:
  `paper_experiments_outputs/paper_experiments_final/_cache/counterfactual_horizon30/{market}_seed{seed}_full_controller_horizon30_actions.csv`.
- Selected representative windows:
  - NASDAQ-100: 2023-10-27 through 2023-12-28, 42 free decisions and
    20 Switch decisions.
  - CSI-300: 2024-05-31 through 2024-07-19, 35 free decisions and
    4 Switch decisions.

The windows were selected ex post from approximately 50–75-calendar-day
windows. Eligible windows must contain at least 32 free decisions, at least
three Switch decisions, and positive mean chosen-action advantage. Ranking uses
the minimum Adv-probability Spearman correlation across the adaptive, fixed
20-day, and fixed 30-day counterfactual targets. This selection rule and its
post-hoc status must be disclosed in the caption or accompanying text.

## Quantities

### Base probability

For each date,

\[
p_t^{base}=\sigma(\ell_t^{base}).
\]

This is the Base-only probability of switching. It is a low-volatility
conservative prior, not an outcome predictor. The observed selected-window
ranges are approximately 26.4580%–26.5911% for NASDAQ-100 and
26.9517%–27.1595% for CSI-300.

### Adv probability

For each date,

\[
p_t^{adv}=\sigma\left(1.9\tanh\left(\ell_t^{adv}/0.02\right)\right).
\]

This is the Adv-only switch propensity after applying the exact nonlinear
correction used by the Controller. It must not be presented as a return
forecast.

### Counterfactual switch advantage

The main line uses the training-aligned adaptive-horizon target normalized per
remaining trading day:

\[
\Delta_t^{switch}
=\frac{R_t(\text{candidate})-R_t(\text{current})}
{\max(1,30-d_t)}\times 10{,}000,
\]

where \(d_t\) is the holding duration before the decision. Units are bp/day.
Positive values favor Switch; negative values favor Hold. Fixed 20-day and
30-day targets are robustness statistics and are not additional main-panel
lines.

## Figure layout

Produce one vertically aligned figure per market.

1. Top panel: adaptive counterfactual switch advantage over time.
   - A zero reference line separates Switch-favorable and Hold-favorable
     regions.
   - Positive regions use a muted teal fill; negative regions use a muted
     vermilion fill.
   - Switch dates receive upward triangle markers; Hold dates use small circle
     markers.
   - All dates align exactly with the heatmap columns.
2. Bottom panel: two probability heatmap rows.
   - First row: Base probability.
   - Second row: Adv probability.
   - The Base row uses a market-specific tight probability scale so its small
     variations remain visible. Its colorbar displays the true probability
     range and is explicitly labeled `zoomed scale`.
   - The Adv row uses its own probability scale centered at 50%, with cool
     colors favoring Hold and warm colors favoring Switch.
   - Because the rows use separate scales, the caption states that color
     intensity is comparable over time within a row, not in magnitude across
     rows.
3. A compact statistics box reports:
   - adaptive-horizon Spearman correlation;
   - fixed 20-day and 30-day Spearman correlations;
   - chosen-action direction accuracy;
   - mean chosen-action counterfactual value in bp/day.

The x-axis uses trading dates with sparse readable labels. The figures use the
existing paper typography and colorblind-safe palette and avoid heavy borders
or dense per-cell numeric annotations.

## Verified expected statistics

| Market | Adaptive rho | Fixed-20 rho | Fixed-30 rho | Direction accuracy | Mean chosen value |
|---|---:|---:|---:|---:|---:|
| NASDAQ-100 | 0.636 | 0.551 | 0.559 | 76.2% | +6.91 bp/day |
| CSI-300 | 0.598 | 0.498 | 0.556 | 62.9% | +3.97 bp/day |

The implementation must recompute these values from source CSVs and fail if
they materially disagree with the expected values rather than hard-coding the
table.

## Outputs

Write new files under
`reproduced_outputs/controller_probability_advantage_resonance/`:

- `controller_probability_advantage_resonance_nas.png`
- `controller_probability_advantage_resonance_nas.pdf`
- `controller_probability_advantage_resonance_sh.png`
- `controller_probability_advantage_resonance_sh.pdf`
- `controller_probability_advantage_resonance_daily.csv`
- `controller_probability_advantage_resonance_summary.csv`

Implement the reproducible plot in
`scripts/plot_controller_probability_advantage_resonance.py`.

## Validation

- Validate one-to-one joins on `(date, step)`.
- Validate the Controller identity
  `base_exit_logit + 1.9*tanh(raw_adv/0.02) == exit_logit` within numerical
  tolerance.
- Validate that only free decisions in the declared windows enter the plots.
- Recompute all correlations, action counts, direction accuracy, and mean
  chosen value from the merged records.
- Validate that the two heatmap rows have the same date columns as the line
  panel.
- Save raster output at publication quality and also save vector PDF output.
- Visually inspect both PNG files for clipping, misleading color scales,
  unreadable dates, and incorrect marker alignment.

## Interpretation boundary

The figures support a mechanism-level statement: in these representative
windows, higher Adv switch propensity co-moves with higher candidate-versus-
current counterfactual advantage, while Base remains a stable conservative
prior. They do not establish full-test-set significance because the windows
were selected after inspecting the test data. Full-test-set evidence remains
the responsibility of the existing Controller statistical report.
