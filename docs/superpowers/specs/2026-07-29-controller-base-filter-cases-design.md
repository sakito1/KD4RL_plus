# Controller Base Filter Cases Figure Design

## Summary

Create one publication-ready 2×3 figure and a Chinese evidence report for two
verified Base-filter cases: NASDAQ-100 on 2023-01-17 and CSI-300 on 2021-09-22.
The supported claim is that Base provides a conservative hurdle that can block
a weak positive Adv signal. The figure must not claim that Base is a calibrated
current-portfolio-quality score.

## Chosen approach

Each market occupies one row:

1. current holding-segment cumulative-return path;
2. waterfall-style Controller decomposition showing Base, Adv correction, and
   final logit/probability;
3. 30-day frozen Hold-versus-candidate paths with 20-day and 30-day gaps.

The middle panel also reports the full-test-set count of positive-Adv decisions
blocked by Base. This separates population-level mechanism evidence from
case-level economic illustration.

## Evidence boundary

- Mechanism: supported in both markets by exact logit decomposition and
  hundreds of positive-Adv decisions blocked by Base.
- Case outcome: both selected cases favor Hold at 20 and 30 days.
- Population economic effectiveness: supported for CSI-300, not for
  NASDAQ-100; this limitation must appear in the report.

## Outputs

- `scripts/plot_controller_base_filter_cases.py`
- `reproduced_outputs/controller_base_filter_cases/controller_base_filter_cases.png`
- `reproduced_outputs/controller_base_filter_cases/controller_base_filter_cases.pdf`
- `reproduced_outputs/controller_base_filter_cases/controller_base_filter_cases.csv`
- `reproduced_outputs/controller_base_filter_cases/CONTROLLER_BASE_FILTER_CASES_CN.md`

## Verification

Recompute Base + Adv correction = final logit, action threshold, future
20/30-day return directions, and full-test positive-Adv blocked counts from the
saved traces and counterfactual files.
