# AAAI Reviewer-Facing Self Check

## Core Story

CMTFlow should be presented as a portfolio-memory control framework, not as a simple stack of three policies. The central object is the active base portfolio `b_t`: it can be held, drifted with prices, compared with a candidate base `w_t^{cand}`, replaced by the controller, and refined locally by the inner actor.

## Claim-Evidence Map

- Claim: Existing RL portfolio methods entangle revision timing, base construction, and daily refinement.
  Evidence: Introduction and Related Work distinguish daily allocation, adaptive interval methods, and CMTFlow's explicit active-base memory.
  Status: supported by positioning; avoid saying all prior methods fail universally.

- Claim: The controller is the dominant adaptive component.
  Evidence: Table 4 shows Outer + Controller improves TR/MDD/CR over Outer-only on Nasdaq-100 and achieves the strongest ablation TR/Sharpe/CR on CSI-300.
  Status: supported.

- Claim: CMTFlow improves risk-return trade-off, not every metric.
  Evidence: Table 3: best Nasdaq-100 TR, best CSI-300 Sharpe, second-best CR in both markets, lower CSI-300 MDD than DeepTrader.
  Status: supported; keep the wording as trade-off rather than absolute dominance.

- Claim: Fixed holding windows cannot replace learned revision timing.
  Evidence: Table 4 and Figures 8-9 compare fixed 5/10/20/30/60-day controllers under the same evaluation setting.
  Status: supported.

- Claim: Inner actor provides local refinement rather than the main alpha source.
  Evidence: Inner interpretability figures show tilt-return resonance, while ablation shows its aggregate contribution is market-dependent.
  Status: supported with careful wording.

## Remaining Reviewer Risks

- Page budget: the compiled PDF includes appendix pages. Before submission, confirm the AAAI main-paper and appendix policy and split supplementary material if required.
- Baseline reproducibility: keep the baseline matching manifest available, because reviewers may ask why some metric-only baselines do not appear in wealth curves.
- Statistical robustness: current evidence is primarily selected-checkpoint and matched-run based. If time permits, add seed variance or bootstrap confidence intervals for key metrics.
- Transaction sensitivity: a cost-rate sensitivity table would strengthen the practical finance claim.
- Controller interpretability: the aggregate switch advantage is intentionally conservative. Do not overclaim that every switch improves return.

## Recommended Submission Positioning

Use this sentence as the safest high-level claim:

> CMTFlow improves the return-risk trade-off by learning when to revise a persistent active base portfolio, while using the outer actor for candidate-base construction and the inner actor for local within-base refinement.

