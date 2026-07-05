# CMTFlow Paper Review

This review follows the `research-paper-writing` checklist: contribution, clarity, experimental strength, evaluation completeness, method soundness, and claim-evidence alignment. The review is written from a skeptical AAAI-style reviewer perspective.

## Overall Verdict

The current draft is much stronger than the earlier version: the story now matches the actual implementation, the controller is positioned as the main adaptive mechanism, and the experiments are organized around main results, ablation, and interpretability. The paper is potentially defensible, but several claims still need tightening before submission.

The main risk is not that the method has no evidence. The risk is that some sentences sound stronger than the evidence shown in the tables. In particular, CMTFlow is not best on every risk metric, the all-switch counterfactual statistics are only marginal on average, and the inner actor evidence is selected-window interpretability rather than universal standalone alpha.

## Priority Findings

### P0. The abstract and contribution claims should be more precise

Current risk:

- The abstract says CMTFlow improves the risk-return trade-off against matched baselines.
- The contribution bullet says it improves risk-adjusted return and downside-risk control.
- Table 3 shows a more nuanced pattern: CMTFlow has the best Nasdaq total return and best CSI-300 Sharpe, but DeepAries has better Nasdaq Sharpe and MDD, DeepTrader has higher CSI-300 total return, and DeepAries has lower CSI-300 MDD.

Recommended framing:

- Say CMTFlow achieves a favorable trade-off, not universal dominance.
- Emphasize the concrete strengths: highest Nasdaq return, best CSI-300 Sharpe, lower drawdown than high-return baselines, and strong full-model ablation.
- Avoid phrasing that implies every risk-adjusted or downside-risk metric is best.

Suggested sentence:

> Experiments on Nasdaq-100 and CSI-300 show that CMTFlow delivers a favorable high-return and controlled-drawdown trade-off: it achieves the highest Nasdaq total return and the best CSI-300 Sharpe ratio among matched methods, while keeping drawdown substantially lower than high-return traditional and DeepTrader baselines.

### P0. The controller counterfactual evidence must not be overclaimed

Current evidence:

- Representative cases are convincing: the selected switch windows show avoided drawdown.
- Fixed-window comparison is useful, especially on CSI-300.
- The all-switch aggregate counterfactual is weak on mean advantage:
  - Nasdaq: switch mean 3.29% vs old-holding 3.24%, positive-gain ratio 50.22%.
  - CSI-300: switch mean 4.43% vs old-holding 4.39%, positive-gain ratio 48.04%.

This does not support a claim that all switch decisions are statistically superior. It supports a more careful claim:

- Switching is not systematically worse than continuing.
- Many switches are maintenance decisions.
- The realized advantage comes from path-level timing and a subset of high-impact deterioration cases.
- Random-switch comparison supports learned timing more strongly than the all-switch average.

Suggested wording:

> The all-switch frozen counterfactual is intentionally conservative. Its mean advantage is small, which indicates that many switches are maintenance decisions rather than large alpha events. However, the switches are not systematically harmful relative to continuation, and the random-switch experiment plus representative deterioration cases show that the controller's timing matters for realized path-level risk control.

### P0. Baseline fairness and reproducibility need a stronger audit trail

Current risk:

- The paper says "matched baselines", but one AlphaStock CSI-300 curve is missing and included only as a metric.
- The text does not say how seeds/checkpoints were matched to table values.
- Reviewers may suspect cherry-picking unless the matching protocol is explicit.

Recommended additions:

- Add a short paragraph or appendix table named "Baseline Matching Protocol".
- State that baseline curves are included only when the stored trajectory reproduces the reported metric within tolerance.
- State that AlphaStock CSI-300 is metric-only because the matched action trajectory was overwritten.
- Add source/seed manifest summary, not raw hyperparameter clutter.

Suggested footnote/table note:

> We only plot baseline trajectories whose stored portfolio path reproduces the corresponding table metric. For AlphaStock on CSI-300, the matched log metric is available but the corresponding action trajectory is unavailable; therefore it is included in Table 3 but omitted from Figure 3.

### Resolved. Method terminology is unified

Previous terminology risk:

- The paper previously mixed outer/inner wording with older alias terms in equations.
- A note explains the mapping, but the switch still increases cognitive load.

Current status:

- The current TeX now uses outer actor / inner actor / controller consistently in the method and appendix formulas.
- Historical aliases were removed from the active draft to match the actual code modules.

### P1. The controller architecture and training details should remain explicit

The paper now states that the controller uses the recent market tensor, portfolio-state features, current-vs-candidate portfolio comparison features, two attention steps, and an exit logit with switch-advantage modulation. Reviewers may still ask:

- What exactly is in the compact portfolio-state vector `u_t`?
- What is the maximum holding cap `H_max`?
- What is the switching budget used in the controller reward?
- Which fixed holding windows are compared?
- Are fixed-window variants evaluated with the same cost model and actor stack?
- What PPO/actor-critic settings are used for each stage?
- How are seeds selected, and are results single-seed or best-validation-seed?

Recommended fix:

- Add one compact implementation table in the appendix.
- Include only necessary reproducibility details: hidden size, optimizer, learning rate, rollout length, PPO clip, entropy coefficient, training epochs, switch budget, `H_max`, number of random schedules, and model-selection criterion.

### P1. The related work needs sharper contrast with the closest baselines

Current related work is readable but too broad. The paper should explicitly differentiate CMTFlow from:

- DeepTrader: market-condition embedding and risk-return balancing, but still daily/monolithic portfolio generation.
- DeepAries: adaptive rebalancing interval selection, but not the same as controller-gated replacement of a base memory plus support-constrained inner refinement.
- HADAPS: hierarchical portfolio selection, but not explicit daily hold/switch base revision.
- AlphaStock: interpretable cross-asset attention, but focused on winner/loser attention rather than base-memory revision.

Recommended addition:

Add one final paragraph in Related Work:

> Closest to our work are adaptive and hierarchical portfolio methods such as HADAPS and DeepAries. CMTFlow differs in that it treats the active base portfolio as a persistent memory state and trains a controller to decide when this memory should be replaced by a candidate outer portfolio, while the inner actor is restricted to support-constrained tilting inside the active base. This separates revision timing, base construction, and daily refinement rather than learning only an adaptive interval or a single daily allocation policy.

### P1. Figure placement is acceptable, but page 9 is visually sparse

Current PDF order is now coherent, but the `FloatBarrier` before ablation creates a page with large blank space. This is not fatal, but it looks less polished.

Possible fixes:

- Move Table 4 and Figure 5 earlier/later so they fill the sparse page.
- Reduce vertical pressure from Figure 3/4 or Table 3.
- Keep the barrier only where it prevents a severe ordering problem.

Do not remove barriers blindly; the previous version had worse figure ordering.

### P1. The source file still contains template residue

Source-level issues:

- The author block still contains AAAI Press Staff placeholder text.
- The file still contains sample title blocks inside `\iffalse`.
- `\usepackage{bibentry}` is explicitly marked by the template as removable.
- The `.bib` file still contains many sample template entries.

The compiled PDF hides some of this under anonymous submission mode, but source hygiene matters before submission.

Recommended fix:

- Replace the author block with a clean anonymous-compatible placeholder or real author block for camera-ready.
- Remove unused template examples and unused packages.
- Trim the bibliography file to cited entries only.

### P1. Bibliography warnings should be fixed

Current BibTeX warnings:

- `empty booktitle in ANTICOR`
- `empty booktitle in UCRP`

Also check:

- `CAMP` should probably be renamed `CAPM` in the key and text where relevant.
- `Gao and guo Zhang` should be normalized to `Gao and Zhang` or `Wei-Guo Zhang`.
- Kalman bib entry has malformed title text, although it may not be cited.

Recommended fix:

- Add proper `booktitle` for Anticor and Universal Portfolios.
- Remove unused sample bib entries.
- Normalize author capitalization.

### P2. Inner actor interpretability is honest but should be positioned as selected-window evidence

The current text is already careful, which is good. The key is to avoid implying that the inner actor universally predicts future relative returns.

Best interpretation:

- Inner actor is not a standalone alpha engine.
- It is a within-holding-period tilting mechanism.
- Its interpretability evidence shows that in selected representative windows, overweight/underweight decisions resonate with subsequent relative performance.
- Its aggregate effect is market-dependent, which is consistent with the ablation.

Suggested final wording:

> The inner actor should be interpreted as a local execution refinement layer rather than an independent stock selector. Its role is to adjust exposure around the active base portfolio; selected-window evidence shows that these tilts can align with subsequent relative returns, while the ablation indicates that its standalone contribution is secondary to the controller.

### P2. Experimental statistics would be stronger with uncertainty or robustness checks

Current tables appear to be single matched runs. This is common in financial backtesting, but reviewers may ask for robustness:

- Multiple seeds for CMTFlow.
- Bootstrap confidence intervals on Sharpe/MDD.
- Subperiod performance, especially around bear/volatile regimes.
- Turnover and transaction cost sensitivity.
- Switch count and average holding length.

Recommended minimal additions:

- Add switch count, average holding length, and turnover to an appendix table.
- Add cost sensitivity with 2-3 transaction cost values if available.
- If multiple seeds are expensive, state model selection protocol clearly and avoid statistical significance claims.

## Claim-Evidence Map

| Claim | Evidence in draft | Status | Review comment |
|---|---|---|---|
| CMTFlow improves overall risk-return trade-off | Table 3, Figures 3-4 | Mostly supported | Must be framed as trade-off, not all-metric dominance. |
| Controller is the dominant adaptive mechanism | Table 4, Figure 5 | Supported | Stronger on Nasdaq; on CSI, Outer+Controller has higher return/Sharpe than full model, so explain inner actor as risk/local refinement. |
| Fixed-window switching cannot replace learned controller | Table 4, Figure 5 | Supported | Good evidence. Add switch counts/holding lengths if possible. |
| Controller switching is not random | Figure 7 | Supported, stronger on CSI | On Nasdaq random return is close; emphasize MDD and path stability. |
| Controller can avoid deteriorating holdings | Figure 6 cases | Supported as case evidence | Do not generalize case studies into global statistical dominance. |
| All-switch counterfactual supports switch quality | Text stats | Weak/modest | Use conservative wording: not systematically worse; many switches are maintenance. |
| Exit probability resonates with investment state | Figure 6 local traces, text correlation | Partly supported | Weak global correlation should be explicitly interpreted as nonlinear policy signal. |
| Inner actor earns short-term relative-return opportunities | Figure 8 selected windows | Partly supported | Say selected-window resonance, not universal alpha. |
| Method is reproducible | Method + appendix | Needs revision | Add controller details, hyperparameters, model-selection protocol, switch budget, random schedule count. |
| Baseline comparison is fair | Table 3 + baseline note | Needs revision | Add matched baseline manifest/protocol summary and clarify missing AlphaStock CSI curve. |

## Section-by-Section Review

### Abstract

Strength:

- Clear problem decomposition.
- Accurately describes the three components.

Needs revision:

- "Improves the risk-return trade-off" is acceptable, but "risk-adjusted return and downside-risk control" should be made concrete.
- "Controller is the dominant source of adaptive risk control" should be tied to ablation evidence rather than stated as universal.

### Introduction

Strength:

- The three-question framing is strong: when to revise, what base to hold, how to refine.
- Contribution bullets are aligned with the method.

Needs revision:

- The novelty claim should contrast directly with DeepAries/HADAPS/DeepTrader.
- Contribution bullet 1 and 2 overlap. Consider making bullet 1 the problem insight, bullet 2 the architecture, bullet 3 the empirical evidence.

### Related Work

Strength:

- Covers classical PM, RL PM, cost/risk control, and interpretable RL baselines.

Needs revision:

- Too much broad survey, not enough closest-method contrast.
- Add a paragraph that explicitly says why adaptive interval selection is not the same as controller-gated base-memory revision.

### Problem Formulation

Strength:

- Drift operator and transaction-cost factor are useful and clear.

Needs revision:

- Objective says maximize terminal wealth, while many rewards use Sharpe and relative log-return. Add one bridging sentence: terminal wealth defines the environment objective, while learning uses risk-adjusted and relative rewards for stable optimization.

### Methodology

Strength:

- Outer/inner/controller decomposition is coherent.
- Support-constrained refinement is technically meaningful.

Current check:

- Terminology has been unified in the active TeX.
- Controller-state vector and budget constraints are now explicitly described in the method and appendix.
- Keep the controller state construction visible in future edits: recent market window, holding-state features, and candidate-action features should remain stated before deferring to appendix.

### Experiments

Strength:

- Two markets, matched baselines, main/ablation/interpretability structure is good.
- Text already avoids claiming dominance on every metric.

Needs revision:

- Baseline matching protocol must be explicit.
- Report model-selection criterion.
- Add switch count/turnover/cost sensitivity if possible.
- Fixed-window controller setup needs the compared horizons and matching protocol.

### Interpretability

Strength:

- Good decision to include only interpretable figures that support the story.
- Controller cases are meaningful.
- Inner actor explanation is honest and not overclaimed.

Needs revision:

- All-switch counterfactual paragraph should be more conservative.
- Exit probability should not be interpreted as a direct return predictor.
- Consider adding a small table of switch statistics if space allows.

### Conclusion

Strength:

- Clearly restates the three-decision decomposition.
- Limitations are present.

Needs revision:

- "Yields non-negative all-switch remaining-horizon counterfactual behavior on average" is technically true but fragile because the margin is tiny. Use a less assertive phrase.

Suggested replacement:

> The all-switch counterfactual analysis shows that learned switching is not systematically worse than continuing the old base on average, while representative cases and random-switch comparisons identify where the controller contributes path-level downside control.

## Five-Dimension Self-Review

### 1. Contribution

Status: pass with revisions.

The paper gives a meaningful decomposition of portfolio RL into base revision, base construction, and daily refinement. The strongest novelty is the controller-gated base memory, not simply hierarchical RL.

Main revision: sharpen contrast with adaptive interval and hierarchical portfolio baselines.

### 2. Writing Clarity

Status: pass with minor reproducibility risk.

The paper is readable and the active terminology now matches the code modules. The remaining writing risk is reproducibility detail: controller budget, stage-specific checkpoint selection, and baseline matching should stay explicit.

### 3. Experimental Strength

Status: pass with caveats.

The empirical result is credible as a trade-off story. It is not a clean "best on all metrics" story. The paper should lean into the honest interpretation: high return with controlled drawdown, strong learned-controller ablation, and market-dependent inner actor value.

### 4. Evaluation Completeness

Status: needs revision.

Main results, ablations, fixed-window controller tests, and interpretability are present. Missing or weak: uncertainty/seed protocol, switch statistics, turnover/cost sensitivity, and explicit baseline matching protocol.

### 5. Method Design Soundness

Status: pass with details needed.

The design is reasonable: fixed reference training for stable actors, then controller-based event switching. The main hidden risk is whether model selection and controller budget are tuned per market. This should be disclosed.

## Recommended Next Edit Plan

1. Clean claims in abstract, contribution bullet 3, controller interpretability, and conclusion.
2. Add a baseline matching paragraph/table note.
3. Add a compact implementation appendix table for controller/training/random-switch details.
4. Add closest-work contrast paragraph in Related Work.
5. Clean source hygiene: author placeholder, template sample blocks, `bibentry`, unused bib entries, BibTeX warnings.
6. Rebalance floats only after textual revisions, because the current ordering is acceptable but page 9 is sparse.
