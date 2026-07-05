# Related Work Slide (PPT-ready)

> Color guide for manual PPT formatting: use blue for phrases wrapped by `[blue]...[/blue]`, and purple for citations wrapped by `[purple]...[/purple]`. This slide follows the "method family -> representative idea -> limitation -> our position" structure.

## Related Work

### Classical Portfolio Optimization

- Build portfolio allocation models from [blue]historical return-risk estimates[/blue] and solve for optimal weights under mean-variance, asset-pricing, or rule-based rebalancing assumptions [purple](Markowitz 1952; Sharpe 1964; Perold and Sharpe 1988; Sullivan 2008)[/purple].
- Limitations: (i) rely heavily on [blue]stationary or single-period assumptions[/blue]; (ii) fixed or heuristic rebalancing can be misaligned with non-stationary markets; (iii) transaction costs make frequent manual rebalancing less reliable.

### Deep RL Portfolio Management

- Model portfolio management as a [blue]sequential decision-making problem[/blue] and learn allocation policies directly from market data [purple](Jiang, Xu, and Liang 2017; Liu et al. 2022)[/purple].
- Representative DRL methods improve market representation, risk-return balancing, cross-asset attention, and knowledge-guided exploration [purple](Wang et al. 2021; Wang et al. 2019; Ding et al. 2018)[/purple].
- Limitations: many methods still compress [blue]medium-term base allocation[/blue] and [blue]daily execution adjustment[/blue] into one monolithic portfolio action, making it hard to explain when a base portfolio should be kept or revised.

### Hierarchical and Adaptive Rebalancing

- Recent methods explore [blue]hierarchical portfolio selection[/blue] and [blue]adaptive rebalancing intervals[/blue], making portfolio policies more flexible than fixed-window rules [purple](Kim et al. 2023; Kim et al. 2025)[/purple].
- DeepAries is especially close because it studies [blue]adaptive rebalancing interval selection[/blue], while HADAPS studies hierarchical adaptive multi-asset selection.
- Limitations: existing adaptive methods usually do not explicitly maintain a [blue]persistent base portfolio memory[/blue] that can be held, drifted with prices, compared with a candidate base, and replaced by a learned controller.

### Position of CMTFlow

- CMTFlow separates portfolio decisions into [blue]when to revise[/blue] (controller), [blue]what base portfolio to hold[/blue] (outer actor), and [blue]how to refine daily exposure[/blue] (inner actor).
- This makes the model different from both fixed-window rebalancing and single-action DRL: the controller learns base-revision timing, the outer actor proposes segment-level candidates, and the inner actor performs support-constrained daily tilting inside the active base.

---

## Short Version For A Dense PPT Page

**Classical Portfolio Optimization**

- Optimize weights from [blue]historical return-risk estimates[/blue] [purple](Markowitz 1952; Sharpe 1964; Perold and Sharpe 1988)[/purple].
- Limitations: [blue]single-period/static assumptions[/blue], fixed rebalancing, weak adaptation to regime shifts.

**Deep RL Portfolio Management**

- Learn long-horizon portfolio policies as [blue]sequential decisions[/blue] [purple](Jiang, Xu, and Liang 2017; Liu et al. 2022; Wang et al. 2021)[/purple].
- Limitations: often treats allocation as one [blue]daily monolithic action[/blue], mixing base construction and daily refinement.

**Hierarchical / Adaptive Methods**

- Use [blue]hierarchical selection[/blue] or [blue]adaptive rebalancing intervals[/blue] [purple](Kim et al. 2023; Kim et al. 2025)[/purple].
- Limitations: rarely model an explicit [blue]base portfolio memory[/blue] that is held, drifted, compared, and replaced by a controller.

**CMTFlow**

- Learns [blue]when to revise[/blue], [blue]what base to hold[/blue], and [blue]how to refine daily[/blue] in one coordinated decision flow.

---

## Speaker Note

The key distinction is not that CMTFlow simply adds another RL policy. Existing DRL methods mainly focus on generating portfolio weights, and adaptive methods such as DeepAries focus on when to rebalance. CMTFlow instead keeps an active base portfolio as a stateful memory: the controller decides whether to hold or replace it, the outer actor proposes the candidate base, and the inner actor makes daily tilts within that base. This directly matches the practical portfolio workflow of keeping a core position, revising it when risk-return evidence changes, and making short-term exposure adjustments during the holding period.
