# Inner Actor Interpretability

这个目录解释 inner actor 的作用。`inner_actor_alpha_*.png` 比较 Ours 与 Outer + Controller，并展示累计 inner alpha、rolling executed/base return 和 turnover。它用于说明 inner actor 不是静态噪声，而是在收益波动和持仓调整之间产生可观察的贡献。

`inner_actor_weight_stack_*.png` 展示关键 switch 时“继续持有候选组合”和“切到新组合候选组合”的 top-weight 分布。读图时看两条堆叠条的资产权重变化：权重迁移对应 controller 决策后的新持仓方向。
