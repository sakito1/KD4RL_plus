# Switch Event Study 实验说明

## 这个实验是做什么的

该实验把所有 controller switch 事件汇总，比较 switch 前后的 20 日收益、继续持有反事实和切换反事实。它用于观察 switch 行为的平均效果和分布异质性。

## 图怎么看

- `fig06_switch_event_study_*.pdf`：展示 switch 前 20 日收益、switch 后实际路径、继续持有反事实和切换反事实的平均 20 日收益。切换反事实高于继续持有反事实时，说明平均意义上 switch 改善了后续路径。
- `fig06b_switch_avoided_loss_distribution_*.pdf`：展示 avoided loss 的分布。右侧长尾表示存在一些 switch 明显避免了后续损失；分布跨过 0 表示不是所有 switch 都有效。

这组图应按“事件平均和分布”来读，不应把它解释成每个 switch 都有正收益。

## 主要结论

SH 市场中，Full Controller 的 switch 前 20 日平均收益为 -4.94%，switch 后实际 20 日平均收益为 3.30%。这说明 SH 的 switch 往往出现在前期路径走弱之后，整体上具有风险暴露调整的含义。

NAS 市场中，全体 switch 的平均 avoided loss 接近 0，说明 NAS 的 switch 行为更频繁、更异质，不能用“所有 switch 平均都避免损失”来讲。NAS 更适合通过 case window 展示多个 switch 如何共同压低回撤。

## 论文中怎么用

这组图适合作为补充证据，尤其用于说明 switch 效果存在异质性。主文中不要只用 NAS 的全事件均值来证明 controller 有效；NAS 的解释重点应放在消融、随机时机对照和 case window。
