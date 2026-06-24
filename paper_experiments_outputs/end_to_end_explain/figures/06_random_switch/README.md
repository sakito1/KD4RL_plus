# Matched-Count Random Timing Baseline 实验说明

## 这个实验是做什么的

该实验是 controller 的随机时机对照。它保留与真实 controller 相同量级的 free switch 预算，但把 switch 时点随机放到可自由决策日上，然后重新跑组合路径。

它不是随机交易策略，也不是随机组合权重；它只是在检验一个问题：controller 的效果是否来自“何时 switch”的时机选择，而不是简单因为切换次数更多。

## 图怎么看

- 灰色细线：多组随机 switch 时机的组合价值路径。
- 红色粗线：真实 Full Controller。
- 黑色虚线：Fixed HRL。

如果红色线明显高于灰色线，说明真实 controller 的 switch 时机优于随机时机。如果红色线收益不一定最高，但回撤明显更小，也可以说明 controller 的主要贡献是风险控制。

## 主要结论

SH 市场中，Full Controller 的总收益和 Sharpe 均高于所有 matched-count random timing baselines。Full Controller 总收益为 200.73%，随机对照平均为 101.25%；Full Sharpe 为 1.12，随机平均为 0.74。这支持 controller 的时机选择具有收益贡献。

NAS 市场中，Full Controller 的收益和 Sharpe 约处于随机对照的 54 分位，但最大回撤为 18.62%，优于随机对照平均的 23.53%。因此 NAS 不应强调随机对照下的收益优势，而应强调 controller 的回撤控制能力。

## 论文中怎么用

这组图可以作为主文或附录中的对照实验。建议英文图题使用 “Matched-Count Random Timing Baseline”，避免让读者误解为随机交易策略。
