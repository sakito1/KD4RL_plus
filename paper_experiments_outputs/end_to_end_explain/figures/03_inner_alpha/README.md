# Inner Alpha 实验说明

## 这个实验是做什么的

该实验解释 inner module 在组合内部微调中的作用。它不直接替代最终收益指标，而是观察 inner 调整在每日路径上是否持续贡献正 alpha，以及这种贡献在不同市场和不同推理设置下是否稳定。

## 图怎么看

- `fig04_cumulative_inner_alpha_*.pdf`：累计 inner alpha。曲线持续上行表示 inner 调整在路径上累计贡献为正；曲线下行表示 inner 调整在该设置下没有稳定增加收益。
- `fig04b_inner_alpha_distribution_*.pdf`：Full Controller 下每日 inner alpha 的分布。红色均值线在 0 右侧表示平均日 alpha 为正，在 0 左侧表示平均日 alpha 为负。

这类图要和 `02_inference_ablation` 一起读。inner alpha 是局部解释量，最终结论仍应以组合收益、Sharpe 和回撤为准。

## 主要结论

在固定 HRL 设置下，inner 对两个市场都有正向贡献：NAS 中 Fixed HRL 相比 No-Inner Fixed 的总收益提高 7.03 个百分点；SH 中提高 11.77 个百分点。这说明 inner 在固定持仓框架中能改善组合内部配置。

在启用 controller 后，inner 的效果具有市场依赖性。NAS 中 Full Controller 相比 Controller+Outer 进一步提高收益并降低回撤；SH 中 Controller+Outer 的收益和 Sharpe 更高，而 Full Controller 略微降低回撤但收益回落。因此论文中不要写成“inner 总是提高最终收益”，更准确的说法是：inner 在固定持仓框架中提升组合内部配置能力，在 controller-active 框架中与切换频率和交易路径共同作用。

## 论文中怎么用

这组图更适合放附录或作为 inner 解释补充。主文中可以引用其结论，但不建议只用这组图证明最终性能。
