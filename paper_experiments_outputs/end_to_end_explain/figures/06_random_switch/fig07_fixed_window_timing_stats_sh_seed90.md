# Dense Fixed Holding-Window Timing Baseline (CSI-300, seed 90)

这张图展示 Dense Fixed Holding-Window Timing Baseline。比较对象是大量不同固定持仓期窗口：1d 到 60d，共 60 个 fixed holding-window baselines。

## 怎么看图

- 灰紫色细线：不同固定持仓期窗口的累计财富曲线。
- 红色线：learned controller 的实际累计财富曲线。
- 左图统计框：controller 相比固定窗口集合的胜出次数。
- 右图柱形面板：controller 在 TR、Sharpe、MDD 和 CR 上相对 60 个固定窗口的胜出比例。

## 统计结论

- Learned controller 的 TR 为 204.99%，Sharpe 为 1.14，MDD 为 22.78%，CR 为 1.09。
- 固定窗口中最高 TR 来自 50d，TR 为 292.16%；controller 在 TR 上优于 59/60 个固定窗口。
- 固定窗口中最高 Sharpe 来自 50d，Sharpe 为 1.40；controller 在 Sharpe 上优于 58/60 个固定窗口。
- 固定窗口中最低 MDD 来自 27d，MDD 为 17.17%；controller 在 MDD 上优于 43/60 个固定窗口。
- 固定窗口中最高 CR 来自 27d，CR 为 1.42；controller 在 CR 上优于 57/60 个固定窗口。

注意：CSI-300 上可能存在少数事后挑选的固定窗口（例如 50d）在部分指标上高于 controller。因此论文中不应写成“controller 在所有固定窗口和所有指标上都是第一”。更合理的结论是：controller 不需要事后选择固定窗口，却在固定窗口集合中取得高分位表现，并在关键风险指标上体现出更稳定的控制能力。

## 可写入论文的表述

Compared with a dense set of fixed holding-window baselines, the learned controller achieves high-percentile risk-return performance without ex-post selection of a constant holding period. This result indicates that the controller learns state-dependent timing for revising the active base portfolio rather than relying on a manually tuned fixed window.
