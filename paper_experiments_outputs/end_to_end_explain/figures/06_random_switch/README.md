# Dense Fixed Holding-Window Timing Baseline 实验说明

## 这个实验是做什么的

该实验是 Dense Fixed Holding-Window Timing Baseline。对照对象是大量固定持仓期窗口：当前 Nasdaq-100 seed 49 和 CSI-300 seed 90 均覆盖 1d 到 60d，共 60 条 eval-only replay 路径。

实验检验的问题是：controller 的收益和风险表现是否只是来自某个固定持仓周期。如果 learned controller 相比大多数固定窗口具有更好的收益风险指标，尤其能降低最大回撤，就说明 controller 的价值不只是“固定几天换一次仓”，而是根据持仓状态动态决定是否切换。

## 图怎么看

- 灰紫色细线：不同固定持仓期窗口的累计财富曲线。
- 黑色虚线：参考 Fixed HRL 30d 路径。
- 红色粗线：learned controller 路径。
- 右侧柱形图：controller 在 TR、Sharpe、MDD、CR 上相对所有固定窗口的胜出比例。

## Nasdaq-100 统计结论

对应文件：

- `fig07_random_switch_comparison_nas_seed49.png`
- `fig07_fixed_window_timing_stats_nas_seed49.csv`
- `fig07_fixed_window_timing_stats_nas_seed49.md`

Learned controller 的 TR 为 265.53%，Sharpe 为 1.15，MDD 为 18.62%，CR 为 1.42。相比 60 个固定窗口，它在 TR 上优于 46/60，在 Sharpe 上优于 44/60，在 MDD 上优于 60/60，在 CR 上优于 57/60。

## CSI-300 统计结论

对应文件：

- `fig07_random_switch_comparison_sh_seed90.png`
- `fig07_fixed_window_timing_stats_sh_seed90.csv`
- `fig07_fixed_window_timing_stats_sh_seed90.md`

Learned controller 的 TR 为 204.99%，Sharpe 为 1.14，MDD 为 22.78%，CR 为 1.09。相比 60 个固定窗口，它在 TR 上优于 59/60，在 Sharpe 上优于 58/60，在 MDD 上优于 43/60，在 CR 上优于 57/60。

需要注意，少数事后挑选的固定窗口可以在个别指标上超过 controller，例如 Nasdaq-100 的 8d 固定窗口、CSI-300 的 50d 固定窗口。因此论文中不应写成 controller 在所有固定窗口和所有指标上绝对第一。更合理的结论是：controller 不需要事后调参选择固定窗口，却能取得高分位收益风险表现，并表现出更稳定的动态切仓能力。

## 论文中怎么用

建议英文图题使用 “Dense Fixed Holding-Window Timing Baseline”。可写成：

> Compared with a dense set of fixed holding-window baselines, the learned controller achieves high-percentile risk-return performance without ex-post selection of a constant holding period. This indicates that the controller learns state-dependent timing rather than relying on a manually tuned fixed window.
