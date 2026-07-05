# Inner Actor Base-Adjustment Resonance

本说明对应以下新增图：

- `inner_actor_base_adjustment_future_return_nas.png`
- `inner_actor_base_adjustment_future_return_sh.png`
- `inner_actor_base_adjustment_future_return_summary_display.csv`

这组图是当前更推荐使用的 inner actor 主解释图。

## 核心定义

这里的共振不是 raw weight 和同期价格走势的关系，而是：

> `inner tilt = executed weight - base weight`

其中：

- `base weight` 是 outer/controller 给出的基础组合权重。
- `executed weight` 是 inner actor 调整后真正执行的权重。
- `inner tilt > 0` 表示 inner actor 相对 base 加大该资产。
- `inner tilt < 0` 表示 inner actor 相对 base 降低该资产。

因此，这张图要解释的是：inner actor 相对 base 加大的资产，未来短期相对收益是否更强；相对 base 降低的资产，未来短期相对收益是否更弱。

## 图怎么看

第一行：`Future 5-day relative return`

- 表示每只资产未来 5 个交易日相对股票池平均水平的收益。
- 绿色表示未来相对更强。
- 红色表示未来相对更弱。

第二行：`Inner tilt = executed weight - base weight`

- 绿色表示 inner actor 相对 base 加大权重。
- 棕色表示 inner actor 相对 base 降低权重。
- 这行是最关键的解释对象。

第三行：`Executed portfolio weights after inner adjustment`

- 展示 inner actor 调整后最终执行的权重。
- 用来说明 tilt 不是抽象信号，而是实际进入组合执行的持仓变化。

第四行：`Asset-level contribution`

- 绿色横条表示该资产在整个窗口内的累计 tilt-return alignment 为正。
- 红色横条表示该资产在整个窗口内的累计 alignment 为负。
- 横条右侧的 `hit` 表示逐日方向一致比例；即使 hit 不是最高，少数幅度较大的正确 tilt 也可能带来正的累计贡献。
- 这比日度折线更直观：前三行看“未来相对收益”和“inner tilt”的颜色块是否同向，第四行直接总结“哪些资产的加减权方向最终贡献了正对齐”。

## 当前选中的 case

Nasdaq-100 case：

- 窗口：`2024-06-04` 到 `2024-07-17`
- 资产：`AMD.O`, `GILD.O`, `FAST.O`, `INTC.O`, `EXC.O`, `ODFL.O`
- `Mean corr(tilt, future relative return) = 0.46`
- `Positive alignment days = 73.33%`

CSI-300 case：

- 窗口：`2021-11-05` 到 `2021-12-16`
- 资产：`000408.SZ`, `000708.SZ`, `600111.SH`, `000786.SZ`, `600161.SH`, `000063.SZ`
- `Mean corr(tilt, future relative return) = 0.33`
- `Positive alignment days = 70.00%`

## 能说明什么

这组图说明 inner actor 不是简单复制 base 组合，也不是负责切仓。它是在 base 组合上做持仓期内的局部偏离：

- 对未来相对更强的资产增加 tilt。
- 对未来相对更弱的资产降低 tilt。
- 通过这些小幅相对权重调整，尝试捕捉持仓期内的横截面波动收益。

新版底部不再使用日度折线，而是把所有日度 `inner tilt × future relative return` 聚合到资产层面。因此它更适合说明 inner actor 的调整方向总体是否偏向未来相对更强的资产，而不是要求读者从噪声较强的时间序列中判断每一天是否正确。

更适合的论文表述是：

> The inner actor operates as an intra-holding adjustment module. Given the base portfolio from the outer/controller policy, it produces an executed portfolio by applying small relative tilts to asset weights. The base-adjustment resonance visualization shows that positive tilts tend to appear on assets with stronger future short-horizon relative returns, while negative tilts appear on relatively weaker assets. This suggests that the inner actor attempts to harvest cross-sectional volatility within a holding period rather than making explicit switching decisions.

中文可以写成：

> Inner actor 的作用不是决定何时切仓，而是在 base 组合上做持仓期内的相对权重修正。图中 `executed weight - base weight` 表示 inner actor 对每个资产的加权或减权。可以看到，在选取的 NASDAQ 和 CSI-300窗口中，inner actor 对未来 5 日相对收益更强的资产倾向于给出正 tilt，对未来相对较弱的资产倾向于给出负 tilt。这说明 inner actor 主要用于持仓期内捕捉资产间短期相对收益波动。

## 需要谨慎的地方

这仍然是局部 case 解释，不应写成全局强因果结论。

建议写：

- inner actor 在代表性窗口中体现出 base-adjustment 与未来相对收益的局部共振。
- 它承担持仓期内的 allocation refinement，而不是 controller 的 switch 功能。
- 它的收益贡献是辅助性的、市场依赖的。

不建议写：

- “inner actor 可以稳定预测未来收益。”
- “每次相对加权都能带来正收益。”
- “inner actor 是主要收益来源。”
