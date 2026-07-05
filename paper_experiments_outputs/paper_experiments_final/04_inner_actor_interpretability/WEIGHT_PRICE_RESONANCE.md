# Inner Actor Weight-Price Resonance

本说明对应以下新增图：

- `inner_actor_weight_price_resonance_heatmap_nas.png`
- `inner_actor_weight_price_resonance_heatmap_sh.png`
- `inner_actor_weight_price_resonance_nas.png`
- `inner_actor_weight_price_resonance_sh.png`
- `inner_actor_weight_price_resonance_summary_display.csv`

注意：这组图展示的是 raw executed weight 与相对价格的局部关系。若要严格解释“inner actor 相对 base 加大/减小权重是否对应未来收益强弱”，请优先使用 `INNER_BASE_ADJUSTMENT_RESONANCE.md` 和 `inner_actor_base_adjustment_future_return_*.png`。后者使用 `executed weight - base weight` 作为 inner actor 的真实调整量。

## 这组图想说明什么

这组图用于解释 inner actor 在持仓期内做什么。更合适的表述不是“inner actor 独立产生稳定 alpha”，而是：

> Inner actor 是持仓期内的权重微调模块。它在 controller 不切仓的情况下，根据资产间的相对价格波动调整组合暴露，使权重变化与价格相对强弱产生局部共振。

也就是说，controller 解释“什么时候换一个持仓段”，inner actor 解释“在这个持仓段内部如何调权捕捉波动”。

## 图怎么看

热力图版本建议作为论文主图。

第一行是相对价格变化：

- 每一行是一只资产。
- 绿色表示该资产相对窗口内股票池走强。
- 红色表示该资产相对走弱。

第二行是同一批资产的 inner actor 权重：

- 深蓝表示权重较高。
- 浅色表示权重较低。
- 读图时把第一行和第二行同一资产横向对齐看。如果资产走强时权重更高、资产走弱时权重降低，就说明权重和价格有共振。

第三行是日度共振指标：

- 绿色柱表示当天权重变化方向与相对收益方向一致。
- 红色柱表示当天调权方向与相对收益方向相反。
- 灰线是日度权重 turnover，表示 inner actor 的调权强度。

图中文字给出两个局部统计：

- `Mean corr(weight, relative price)`：持仓期内权重水平与相对价格水平的平均相关性。
- `Positive resonance days`：日度调权方向与相对收益方向一致的天数比例。

## 选中的 case

NASDAQ case：

- 窗口：`2021-08-11` 到 `2021-10-06`
- 资产：`CCEP.O`, `ADSK.O`, `COST.O`, `BKNG.O`, `MNST.O`, `CPRT.O`
- `Mean corr(weight, relative price) = 0.56`
- `Positive resonance days = 55.00%`

这个 case 里，CCEP 和 ADSK 相对走弱后权重明显退出；COST 和 CPRT 保持较高暴露；BKNG 相对走强但权重没有完全追随，因此这个 case 不是“完美追涨”，而是说明 inner actor 在多个资产之间做局部暴露迁移。

CSI-300 case：

- 窗口：`2023-01-05` 到 `2023-02-22`
- 资产：`600875.SH`, `000733.SZ`, `600183.SH`, `600111.SH`, `600219.SH`, `600150.SH`
- `Mean corr(weight, relative price) = 0.51`
- `Positive resonance days = 70.00%`

这个 case 更直观。600183、600111、600219 等相对走强资产在窗口内保持了更高权重，而 600875、000733 等走弱资产的权重有所降低，说明 inner actor 对资产相对强弱变化有响应。

## 可以写进论文的结论

> The weight-price resonance visualization explains the role of the inner actor inside a holding period. Unlike the controller, which decides when to switch the holding segment, the inner actor continuously adjusts portfolio weights within the segment. In the selected NASDAQ and CSI-300 windows, asset weights exhibit positive local correlation with relative price movements. Assets with stronger relative performance tend to receive higher or more persistent exposure, while weakening assets are reduced. This suggests that the inner actor performs intra-holding allocation refinement and attempts to harvest short-term cross-sectional volatility, rather than acting as an independent market-timing module.

中文表述可以写成：

> 权重-价格共振图进一步说明了 inner actor 的作用。与 controller 负责决定何时切换持仓段不同，inner actor 在持仓期内部连续调整资产权重。选取的 NASDAQ 和 CSI-300窗口显示，资产权重与相对价格走势存在正向局部相关：相对走强资产获得更高或更持续的权重暴露，相对走弱资产的权重被降低。这说明 inner actor 更像是持仓期内的局部权重修正模块，用于捕捉资产间短期波动收益，而不是独立的市场择时模块。

## 需要谨慎的地方

这组图是局部 case 解释，不应写成全局强因果结论。更稳妥的说法是：

- inner actor 在代表性窗口中表现出权重-价格共振。
- 它提供持仓期内的局部 allocation refinement。
- 它的贡献是辅助性的、市场依赖的，不能替代 controller 的动态切仓解释。

不建议写：

- “inner actor 每次调权都能赚取收益。”
- “inner actor 具有稳定独立 alpha。”
- “权重变化可以强预测未来价格。”
