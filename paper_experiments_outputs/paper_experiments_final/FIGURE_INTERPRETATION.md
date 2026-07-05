# 论文实验图解读与结论说明

本文档用于说明 `paper_experiments_outputs/paper_experiments_final/` 下各类图应该怎么看、能够证明什么，以及哪些结论需要谨慎表述。建议论文写作时把结论分成三层：主实验证明整体有效，消融证明组件贡献，解释性实验说明 controller 和 inner actor 的行为机制。

## 1. 总体阅读原则

这些实验不是只证明某一个指标绝对第一，而是证明模型在收益、风险、换仓机制和反事实收益之间形成了更合理的风险收益权衡。

因此，建议论文中不要只写“所有指标最优”。更稳妥的表述是：

> Ours 在两个市场上表现出较强的风险调整收益。相较传统高收益策略，Ours 明显降低最大回撤；相较低回撤深度基线，Ours 保持了更高收益。消融和解释性结果进一步说明，controller 是主要的动态风险控制来源，inner actor 提供局部持仓修正能力，但其贡献具有市场依赖性。

这句话能覆盖一些不是绝对最优的情况，也更符合结果本身。

## 2. 主实验图怎么看

对应目录：`01_main_experiment/`

主要图：
- `main_equity_nas.png`
- `main_equity_sh.png`
- `main_metrics_nas.png`
- `main_metrics_sh.png`

### 图怎么看

收益曲线图展示不同方法的累计财富变化。柱形图展示四个指标：
- `Total return`：累计收益，越高越好。
- `Sharpe ratio`：单位波动下的收益，越高越好。
- `Max drawdown`：最大回撤，越低越好。
- `CR`：年化收益率除以最大回撤，越高表示单位回撤获得的年化收益越高。

读图时不要只看收益曲线终点，还要同时看 Sharpe、MaxDD 和 CR。金融实验里，如果一个方法收益高但回撤极大，它不一定是更好的投资策略；CR 可以直接补充说明收益是否足以补偿回撤风险。

### NASDAQ 结果怎么解释

在 NASDAQ 上，Ours 的总收益为 `265.53%`，高于 WMAMR 的 `264.39%` 和 Anticor 的 `259.97%`，同时最大回撤只有 `18.62%`，显著低于 WMAMR 的 `33.88%` 和 Anticor 的 `44.59%`。

DeepAries 的 Sharpe 为 `1.50`、最大回撤为 `10.83%`、CR 为 `1.78`，在风险指标上更强，但总收益只有 `162.96%`。Ours 的 CR 为 `1.42`，排名第二。所以这里不能说 Ours 在所有单项指标上都第一，而应该说：

> Ours 在 NASDAQ 上取得最高或接近最高的累计收益，同时显著压低回撤，属于收益和风险之间更均衡的结果。

### CSI-300 结果怎么解释

在 CSI-300市场上，DeepTrader 的总收益为 `212.81%`，高于 Ours 的 `204.99%`。但 DeepTrader 的 Sharpe 为 `1.03`、最大回撤为 `31.86%`、CR 为 `0.83`，而 Ours 的 Sharpe 为 `1.14`、最大回撤为 `22.78%`、CR 为 `1.09`。

所以这里更合理的结论不是“收益绝对最高”，而是：

> Ours 在 CSI-300市场上牺牲了少量绝对收益，但获得了更好的风险调整收益和更低回撤。相比 DeepTrader，Ours 的收益略低，但 Sharpe 和 CR 更高、最大回撤更小，说明策略更稳健。

这也是论文中解释“不是所有指标都最优”的核心逻辑：投资策略的好坏不只由最终收益决定，还取决于收益是否需要承担过大的回撤和波动。

## 3. 消融实验图怎么看

对应目录：`02_ablation/`

主要图：
- `ablation_equity_nas.png`
- `ablation_equity_sh.png`
- `ablation_metrics_nas.png`
- `ablation_metrics_sh.png`

### 图怎么看

消融实验比较：
- `Outer-only`：只有外层 portfolio decision。
- `Outer + Inner`：加入 inner actor，但没有动态 controller。
- `Outer + Controller`：加入 controller，但去掉 inner actor。
- `Fixed 5d/10d/20d/30d/60d`：固定周期切仓。
- `Ours`：完整 HRL + controller。

这个实验的重点是区分“动态切仓”是不是比固定切仓更有效，以及 inner actor 是否真的提供了持仓修正能力。

### Controller 的贡献

Controller 的证据比较强。

在 NASDAQ 上：
- `Outer-only` 总收益 `220.42%`，最大回撤 `32.09%`。
- `Outer + Controller` 总收益 `237.50%`，最大回撤 `21.24%`。
- `Ours` 总收益 `265.53%`，最大回撤 `18.62%`。

这说明 controller 不是简单增加交易次数，而是在收益提高的同时明显降低最大回撤。

在 CSI-300市场上：
- `Outer-only` 总收益 `147.05%`，Sharpe `0.94`。
- `Outer + Controller` 总收益 `237.77%`，Sharpe `1.22`。
- `Ours` 总收益 `204.99%`，Sharpe `1.14`。

这里 `Outer + Controller` 的收益高于 Ours，说明 inner actor 在 CSI-300市场上并没有进一步提高绝对收益。但这不削弱 controller 的结论，反而说明 controller 是主要收益改善来源。

### 固定窗口 controller 的作用

固定窗口实验用于排除一个质疑：模型是否只是“多换仓就更好”。

结果显示，固定 5/10/20/30/60 天切仓并不能稳定复制 Ours 的表现。例如 NASDAQ 上，Ours 总收益 `265.53%`、最大回撤 `18.62%`，明显优于所有固定窗口变体。CSI-300市场上，Ours 也明显高于固定窗口变体的收益。

因此可以写：

> Dynamic controller is not equivalent to a fixed rebalancing frequency. Its advantage comes from state-dependent switching rather than switching more often or less often.

## 4. Controller 解释性图怎么看

对应目录：`03_controller_interpretability/`

主要图：
- `controller_case_nas_01.png`
- `controller_case_nas_02.png`
- `controller_case_sh_01.png`
- `controller_case_sh_02.png`
- `switch_counterfactual_distribution_nas.png`
- `switch_counterfactual_distribution_sh.png`
- `switch_remaining_horizon_counterfactual_distribution_nas.png`
- `switch_remaining_horizon_counterfactual_distribution_sh.png`
- `fixed_window_comparison_nas.png`
- `fixed_window_comparison_sh.png`
- `controller_probability_resonance_nas.png`
- `controller_probability_resonance_sh.png`

## 4.1 Case 图怎么看

`controller_case_*.png` 每张图有三行。

第一行是 30 日窗口内的实际财富曲线：
- 红线是 Ours 的窗口财富。
- 浅红阴影表示回撤区域。
- 竖线表示窗口内所有 switch。
- 加粗竖线和红点表示关键 switch。

第二行是关键 switch 时刻的反事实比较：
- 灰线：继续持有旧组合。
- 绿色线：切到新组合。
- 如果绿色线高于灰线，说明 switch 后的未来收益更好。
- 如果绿色线回撤更浅，说明 switch 降低了未来风险。

第三行是 controller 的行为信号：
- 蓝线是 `exit probability`，即 controller 对“结束当前持仓/切仓”的倾向。
- 绿色/红色柱表示 switch 相对 hold 的未来优势。
- 竖线标出实际 switch 发生的位置。

### Case 证明了什么

这些 case 的作用不是证明每一次 switch 都完美，而是展示 controller 在关键风险窗口中确实做了有经济意义的切换。

例如：
- Nasdaq-100 case 1：继续持有的 20 日未来收益为 `-6.05%`，switch 后为 `-2.63%`，最大回撤从 `8.82%` 降到 `5.51%`。
- Nasdaq-100 case 2：继续持有为 `3.79%`，switch 后为 `8.29%`，最大回撤从 `5.34%` 降到 `2.53%`。
- CSI-300 case 1：继续持有为 `-10.74%`，switch 后为 `4.31%`，最大回撤从 `13.59%` 降到 `8.23%`。
- CSI-300 case 2：继续持有为 `2.38%`，switch 后为 `10.53%`，最大回撤从 `8.60%` 降到 `3.81%`。

这些例子可以支撑论文中的直观解释：

> The controller switches when the continuation portfolio is exposed to unfavorable short-term return or drawdown risk. In selected cases, switching improves the 20-day counterfactual return and reduces future drawdown.

## 4.2 Switch probability 和实际收益风险是什么关系

这里要谨慎解释。`exit probability` 不是未来收益的直接预测值，而是 controller 基于当前状态给出的“是否结束当前持仓”的倾向。它与收益风险的关系不是简单线性相关，而是通过决策机制间接体现。

可以分三层解释。

第一层是局部 case 关系：
- 当当前持仓的未来反事实收益更差或回撤更大时，controller 往往给出较高的 exit probability，并在窗口中触发 switch。
- 在 case 图第三行，较高的 exit probability 和正的 switch advantage 同时出现，说明 controller 的 switch 不是机械固定周期点，而是和未来收益/风险恶化有关。

第二层是 switch 点反事实统计分布关系：
- `switch_counterfactual_distribution_*.png` 比较所有实际 free switch 点之后 20 日的“继续持有”和“切仓”反事实收益分布。
- NASDAQ 上，实际 switch 点的平均收益优势接近 `0.00%`，但最大回撤改善为正的比例为 `51.95%`。
- CSI-300市场上，实际 switch 点的平均收益优势为 `0.06%`，收益改善比例为 `53.92%`，最大回撤改善比例为 `54.90%`。

第三层是原持仓剩余期限的全量反事实分布：
- `switch_remaining_horizon_counterfactual_distribution_*.png` 是对你要求的“所有 switch 与反事实收益分布比较”的更严格版本。
- 不能直接用真实路径从 switch 点算到下一次切仓，因为真实路径中可能继续发生多次 switch，会混入后续 controller 决策，导致单个 switch 与旧组合继续持有不可比。
- 当前做法是在每个真实 controller free switch 时刻，把两个候选动作都冻结：一条是“继续旧组合”，另一条是“切到新组合”。两条都持有到切仓前旧组合原本的剩余持仓期终点，即 `max_hold - duration_before_decision`。
- 这样每个 switch 都在同一个起点、同一个 horizon、无后续 switch 干扰下比较，才是可比的 decision-level counterfactual。
- 该图不是只挑 case，而是全量比较：NASDAQ 覆盖 `231/231` 个 free switch，CSI-300覆盖 `102/102` 个 free switch。
- NASDAQ：switch 冻结收益均值 `3.29%`，继续旧组合反事实均值 `3.24%`，平均优势 `0.05%`，正收益优势比例 `50.22%`。
- CSI-300：switch 冻结收益均值 `4.43%`，继续旧组合反事实均值 `4.39%`，平均优势 `0.04%`，正收益优势比例 `48.04%`。

这些分布结果说明：从所有 switch 的冻结反事实统计看，平均收益优势为正但很温和，并不是每一次 switch 都显著优于继续持有。这个结果应该谨慎解释，不能写成“所有 switch 都带来收益提升”。更稳妥的解释是：

> Across all switches, many decisions are small rebalancing or maintenance trades, so the average counterfactual gain is modest. However, the distribution and selected cases show that controller can identify important downside windows where switching materially reduces losses or drawdowns.

第四层是最终投资结果关系：
- `fixed_window_comparison_*.png` 比较真实 controller 和 5/10/20/30/60 日固定持仓窗口。
- NASDAQ 上，真实 controller 的 TR、Sharpe、MDD 和 CR 均优于固定窗口变体。
- CSI-300市场上，真实 controller 的 TR、Sharpe 和 CR 均高于固定窗口变体；部分固定窗口的 MDD 略低，但收益牺牲很大。

这说明 controller 的 switch 不是“选择一个固定持仓周期就能得到同样结果”。即使全量冻结反事实的平均优势不大，真实 controller 在最终投资路径上仍然表现出更好的风险收益结构。论文里建议把这四类证据串起来说：case 证明关键风险窗口有效，remaining-horizon distribution 证明全量 switch 都被纳入可比反事实比较，fixed-window comparison 证明不是固定周期换仓即可复现，主实验/消融证明最终风险收益表现提升。

### 为什么概率相关性可能不强

`controller_probability_resonance_*.png` 中的相关性不是特别强，这是合理的。原因包括：
- controller 的动作受最短/最长持仓期限制，不是每天都能自由切换。
- exit probability 同时受到收益、风险、交易成本、隐藏状态和当前持仓持续时间影响。
- `switch_advantage_20` 是事后 20 日反事实指标，而 controller 决策时只能基于当时可见状态。
- 很多 switch 是小幅再平衡，不一定对应剧烈下跌窗口。

因此，论文里不要写“exit probability 与未来收益优势强线性相关”。更稳妥的写法是：

> Exit probability is locally aligned with switching opportunities in high-risk windows, while the overall relation is nonlinear and constrained by holding-period rules and transaction costs.

## 5. Inner actor 解释性图怎么看

对应目录：`04_inner_actor_interpretability/`

主要图：
- `inner_actor_base_adjustment_future_return_nas.png`
- `inner_actor_base_adjustment_future_return_sh.png`
- `inner_actor_alpha_nas.png`
- `inner_actor_alpha_sh.png`
- `inner_actor_weight_stack_nas.png`
- `inner_actor_weight_stack_sh.png`

## 5.1 Inner actor 的证据为什么比 controller 弱

Inner actor 的解释性确实比 controller 弱，这是正常的，因为 inner actor 不是一个显式的“是否切仓”模块，而是一个局部权重修正模块。它的作用更像 execution refinement：在 outer/controller 给定方向后，对持仓权重做细粒度调整。

所以 inner actor 不适合被解释成“独立发现下跌并切仓”。这个角色应该留给 controller。Inner actor 更适合解释为：

> Inner actor refines portfolio weights within a holding segment. It provides local allocation flexibility rather than acting as the primary switching mechanism.

## 5.2 Inner actor 图怎么看

最推荐作为论文主解释图的是 `inner_actor_base_adjustment_future_return_*.png`。这张图使用更严格的定义：

> inner tilt = executed weight - base weight

其中正 tilt 表示 inner actor 相对 base 加大某资产，负 tilt 表示相对 base 降低某资产。第一行展示未来 5 日相对收益，第二行展示 inner tilt，第三行展示 inner 调整后的执行权重，第四行把日度 `inner tilt × future relative return` 聚合为资产级 contribution。读图重点是先看前两行颜色块是否同向，再看底部哪些资产的累计 contribution 为正：正 contribution 表示 inner actor 在该资产上的加减权总体上与未来相对收益方向一致。当前 Nasdaq-100 case 的 `corr(tilt, future relative return)` 为 `0.46`、正对齐天数为 `73.33%`；CSI-300 case 的相关性为 `0.33`、正对齐天数为 `70.00%`。

这张图最贴合“inner actor 在 base 持仓上做持仓期内微调，以捕捉未来短期相对收益波动”的解释。

`inner_actor_alpha_*.png` 有三行：

第一行比较 `Ours` 和 `Outer + Controller` 的财富曲线，用于观察加入 inner actor 后最终路径是否改善。

第二行展示累计 inner alpha。这个指标表示执行组合相对基础组合的局部收益修正。它不是每天都为正，也不应该被理解成稳定独立 alpha。

第三行展示 rolling executed return、rolling base return 和 turnover。读图重点是看 turnover 增大的时段是否对应收益波动或执行收益与基础收益的偏离。如果存在共振，说明 inner actor 在市场状态变化时确实改变了执行持仓。

`inner_actor_weight_stack_*.png` 展示关键 switch 时继续持有候选组合和切仓候选组合的 top weights。它说明模型在 switch 时不是只改变一个抽象信号，而是实际改变了组合中资产权重。

## 5.3 Inner actor 的合理结论

消融结果显示：
- NASDAQ：`Outer-only` 总收益 `220.42%`，`Outer + Inner` 为 `227.43%`，说明 inner actor 单独加入时有边际提升。
- CSI-300：`Outer-only` 总收益 `147.05%`，`Outer + Inner` 为 `158.99%`，也有边际提升。

但在加入 controller 后：
- NASDAQ：`Outer + Controller` 为 `237.50%`，Ours 为 `265.53%`，inner actor 进一步提升收益并降低回撤。
- CSI-300：`Outer + Controller` 为 `237.77%`，Ours 为 `204.99%`，inner actor 没有进一步提高收益，但 Ours 的最大回撤 `22.78%` 略低于 `Outer + Controller` 的 `23.29%`。

因此，最合理的结论是：

> Inner actor has a positive but secondary and market-dependent effect. It improves the outer-only policy in both markets and further improves NASDAQ under the controller. In CSI-300, however, the controller-only variant achieves higher return, indicating that the inner actor may trade off upside for smoother execution. Therefore, the dominant evidence should be attributed to the controller, while inner actor is interpreted as a local portfolio refinement module.

这比直接说“inner actor 一定有效”更可信。

## 6. 非最优结果应该怎么写

论文中如果遇到某个指标不是最优，建议按以下逻辑解释。

第一，明确评价目标不是单一收益最大化：

> Portfolio learning is a multi-objective problem. A strategy with higher terminal wealth but substantially larger drawdown may be less desirable than a slightly lower-return strategy with better risk-adjusted performance.

第二，强调 Ours 是风险收益折中：

> Ours is not always the best on every single metric, but it is consistently competitive in return while controlling drawdown and improving Sharpe.

第三，避免把 inner actor 说成主贡献：

> The ablation suggests that the controller is the major contributor to dynamic risk control. The inner actor contributes local weight refinement, and its effect is conditional on market regime.

第四，对 CSI-300的 `Outer + Controller` 高于 Ours 要诚实说明：

> In CSI-300, the controller-only variant obtains higher total return than the full model. This indicates that the inner actor is not uniformly beneficial after controller integration. The full model remains risk-adjusted competitive, while the component study highlights controller as the most robust source of improvement.

这样写不会回避结果，也不会让审稿人觉得过度包装。

## 7. 可以直接写进论文的结论文字

### 主实验结论

> The proposed method achieves a favorable risk-return tradeoff across both markets. On NASDAQ, it reaches the highest cumulative return among matched baselines while substantially reducing maximum drawdown compared with high-return traditional strategies. On CSI-300, although DeepTrader obtains slightly higher cumulative return, the proposed method achieves a higher Sharpe ratio and lower maximum drawdown, indicating more stable risk-adjusted performance.

### 消融结论

> The ablation study shows that the controller is the major source of improvement. Compared with fixed-window rebalancing, the learned controller adapts switching decisions to market states and achieves better risk-return performance. The inner actor provides additional local allocation flexibility, improving the outer-only policy in both markets, but its marginal effect after controller integration is market-dependent.

### Controller 解释性结论

> Case studies show that controller switches often occur before or during unfavorable continuation windows. Counterfactual trajectories demonstrate that switching can improve subsequent return and reduce future drawdown. Fixed-window comparisons further confirm that the learned controller is not equivalent to a manually chosen constant holding period.

### Inner actor 解释性结论

> Inner actor should be interpreted as a local portfolio refinement module rather than the primary switching mechanism. Weight-stack visualizations show that it changes the composition of candidate portfolios, while alpha and turnover traces indicate that it reacts to changing return conditions. Its contribution is meaningful but secondary to the controller and should be described as conditional rather than universally dominant.

## 8. 最推荐的论文叙述顺序

建议论文实验部分按下面顺序讲：

1. 主实验：Ours 在两市场上整体风险收益表现强。
2. 消融实验：controller 是主要贡献，动态切仓优于固定窗口。
3. Controller case：展示具体什么时候 switch，以及 switch 如何避免继续持有导致的收益恶化或回撤扩大。
4. Fixed-window comparison：证明不是固定持仓周期带来的效果。
5. Counterfactual distribution：说明统计意义上 switch 后未来收益/回撤分布更优或更稳。
6. Inner actor：作为辅助模块展示权重迁移和局部收益修正，但不夸大为最主要贡献。

这样写最稳，也最符合目前图和表所能支持的证据强度。
