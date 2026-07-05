# 论文实验结论与写作文本

本文档面向论文写作，结合当前输出图和表格总结实验能够支持的结论。建议论文中把证据链写成三层：主实验说明整体风险收益表现，消融实验说明 controller 是主要有效组件，解释性实验说明 controller 的 switch 不是固定持仓窗口可以替代的规则，而是在部分高风险窗口中降低了继续持有带来的收益劣化和回撤风险。

输出图均位于 `paper_experiments_outputs/paper_experiments_final/`。论文排版时建议图题中使用市场名和实验类型，不要出现 seed、checkpoint、超参数等实现细节。

## 1. 总体结论

从两个市场的实验结果看，Ours 并不是单纯追求最高终点收益的策略，而是在累计收益、Sharpe ratio、最大回撤和 CR 之间取得了更稳健的风险收益权衡。CR 定义为年化收益率除以最大回撤，用于衡量单位回撤下获得的年化收益。在 NASDAQ 市场上，Ours 获得最高或接近最高的累计收益，同时显著降低了传统高收益策略的最大回撤，并取得第二高 CR；在 CSI-300市场上，Ours 的累计收益略低于 DeepTrader，但 Sharpe ratio 更高、最大回撤更低、CR 更高，说明其收益质量更好。

消融结果进一步表明，controller 是模型中最关键的有效组件。加入 controller 后，模型不只是增加了换仓次数，而是通过状态依赖的切仓机制改善风险收益表现，并且明显优于固定 5/10/20/30/60 天切仓规则。Inner actor 的贡献相对更弱，也更依赖市场：它更适合被解释为局部持仓权重修正模块，而不是主要的风险规避模块。

解释性结果说明，controller 的价值主要体现在关键风险窗口。个例图显示，当继续持有旧组合会带来较差未来收益或更大回撤时，controller 的 switch 能够改善 20 日反事实收益并降低未来最大回撤。全量 switch 的 remaining-horizon 反事实分布显示，所有真实 switch 都被纳入了同一起点、同一 horizon 的可比反事实比较；平均优势为正但幅度较小，说明大量 switch 属于小幅再平衡或维护性决策。因此，论文中应避免写成“每一次 switch 都显著更优”，更合理的表述是：controller 在整体路径上优于多个固定持仓窗口，并且在关键下跌窗口能够产生明确的风险缓释作用。

## 2. 主实验结论

对应图：

- `01_main_experiment/main_equity_nas.png`
- `01_main_experiment/main_equity_sh.png`
- `01_main_experiment/main_metrics_nas.png`
- `01_main_experiment/main_metrics_sh.png`
- `tables/main_experiment_metrics_display.csv`

### NASDAQ 市场

在 NASDAQ 市场上，Ours 的累计收益为 `265.53%`，高于 WMAMR 的 `264.39%`、Anticor 的 `259.97%` 以及其他深度学习 baseline。同时，Ours 的最大回撤为 `18.62%`，明显低于 WMAMR 的 `33.88%` 和 Anticor 的 `44.59%`。这说明 Ours 不是通过承担更大下行风险换取收益，而是在保持高收益的同时控制了回撤。

需要注意的是，DeepAries 在 NASDAQ 上的 Sharpe ratio 为 `1.50`、最大回撤为 `10.83%`、CR 为 `1.78`，风险指标优于 Ours，但其累计收益只有 `162.96%`。Ours 的 CR 为 `1.42`，排名第二。因此，论文中不应写“Ours 在所有指标上均最优”，而应强调：

> Ours 在 NASDAQ 市场上取得了最高或接近最高的累计收益，并显著降低了高收益 baseline 的最大回撤，表现出更均衡的风险收益特征。

### CSI-300市场

在 CSI-300市场上，DeepTrader 的累计收益为 `212.81%`，高于 Ours 的 `204.99%`。但 DeepTrader 的 Sharpe ratio 为 `1.03`，最大回撤为 `31.86%`，CR 为 `0.83`；Ours 的 Sharpe ratio 为 `1.14`，最大回撤为 `22.78%`，CR 为 `1.09`。因此，Ours 虽然没有取得最高终点收益，但用更低的最大回撤获得了接近最高的收益。

这一结果适合写成：

> In the CSI-300 market, Ours sacrifices a small amount of terminal return compared with DeepTrader, but achieves a higher Sharpe ratio and a substantially lower maximum drawdown. This indicates that the learned controller improves the quality of returns rather than merely increasing terminal wealth.

### 可写入论文的主实验结论

> The main results show that Ours achieves a more favorable risk-return trade-off across both markets. In NASDAQ, Ours delivers the highest cumulative return while keeping the maximum drawdown much lower than high-return traditional baselines and achieving the second-best CR. In the CSI-300 market, Ours obtains slightly lower cumulative return than DeepTrader, but improves Sharpe ratio, reduces maximum drawdown, and increases CR. These results suggest that the proposed hierarchical policy with controller improves the stability and risk-adjusted performance of portfolio management.

## 3. 消融实验结论

对应图：

- `02_ablation/ablation_equity_nas.png`
- `02_ablation/ablation_equity_sh.png`
- `02_ablation/ablation_metrics_nas.png`
- `02_ablation/ablation_metrics_sh.png`
- `tables/ablation_metrics_display.csv`

### Controller 是主要有效组件

NASDAQ 上，`Outer-only` 的累计收益为 `220.42%`，最大回撤为 `32.09%`，CR 为 `0.74`；加入 controller 后，`Outer + Controller` 的累计收益提高到 `237.50%`，最大回撤降至 `21.24%`，CR 提高到 `1.18`；完整 Ours 进一步达到 `265.53%`，最大回撤降至 `18.62%`，CR 提高到 `1.42`。这说明 controller 同时改善了收益、回撤和收益/回撤效率。

CSI-300市场上，controller 的贡献更明显：`Outer-only` 的累计收益为 `147.05%`、Sharpe ratio 为 `0.94`、CR 为 `0.99`；`Outer + Controller` 的累计收益提高到 `237.77%`、Sharpe ratio 提高到 `1.22`、CR 提高到 `1.16`。这说明 controller 能够在更复杂的市场中显著增强策略的动态适应能力。

### Dynamic controller 优于固定窗口切仓

固定窗口实验用于回答一个关键问题：模型收益是否只是来自更频繁或更固定的换仓。结果显示，固定 `5d/10d/20d/30d/60d` 切仓都不能稳定复现 Ours 的表现。

在 NASDAQ 上，所有固定窗口方法的累计收益均低于 Ours，且最大回撤普遍高于 Ours。在 CSI-300市场上，Ours 的累计收益也高于所有固定窗口 controller。由此可以说明：

> The advantage of the controller does not come from a fixed rebalancing frequency. Instead, the controller learns when to terminate the current holding state according to market and portfolio conditions.

### Inner actor 的合理解释

Inner actor 的结论需要谨慎。相较 `Outer-only`，`Outer + Inner` 在两个市场上都提高了累计收益：NASDAQ 从 `220.42%` 提高到 `227.43%`，CSI-300从 `147.05%` 提高到 `158.99%`。这说明 inner actor 对局部持仓修正有帮助。

但是，在加入 controller 后，inner actor 的收益贡献并非在所有市场上都单调增强。NASDAQ 中完整 Ours 优于 `Outer + Controller`；CSI-300中 `Outer + Controller` 的累计收益高于 Ours，而 Ours 的最大回撤略低。这说明 inner actor 不是主要风险控制来源，而是一个细粒度权重调整模块，其效果会受到市场结构和 controller 决策路径影响。

论文中建议写成：

> The inner actor provides local allocation refinement, while the controller contributes the dominant dynamic switching effect. The benefit of the inner actor is market-dependent and should be interpreted as adaptive weight adjustment rather than an independent source of universal excess return.

## 4. Controller 解释性结论

对应图：

- `03_controller_interpretability/controller_case_nas_01.png`
- `03_controller_interpretability/controller_case_nas_02.png`
- `03_controller_interpretability/controller_case_sh_01.png`
- `03_controller_interpretability/controller_case_sh_02.png`
- `03_controller_interpretability/fixed_window_comparison_nas.png`
- `03_controller_interpretability/fixed_window_comparison_sh.png`
- `03_controller_interpretability/switch_remaining_horizon_counterfactual_distribution_nas.png`
- `03_controller_interpretability/switch_remaining_horizon_counterfactual_distribution_sh.png`
- `03_controller_interpretability/controller_probability_resonance_nas.png`
- `03_controller_interpretability/controller_probability_resonance_sh.png`

### 4.1 Case study：switch 如何缓解下跌和回撤

Controller case 图展示了 30 日窗口内的实际财富曲线、所有 switch 点、关键 switch 的反事实收益路径，以及 controller exit probability。图中的关键比较是：在同一个 switch 时刻，从同一个起点出发，比较“继续旧组合”和“切到新组合”的未来 20 日表现。

四个代表性 case 的结果如下：

| Market | Key date | Hold return | Switch return | Return gain | Hold MDD | Switch MDD | MDD reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NASDAQ | 2021-04-19 | -6.05% | -2.63% | 3.42% | 8.82% | 5.51% | 3.31% |
| NASDAQ | 2020-07-13 | 3.79% | 8.29% | 4.50% | 5.34% | 2.53% | 2.81% |
| CSI-300 | 2021-07-07 | -10.74% | 4.31% | 15.05% | 13.59% | 8.23% | 5.36% |
| CSI-300 | 2022-03-15 | 2.38% | 10.53% | 8.15% | 8.60% | 3.81% | 4.79% |

这些 case 支持如下结论：

> The controller can identify holding states that are exposed to unfavorable short-term risk. In selected high-risk windows, switching to a new portfolio improves the counterfactual future return and reduces future drawdown compared with continuing the old portfolio.

这里的重点不是证明每个 switch 都能大幅赚钱，而是说明 controller 在关键风险窗口的行为具有经济含义：它能够在继续持有可能产生收益劣化或较大回撤时提前切换组合。

### 4.2 Fixed-window comparison：controller 不是固定周期切仓

固定窗口实验用 5/10/20/30/60 日持仓窗口替换 learned controller，并比较最终收益、Sharpe ratio、最大回撤和 CR。这个实验回答的问题是：controller 的收益是否只是来自某个固定再平衡周期。

NASDAQ 上，Ours 的累计收益为 `265.53%`、Sharpe ratio 为 `1.15`、最大回撤为 `18.62%`、CR 为 `1.42`。所有固定窗口的累计收益均低于 Ours，其中最接近的是 Fixed 30d 的 `227.43%`；所有固定窗口的最大回撤也都高于 Ours，CR 最高的 Fixed 5d 也只有 `0.95`。这说明 NASDAQ 上 learned controller 在收益、回撤和收益/回撤效率上都优于手工固定周期。

CSI-300市场上，Ours 的累计收益为 `204.99%`、Sharpe ratio 为 `1.14`、CR 为 `1.09`，均高于 5/10/20/30/60 日固定窗口。若只看最大回撤，Fixed 5d 和 Fixed 20d 略低于 Ours，但它们的累计收益分别只有 `103.69%` 和 `98.23%`，显著牺牲了收益。因此 CSI-300 上更合理的结论是：固定窗口可以在个别风险指标上略低，但不能同时复现 learned controller 的收益和风险调整表现。

可写入论文的结论：

> Compared with multiple fixed holding-window policies, the learned controller achieves a better risk-return trade-off across markets. This indicates that the controller does not merely select a favorable constant rebalancing period; instead, it learns state-dependent switching rules from the evolving holding state.

### 4.3 全量 switch remaining-horizon 反事实分布

用户之前指出的可比性问题是关键：一个持仓期内可能发生多次 switch，如果从某个 switch 点沿真实路径计算到后续持仓终点，结果会混入后续 controller 决策，不能说明当前这一次 switch 本身是否更好。

因此当前采用更干净的 decision-level counterfactual：

1. 对每个真实发生的 controller free switch，固定 switch 发生时刻作为共同起点。
2. 构造两条冻结路径：一条继续旧组合，一条切到新组合。
3. 两条路径都持有到同一个剩余持仓期限，即旧组合如果不切仓本来还能继续持有的期限。
4. 比较两条冻结路径的收益和最大回撤，避免后续 switch 污染当前 switch 的解释。

该实验覆盖了全部真实 free switch：NASDAQ 覆盖 `231/231`，CSI-300覆盖 `102/102`。统计结果如下：

| Market | Switches | Mean horizon | Switch return | Hold return | Mean gain | Positive gain ratio | Switch MDD | Hold MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NASDAQ | 231 | 28.60 days | 3.29% | 3.24% | 0.05% | 50.22% | 3.82% | 3.86% |
| CSI-300 | 102 | 25.59 days | 4.43% | 4.39% | 0.04% | 48.04% | 6.02% | 6.18% |

这张图的结论要保守表述。全量 switch 的平均收益优势为正，但幅度很小，正收益优势比例接近一半。这说明 controller 的全部 switch 中包含大量小幅再平衡和维护性切换，并不是每个 switch 都对应一个显著的下跌规避机会。

更合理的论文表述是：

> The all-switch counterfactual distribution shows that, under a clean remaining-horizon comparison, switching is not systematically worse than continuing the old portfolio and yields a slightly positive average gain. The magnitude of the average gain is modest because many switches are small maintenance decisions. Therefore, the controller's value should be understood together with the case studies and the fixed-window comparison: it is particularly useful in important downside windows and improves the realized investment path beyond manually fixed holding periods.

### 4.4 Switch probability 与收益风险的关系

`controller_probability_resonance_*.png` 展示 exit probability、实际 switch、以及未来 switch advantage 之间的关系。这里不能把 exit probability 简单理解成“未来收益越高，概率越高”的线性预测器。它更准确的含义是：controller 在当前状态下终止当前持仓的倾向。

全局统计中，exit probability 与 20 日未来收益优势的线性相关性并不强：NASDAQ 为 `-0.05`，CSI-300为 `0.01`。这不是坏结果，因为 controller 的决策同时受持仓约束、交易成本、当前组合状态、市场状态和隐藏状态影响。并且很多 switch 是维护性再平衡，不一定对应剧烈下跌。

论文中建议这样解释：

> Exit probability should be interpreted as a policy signal rather than a direct return predictor. The case studies show that exit probability rises around meaningful switching opportunities, while the global relation is nonlinear due to holding-period constraints, transaction costs, and heterogeneous market states.

## 5. Inner actor 解释性结论

对应图：

- `04_inner_actor_interpretability/inner_actor_alpha_nas.png`
- `04_inner_actor_interpretability/inner_actor_alpha_sh.png`
- `04_inner_actor_interpretability/inner_actor_weight_stack_nas.png`
- `04_inner_actor_interpretability/inner_actor_weight_stack_sh.png`
- `04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_nas.png`
- `04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_sh.png`
- `04_inner_actor_interpretability/inner_actor_weight_price_resonance_heatmap_nas.png`
- `04_inner_actor_interpretability/inner_actor_weight_price_resonance_heatmap_sh.png`
- `tables/inner_actor_summary_display.csv`

Inner actor 的解释性证据弱于 controller，这是符合模型结构的。Controller 是显式决定是否结束当前持仓的模块，因而可以用 switch、反事实收益和回撤来解释；inner actor 则是在给定持仓方向下进行局部权重修正，不直接对应一个清晰的“是否避险”动作。

从图上看，`inner_actor_weight_stack_*.png` 展示了 keep candidate 和 switch candidate 的 top-weight 分布差异，说明 inner actor 会改变组合内部的权重集中度和资产暴露方向。`inner_actor_alpha_*.png` 展示了 Ours 与 `Outer + Controller` 的相对收益、滚动收益和 turnover，用于观察局部权重修正是否与收益波动同步。

更适合作为论文主解释图的是 `inner_actor_base_adjustment_future_return_*.png`。这里不再看 raw weight，而是直接看 `inner tilt = executed weight - base weight`。正 tilt 表示 inner actor 相对 base 加大该资产，负 tilt 表示相对 base 降低该资产。图中将 inner tilt 与未来 5 日相对收益上下对齐，用来观察“相对加权是否对应未来相对更强、相对减权是否对应未来相对更弱”。

当前选择的 NASDAQ 窗口中，`corr(tilt, future relative return)` 为 `0.46`，正对齐天数为 `73.33%`；CSI-300窗口中，相关性为 `0.33`，正对齐天数为 `70.00%`。这些局部 case 说明 inner actor 在持仓期内会围绕 base 组合做小幅相对偏离，并且这些偏离与未来短期横截面收益强弱存在共振。`inner_actor_weight_price_resonance_heatmap_*.png` 可以作为补充图展示执行权重与价格相对强弱的关系，但主结论建议以 base-adjustment 图为准。

但从统计表看，inner alpha 的累计值在两个市场中分别为 `-1.10%` 和 `-2.01%`，没有形成稳定的独立正 alpha。因此不应把 inner actor 写成“单独产生显著超额收益”的模块。更稳妥的解释是：

> The inner actor improves local allocation flexibility and changes the composition of candidate portfolios, but its independent excess-return attribution is not as strong as the controller. It should be viewed as a local portfolio refinement mechanism that complements the controller, rather than the primary source of downside avoidance.

对于“有些表现不是最优”的问题，可以在论文中这样处理：

> Although the inner actor does not dominate in every market after adding the controller, it still plays a useful role in refining allocation decisions. The empirical results indicate that the controller is the main driver of dynamic risk control, while the inner actor provides market-dependent local adjustment. This division of roles is consistent with the hierarchical design of the model.

## 6. 建议论文图注

可以使用以下图注风格，避免出现 seed、checkpoint 等工程细节。

**Main experiment.**

> Performance comparison between Ours and matched baselines in NASDAQ and CSI-300 markets. The equity curves show cumulative wealth, while the bar plots compare total return, Sharpe ratio, and maximum drawdown.

**Ablation study.**

> Ablation study of the hierarchical policy components and fixed-window controllers. The results show that the learned controller contributes the dominant improvement and cannot be replaced by a fixed rebalancing schedule.

**Controller case study.**

> Representative controller switching cases. Each panel compares the realized portfolio trajectory, the counterfactual continuation of the old portfolio, the switched portfolio, and the controller exit probability around the switching window.

**All-switch counterfactual distribution.**

> Remaining-horizon counterfactual distribution over all controller free switches. For each switch, the old and new portfolios are frozen to the same remaining holding horizon, providing a decision-level comparison without contamination from later switches.

**Fixed-window controller comparison.**

> Comparison between the learned controller and fixed holding-window policies. The learned controller achieves better risk-adjusted performance, indicating that the switching rule is state-dependent rather than a manually chosen constant rebalancing period.

**Inner actor visualization.**

> Visualization of inner actor base-adjustment behavior. The figure compares the inner tilt, defined as executed weight minus base weight, with future short-horizon relative returns. Positive alignment indicates that the inner actor overweights future relative winners or underweights future relative losers within the holding period.

## 7. 推荐放入论文的完整结果分析段落

下面这段可以作为论文实验部分的结果分析初稿：

> The proposed method achieves a robust risk-return trade-off across both markets. In NASDAQ, Ours obtains a cumulative return of 265.53%, outperforming high-return traditional baselines such as WMAMR and Anticor, while reducing maximum drawdown from 33.88% and 44.59% to 18.62%. In the CSI-300 market, Ours achieves a cumulative return of 204.99%, slightly below DeepTrader, but improves Sharpe ratio from 1.03 to 1.14 and reduces maximum drawdown from 31.86% to 22.78%. These results indicate that the proposed hierarchical controller improves not only terminal wealth but also the quality and stability of returns.

> The ablation study confirms that the controller is the key component. Adding the controller improves both return and drawdown in NASDAQ and substantially increases return and Sharpe ratio in the CSI-300 market. Fixed-window controllers cannot reproduce the performance of the learned controller, showing that the advantage comes from state-dependent switching rather than a fixed rebalancing frequency. The inner actor provides local allocation refinement, but its contribution is market-dependent; therefore, it should be interpreted as a complementary weight-adjustment module rather than the main source of downside avoidance.

> The interpretability experiments further explain how the controller works. Representative cases show that, when continuing the current portfolio would lead to lower future return or larger drawdown, the controller switches to an alternative portfolio that improves the counterfactual 20-day return and reduces future drawdown. Compared with multiple fixed holding-window policies, the learned controller achieves better risk-adjusted performance across markets. For all switch decisions, the remaining-horizon counterfactual distribution compares the switched portfolio and the old portfolio under the same horizon without later-switch contamination. The average gain is modest but positive, indicating that many switches are small maintenance decisions while important downside windows contribute more visible improvements. Overall, these results support that the controller learns meaningful state-dependent switching behavior rather than a fixed-period rebalancing rule.

## 8. 写作时需要避免的表述

不建议写：

- “Ours 在所有市场、所有指标上都最优。”
- “每一次 switch 都显著提高收益。”
- “Exit probability 与未来收益优势强线性相关。”
- “Inner actor 单独产生稳定超额收益。”

建议改成：

- “Ours achieves a more favorable risk-adjusted trade-off.”
- “The controller is especially effective in important downside windows.”
- “Exit probability is a nonlinear policy signal aligned with switching opportunities.”
- “The inner actor provides local allocation refinement with market-dependent contribution.”
