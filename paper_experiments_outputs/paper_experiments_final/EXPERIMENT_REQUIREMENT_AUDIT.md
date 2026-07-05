# 实验需求复核清单

本文档逐条复核当前输出是否覆盖用户要求。输出根目录为：

`paper_experiments_outputs/paper_experiments_final/`

## 1. 主实验

要求：

> 把 ours（HRL + controller）vs baseline 的收益折线图画出来，画最大回撤和 Sharpe 率等柱形图比较。

当前状态：已完成。

对应文件：
- `01_main_experiment/main_equity_nas.png`
- `01_main_experiment/main_equity_sh.png`
- `01_main_experiment/main_metrics_nas.png`
- `01_main_experiment/main_metrics_sh.png`
- `tables/main_experiment_metrics_display.csv`

说明：
- 收益折线图比较 Ours 和所有已对齐 baseline 曲线。
- 柱形图比较 Total return、Sharpe ratio、Max drawdown 和 CR。
- 数值展示统一为两位小数。

## 2. 消融实验

要求：

> 需要做 outer-only，outer-inner，outer + controller，HRL + 几个 fix 窗口 controller，ours，主要说明各个组件奏效。

当前状态：已完成。

对应文件：
- `02_ablation/ablation_equity_nas.png`
- `02_ablation/ablation_equity_sh.png`
- `02_ablation/ablation_metrics_nas.png`
- `02_ablation/ablation_metrics_sh.png`
- `tables/ablation_metrics_display.csv`

包含方法：
- `Outer-only`
- `Outer + Inner`
- `Outer + Controller`
- `Fixed 5d`
- `Fixed 10d`
- `Fixed 20d`
- `Fixed 30d`
- `Fixed 60d`
- `Ours`

说明：
- 固定窗口结果是 eval-only replay，不是训练新模型。
- 固定窗口用于证明 controller 不是简单固定频率调仓。

## 3. Controller case 解释性

要求：

> controller 抽取一些 case 说明确实缓解了下跌走势。

当前状态：已完成。

对应文件：
- `03_controller_interpretability/controller_case_nas_01.png`
- `03_controller_interpretability/controller_case_nas_02.png`
- `03_controller_interpretability/controller_case_sh_01.png`
- `03_controller_interpretability/controller_case_sh_02.png`
- `tables/controller_case_summary_display.csv`

说明：
- 每个 case 是 30 日窗口。
- 第一行标注窗口内所有 switch。
- 第二行比较关键 switch 的继续持有 vs 切仓反事实路径。
- 第三行展示 exit probability 与 switch advantage。

## 4. Controller 不是固定窗口切仓

要求：

> 说明 controller 切仓是有依据的，不是大量固定持仓窗口可以替代的，相比固定窗口表现更好。

当前状态：已完成。

对应文件：
- `03_controller_interpretability/fixed_window_comparison_nas.png`
- `03_controller_interpretability/fixed_window_comparison_sh.png`
- `tables/ablation_metrics_display.csv`

说明：
- 图中比较 Ours 和 Fixed 5d/10d/20d/30d/60d。
- 四个面板分别看 TR、Sharpe、MDD 和 CR。
- 这个实验用于说明真实 controller 的最终投资路径不是某个手工固定持仓周期可以复现的。

## 5. 所有 switch 的可比反事实收益分布

要求：

> 比较切仓点到持仓终点的收益分布和反事实收益分布，说明 switch 在统计结果上表现更优。特别是应该对所有 switch 和反事实产生的收益分布进行比较。

当前状态：已修正并补齐。

重要修正：
- 不能直接拿真实路径从某个 switch 点算到下一次切仓点或窗口终点来比较，因为真实路径中可能又发生多次 switch，这会混入后续 controller 决策。
- 因此当前采用更干净的 decision-level counterfactual：在每个真实 switch 时刻，把“继续旧组合”和“切到新组合”都冻结到同一个剩余持仓期限，比较两条反事实曲线。
- 这个剩余持仓期限定义为 `max_hold - duration_before_decision`，即切仓前旧组合如果不切仓、本来还能继续持有到的期限。

对应文件：
- `03_controller_interpretability/switch_remaining_horizon_counterfactual_distribution_nas.png`
- `03_controller_interpretability/switch_remaining_horizon_counterfactual_distribution_sh.png`
- `03_controller_interpretability/switch_remaining_horizon_distribution_all.csv`
- `03_controller_interpretability/switch_remaining_horizon_distribution_all_display.csv`
- `03_controller_interpretability/switch_remaining_horizon_summary_display.csv`
- `tables/switch_remaining_horizon_distribution_all_display.csv`
- `tables/switch_remaining_horizon_summary_display.csv`

实现方式：
- 对每个真实发生的 controller free switch，读取 `duration_before_decision`。
- 计算旧组合原本的剩余持仓期限：`remaining_holding_days = max_hold - duration_before_decision`。
- 使用 30 天反事实曲线，在同一个 `remaining_holding_days` 上比较：
  - `counterfactual_hold_return_to_original_end`：不切仓，继续旧组合。
  - `switch_return_to_original_end`：切到新组合。
- 比较 `switch_minus_counterfactual_hold`。

覆盖率：
- NASDAQ：`231/231` 个 free switch，覆盖率 `100.00%`。
- CSI-300：`102/102` 个 free switch，覆盖率 `100.00%`。

当前统计结果：
- NASDAQ：switch 冻结收益均值 `3.29%`，继续旧组合反事实均值 `3.24%`，平均优势 `0.05%`。
- CSI-300：switch 冻结收益均值 `4.43%`，继续旧组合反事实均值 `4.39%`，平均优势 `0.04%`。

解释注意：
- 这张图现在满足“所有 switch 与反事实分布比较”的要求，而且避免了后续多次 switch 污染单次 switch 的比较。
- 但全量平均优势较温和，所以论文里不建议写“每次 switch 都显著更优”。
- 更合理的写法是：全量分布显示 switch 的平均优势为正但不大；结合 case 图和固定持仓窗口对比，可以说明 controller 在关键下跌窗口更有价值，并且最终投资路径优于固定周期切仓。

## 6. Switch probability 与收益表现共振

要求：

> 说明切仓时刻的概率是能够跟收益表现共振的。

当前状态：已完成。

对应文件：
- `03_controller_interpretability/controller_probability_resonance_nas.png`
- `03_controller_interpretability/controller_probability_resonance_sh.png`
- `tables/controller_statistical_summary_display.csv`

说明：
- 图中展示 rolling exit probability 与 future switch advantage 的变化。
- binned panel 展示不同 exit probability 区间下的平均 switch advantage 和 switch rate。
- 需要谨慎表述为局部共振或非线性关联，不建议写成强线性相关。

## 7. Inner actor 可解释性

要求：

> inner actor 可解释就是可视化一下 inner 和 actor 的权重分布堆叠图，然后说明赚取相对收益波动，持仓变化方向和价格走向共振。

当前状态：已完成，但证据强度需要谨慎表述。

对应文件：
- `04_inner_actor_interpretability/inner_actor_alpha_nas.png`
- `04_inner_actor_interpretability/inner_actor_alpha_sh.png`
- `04_inner_actor_interpretability/inner_actor_weight_stack_nas.png`
- `04_inner_actor_interpretability/inner_actor_weight_stack_sh.png`
- `tables/inner_actor_summary_display.csv`

说明：
- `inner_actor_alpha_*.png` 展示 Ours vs Outer + Controller、累计 inner alpha、rolling return 和 turnover。
- `inner_actor_weight_stack_*.png` 展示关键 switch 时 keep candidate 与 switch candidate 的 top-weight 分布。
- Inner actor 的证据比 controller 弱，它应被解释为局部持仓修正模块，而不是主要切仓模块。

## 8. 解读说明

要求：

> 写 MD 文件说明这些图怎么看，说明了什么，怎么证明是好的。

当前状态：已完成。

对应文件：
- `FIGURE_INTERPRETATION.md`
- `01_main_experiment/README.md`
- `02_ablation/README.md`
- `03_controller_interpretability/README.md`
- `04_inner_actor_interpretability/README.md`

最重要的说明文件：
- `FIGURE_INTERPRETATION.md`

该文件包含：
- 主实验怎么解释非最优结果。
- controller switch probability 如何和收益/风险联系。
- remaining-horizon 全量 switch 分布怎么读。
- inner actor 为什么要谨慎解释。
- 可直接写进论文的结论文字。

## 9. 目前仍需谨慎的点

1. CSI-300中 `Outer + Controller` 的收益高于 Ours，所以不能写 inner actor 在所有市场、所有条件下都提升收益。
2. Endpoint 全量 switch 分布的平均优势为正但很小，所以不能写“所有 switch 显著优于继续持有”。
3. Switch probability 与收益优势不是强线性关系，更适合写成“局部共振”或“状态依赖的非线性关联”。
4. Controller 的主要证据应来自消融、case、固定持仓窗口对比、remaining-horizon counterfactual distribution 四者组合，而不是单独依赖某一张图。
