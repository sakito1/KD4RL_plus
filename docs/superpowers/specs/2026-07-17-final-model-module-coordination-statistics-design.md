# Final-Model Module Coordination Statistics Design

日期：2026-07-17

## 目标

基于论文当前最终模型已经保存的逐日 action 与 portfolio trace，生成一套中文、
可复现的模块解释性统计。本分析只研究当前选定模型，不重新训练，不汇总多个训练
随机种子，也不修改论文 LaTeX。

分析回答三个问题：

1. Controller 在测试期如何使用自由切换与强制切换，其选择与已保存的反事实
   Hold/Switch 结果呈现什么关系？
2. Controller 接受 Manager 候选后，底仓支持集发生了多大变化？
3. Trader 的支持集内修正强度是否随底仓持有年龄和近期市场波动系统性变化？

## 数据范围

输入固定为当前论文对应的最终模型 trace：

- Nasdaq-100：`paper_experiments_outputs/end_to_end_explain/traces/nas_seed49_full_controller_actions.csv`
- Nasdaq-100 portfolio：`paper_experiments_outputs/end_to_end_explain/traces/nas_seed49_full_controller_portfolio.csv`
- CSI-300：`paper_experiments_outputs/end_to_end_explain/traces/sh_seed90_full_controller_actions.csv`
- CSI-300 portfolio：`paper_experiments_outputs/end_to_end_explain/traces/sh_seed90_full_controller_portfolio.csv`

所有统计覆盖各 trace 中的完整测试期。市场标签在输出中统一显示为
`Nasdaq-100` 和 `CSI-300`。

## 分析一：Controller 决策行为

### 决策构成

分别统计：

- 总交易日数；
- free decisions 数；
- free switches 数及其占 free decisions 的比例；
- forced switches 数；
- Hold 与 free Switch 的 `exit_prob`、`duration_before_decision`、
  `controller_switch_advantage` 的均值和中位数。

### 已选择动作的反事实优势

对具有20日冻结反事实结果的 free decisions，定义：

```text
switch_advantage_20 = switch_future_return_20 - hold_future_return_20
chosen_action_advantage_20 =
    switch_advantage_20,  if the Controller selected Switch
   -switch_advantage_20,  if the Controller selected Hold
```

正值表示当前 Controller 实际选择的动作优于未选择动作。输出每个市场的样本数、
均值、中位数、正值比例，以及按20个交易日 circular moving-block bootstrap 得到的
均值95%置信区间。该统计只解释当前最终策略在当前测试期的行为，不解释训练随机性。

### Switch probability 分箱

在每个市场内部，按照 `exit_prob` 的五分位数对 free decisions 分箱。每箱报告：

- 样本数；
- 平均 switch probability；
- 实际 free-switch 比例；
- 平均 `switch_advantage_20`；
- switch advantage 为正的比例。

该结果用于检查概率升高是否伴随更强的候选相对优势。报告必须保留可能不单调或
相关较弱的结果，不得选择性删除分箱。

## 分析二：Manager--Controller 底仓转换

仅使用 learned free switches，排除 forced switches。对于每个 free switch，比较
前一交易日 active-base support 与切换日新 active-base support，统计：

- 保留资产数；
- 新增资产数；
- 移除资产数；
- support Jaccard overlap；
- 相邻 active-base weight overlap；
- 相邻 active-base L1 distance。

Support 统计不受价格漂移改变，因为价格漂移不会改变非零支持集。Weight overlap
和 L1 distance 比较的是 trace 中相邻两个交易日的 active bases；它们包含隔夜
价格漂移影响，因此只作为描述性结构变化量，不称为纯粹的 Controller 交易成本或
精确候选距离。

每个市场输出事件级 CSV，以及均值、中位数、四分位数汇总。

## 分析三：Trader 跨时间尺度修正

### 修正强度

从 `inner_tilt_json` 计算：

```text
refinement_l1 = sum_i abs(inner_tilt_i)
```

它等于 Trader 相对 active base 的支持集内总绝对权重修正。

### 持有年龄统计

以 portfolio trace 的 `holding_duration` 为切换后的当前底仓年龄，分组为：

- 0--1；
- 2--5；
- 6--10；
- 11--20；
- 21+ 个交易日。

每组报告样本数、修正强度均值、中位数、25%与75%分位数。另计算持有年龄与
修正强度的 Spearman 相关系数。

### 近期波动统计

近期波动定义为决策日前10个交易日 `base_log_return` 的样本标准差：

```text
recent_volatility_10 = std(base_log_return[t-10:t], ddof=1)
```

实现时必须先 `shift(1)`，确保当天回报不进入当天决策前的波动统计。每个市场
独立按照有效波动值划分四分位数组，报告各组修正强度的样本数、均值、中位数和
四分位数。另计算近期波动与修正强度的 Spearman 相关系数。

### 时间依赖不确定性

对以下统计使用20日 circular moving-block bootstrap、5,000次重复和固定随机种子
20260717：

- holding age 与 refinement L1 的 Spearman 相关；
- recent volatility 与 refinement L1 的 Spearman 相关；
- 高波动四分位与低波动四分位的平均 refinement L1 差值。

输出百分位法95%置信区间。Bootstrap 反映当前测试路径的时间序列不确定性，不能
写成跨训练 seed 稳定性。

## 输出

输出目录：

`paper_experiments_outputs/final_model_module_coordination/`

生成：

- `daily_module_events.csv`
- `controller_decision_summary.csv`
- `controller_probability_bins.csv`
- `base_transition_events.csv`
- `base_transition_summary.csv`
- `trader_holding_age_summary.csv`
- `trader_volatility_summary.csv`
- `trader_correlation_summary.csv`
- `最终模型模块协同统计.md`

Markdown 报告使用中文，包含数据范围、定义、结果表、论文可用解释、限制和建议
措辞。CSV 字段使用稳定的英文 snake_case，便于后续自动生成 LaTeX 表格。

## 实现边界

- 不修改或重新运行训练。
- 不修改 checkpoint。
- 不修改论文 LaTeX。
- 不声称因果特征重要性。
- 不把 bootstrap 重采样称为多随机种子实验。
- 不进行当前 trace 未保存的 Controller 输入组遮蔽或内部特征归因。
- 不隐藏方向较弱或不显著的 Controller 统计。

## 验证

测试覆盖：

- JSON 权重与 tilt 解析；
- refinement L1 计算；
- 切换后 holding age 的合并与分箱边界；
- 波动计算严格使用 `shift(1)`；
- Hold/Switch chosen-action advantage 的符号方向；
- support replacement 与 Jaccard 计算；
- moving-block bootstrap 在固定种子下可重复；
- 两个市场事件数与原 trace 一致；
- 所有汇总值有限，允许原始缺失反事实行被显式排除；
- 重复运行生成一致的 CSV 与 Markdown。
