# Inner 相对 Outer 的统计验证设计

## 1. 研究目标

本分析不检验或宣称 Inner Actor 能稳定产生显著的独立短期 alpha。目标是验证以下两条更窄、与实际实现一致的结论：

1. Inner 在 Outer/Controller 给定的 base portfolio 支撑集内产生非平凡、状态相关的配置调整，并改变组合的集中度、事前风险和交易成本。
2. Inner 作为闭环系统中的补充模块，可能通过改变实际持仓及其后续状态轨迹，改善 Full 系统相对于 Controller + Outer 的整体风险收益路径。

所有主指标与检验方法在查看新结果前固定，不根据显著性更换主指标、窗口或子样本。

## 2. 模型、数据和样本

- 代码：`/home/tongwenxuan/KD_abk/KD4RL_plus`
- NASDAQ-100：论文 selected seed 49 checkpoint
- CSI-300：论文 selected seed 90 checkpoint
- 测试样本：各模型完整测试期
- 输入：
  - action trace 中的 `base_weights_json`、`exec_weights_json`、`inner_tilt_json`
  - 原始复权价格
  - Full 与 No-Inner 闭环推理的日级 portfolio trace
- 分析单位：交易日；两个市场分别报告，不合并成一个总体样本。

当前仅有每个市场一个 selected checkpoint。因此时间序列置信区间只描述给定模型与测试区间的采样不确定性，不代表训练 seed 的不确定性。

## 3. 方案一：配置改善分析

### 3.1 机制一致性

对每个交易日验证：

\[
\delta_t=w_t^{exec}-b_t
\]

- `sum(exec_weight) = sum(base_weight) = 1`
- `sum(delta) = 0`
- Outer base 为零的资产，其 executed weight 也应为零
- trace 中保存的 `inner_tilt` 与 `exec-base` 一致

这些属于实现验证，不进行显著性宣称。

### 3.2 调整强度

主描述指标：

\[
ActiveShare_t=\frac{1}{2}\sum_i|\delta_{i,t}|
\]

同时报告：

- mean、median、IQR、P90
- `ActiveShare > 1%`、`> 5%`、`> 10%` 的交易日比例
- Inner 增量换手及估算交易成本

### 3.3 集中度变化

\[
HHI(w)=\sum_i w_i^2,\qquad N_{\mathrm{eff}}(w)=\frac{1}{HHI(w)}
\]

每天计算：

\[
\Delta HHI_t=HHI(w_t^{exec})-HHI(b_t)
\]

\[
\Delta N_{\mathrm{eff},t}
=N_{\mathrm{eff}}(w_t^{exec})-N_{\mathrm{eff}}(b_t)
\]

集中度降低不预设为必然改善，只解释为 Inner 对配置形态的影响。

### 3.4 事前风险变化

只使用决策日以前的收益估计协方差，避免未来信息泄漏。

主窗口为过去 60 个交易日；20 日窗口作为稳健性检查。协方差使用 Ledoit-Wolf shrinkage，若环境缺少相应依赖，则使用对角收缩样本协方差并明确记录。

\[
\sigma_t(w)=\sqrt{w^\top\Sigma_t w}
\]

主风险指标：

\[
\Delta\sigma_t
=\sigma_t(w_t^{exec})-\sigma_t(b_t)
\]

下行风险使用过去窗口中负收益构造半协方差：

\[
\Delta\sigma_t^{-}
=\sqrt{w_t^{exec\top}\Sigma_t^-w_t^{exec}}
-\sqrt{b_t^\top\Sigma_t^-b_t}
\]

报告均值、中位数、风险下降天数比例、Newey-West t-stat/p-value 和 moving-block bootstrap 95% CI。主结论不以 p 值单独决定，同时报告 bp/年化波动率单位的效应大小。

### 3.5 状态相关性

使用决策日前 20 日市场实现波动率将日期分成预先固定的低、中、高三个 tercile，比较 Active Share 和风险变化。报告各组效应与交互回归：

\[
\Delta\sigma_t
=\beta_0+\beta_1 HighVol_t+\beta_2 ActiveShare_t
+\beta_3 HighVol_t\times ActiveShare_t+\epsilon_t
\]

标准误使用 HAC。该分析用于判断 Inner 是否在高风险状态下进行更积极或更有效的配置调整，不根据结果重新切分阈值。

## 4. 方案二 A：Frozen-path 直接效应

### 4.1 反事实定义

在 Full trace 的同一个日期、同一个 Outer/Controller base 和同一段市场收益上比较：

- Inner：执行 `w_exec`
- Base：执行 `b`

两条路径分别按照自身从上一日持仓漂移到目标权重的换手率扣除交易成本。不得使用“执行组合承担成本、base 不承担成本”的不对称口径。

### 4.2 主结果

主直接效应为下一交易日公平净增量对数收益：

\[
\Delta r_{t+1}^{net}
=r_{t+1}^{exec,net}-r_{t+1}^{base,net}
\]

报告：

- mean net alpha，bp/day
- Newey-West t-stat、双侧 p-value
- 20 日 moving-block bootstrap 95% CI
- positive-alpha days
- annualized alpha Sharpe
- cumulative net alpha

该指标允许不显著。若 CI 覆盖零，结论写为“未检测到稳定独立 alpha”，不能改用事后筛选窗口。

### 4.3 风险与尾部补充

报告配对的：

- 事前波动率和半波动率差
- 日收益标准差差
- 5% expected shortfall 差
- 最差 5% 市场日中的平均配对收益差

“最差市场日”由等权市场收益定义，不按 Base 或 Full 自身的事后表现筛选，避免条件选择偏差。尾部指标是次要结果，采用 Benjamini-Hochberg FDR 校正。

### 4.4 Placebo

每天在 Outer 非零支撑集内随机置换真实 tilt，保持：

- tilt 横截面分布
- 权重和为 1
- Outer 支撑集不变

重复 5,000 次，比较真实累计净 alpha、事前风险变化和尾部损失变化在 placebo 分布中的百分位。若真实 tilt 不优于 placebo，应如实报告，不以其它随机种子替换。

## 5. 方案二 B：Closed-loop 总效应

### 5.1 场景

使用相同 checkpoint、测试期和确定性推理分别执行：

1. `Full`：Controller + Outer + Inner
2. `No-Inner`：Controller + Outer，令 `weights_exec = base_used`

No-Inner 允许实际持仓、Controller 状态和后续 switch 日期自然变化，因此测量的是系统总效应，而不是单步直接效应。

### 5.2 配对统计

按日期对齐两条日收益序列：

\[
d_t=r_t^{Full}-r_t^{NoInner}
\]

主闭环指标为：

- mean daily return difference
- cumulative return difference
- Sharpe difference
- MDD difference
- Calmar/CR difference
- turnover and total-cost difference
- 5% expected shortfall difference

统计推断：

- mean return difference：Newey-West 与 20 日 block bootstrap
- 累计收益、Sharpe、MDD、CR、ES：对配对日期块进行相同重采样，每次重新构造两条财富路径，报告差值的 95% CI
- bootstrap 重复 10,000 次，随机种子固定并写入输出 metadata

### 5.3 解释边界

- Frozen-path 直接效应描述 Inner 在固定 Outer base 下的局部作用。
- Closed-loop 差异描述 Inner 通过持仓状态和 Controller 反馈产生的系统总贡献。
- 两者之差只能称为“反馈/路径交互的迹象”，不能在没有正式中介识别假设时称为严格的 causal mediation effect。

## 6. 主次结果与多重比较

预先固定三个主结果：

1. `ActiveShare`：证明 Inner 非冗余地改变配置。
2. 60 日事前风险差 `DeltaExAnteVol`：检验配置风险是否改善。
3. Full − No-Inner 的配对闭环累计收益差及 MDD 差：检验系统互补贡献。

下一日净 alpha、20 日风险窗口、尾部指标、IC 和 placebo 属于次要或稳健性结果。两个市场分别解释；同一结果族内使用 Benjamini-Hochberg FDR。未来 5 日 IC 不作为主证据。

## 7. 预期输出

建议输出目录：

```text
reproduced_outputs/inner_outer_statistical_validation/
├── metadata/
│   └── run_manifest.json
├── traces/
│   ├── nas_full_daily.csv
│   ├── nas_no_inner_daily.csv
│   ├── sh_full_daily.csv
│   └── sh_no_inner_daily.csv
├── tables/
│   ├── configuration_refinement.csv
│   ├── frozen_path_direct_effect.csv
│   ├── closed_loop_effect.csv
│   ├── regime_analysis.csv
│   └── placebo_analysis.csv
├── figures/
│   ├── configuration_refinement_nas.pdf
│   ├── configuration_refinement_sh.pdf
│   ├── closed_loop_difference_nas.pdf
│   └── closed_loop_difference_sh.pdf
└── INNER_OUTER_STATISTICAL_VALIDATION.md
```

报告必须同时包含成功、零效应和负效应，不隐藏未支持结果。

## 8. 成功标准

分析成功不等于所有指标显著。完成标准是：

1. 两个 selected checkpoints 与数据身份被哈希记录。
2. Full/No-Inner 场景可确定性复现并按日期配对。
3. 配置、直接效应和闭环总效应三个层次均有完整结果。
4. 重叠时间序列使用 HAC 或 block bootstrap，而非普通独立样本 t 检验。
5. 结果能够支持下列之一：
   - Inner 产生非平凡配置调整，并在闭环中改善部分风险收益指标；或
   - Inner 虽改变配置，但未显示可靠的闭环改善。

不得以分析结果不理想为由更换主假设或筛选报告窗口。
