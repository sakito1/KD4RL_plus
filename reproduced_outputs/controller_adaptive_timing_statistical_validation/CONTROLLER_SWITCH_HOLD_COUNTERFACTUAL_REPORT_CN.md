# Controller 切换与不切换反事实统计报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-28
- Verification Status：VERIFIED
- Version Label：controller_switch_hold_validation_v2
- 数据范围：NASDAQ-100（seed 49）与 CSI-300（seed 90）完整测试集
- 分析对象：Controller 的全部自由决策，不包含 forced switch、forced hold 和 fixed-day 对比

| 市场 | 输入轨迹行数 | 有效自由决策 | Switch | Hold | Switch率 |
|---|---:|---:|---:|---:|---:|
| NASDAQ-100 | 1,369 | 1,334 | 231 | 1,103 | 17.32% |
| CSI-300 | 1,247 | 1,220 | 92 | 1,128 | 7.54% |
| 合计 | 2,616 | 2,554 | 323 | 2,231 | 12.65% |

## 1. 分析目的

本报告检验以下问题：

> 在同一个 Controller 决策日，实际选择的 Switch 或 Hold 是否优于当天未选择的反事实动作？

这里的“切换收益”和“不切换收益”不是比较不同日期的真实收益，也不是比较两次独立训练结果，而是在每一个自由决策日构造两个具有相同起点、相同未来行情和相同 Inner-Actor 处理方式的冻结投资组合：

1. **Hold 路径**：维持当前 Outer 配置，经 Inner-Actor 调整后得到 `hold_exec`；
2. **Switch 路径**：采用当天 Outer 候选配置，经相同 Inner-Actor 调整后得到 `switch_exec`。

这种同日反事实比较控制了市场日期和未来行情差异，用于隔离当天 Controller 切换/不切换决策的相对结果。

## 2. 数据来源

Controller 决策轨迹：

```text
paper_experiments_outputs/paper_experiments_final/
_cache/counterfactual_horizon30/
nas_seed49_full_controller_horizon30_actions.csv
sh_seed90_full_controller_horizon30_actions.csv
```

主要字段包括：

| 字段 | 含义 |
|---|---|
| `decision_type` | 是否为 Controller 自由决策 |
| `is_switch` | 实际动作：1=Switch，0=Hold |
| `duration_before_decision` | 决策前已经持有的天数 |
| `hold_curve_30` | 当天选择 Hold 后的冻结反事实净值曲线 |
| `switch_curve_30` | 当天选择 Switch 后的冻结反事实净值曲线 |
| `exit_prob` | Controller 输出的退出/切换概率 |

两个市场合计输入2,616行，其中有效自由决策为2,554行；NASDAQ-100和CSI-300均无无效反事实曲线。两个市场分别统计，不计算合并市场收益均值。

## 3. 反事实路径如何生成

### 3.1 相同未来行情

在决策日 \(t\)，Hold 和 Switch 使用完全相同的未来资产收益矩阵：

\[
\mathbf{G}_{t:t+H_t}
\]

因此，两条路径的差异只来自当天采用的组合权重，而不是未来市场环境不同。

### 3.2 冻结权重

对给定初始权重 \(\mathbf{w}\)，未来冻结组合净值为：

\[
W_{t,h}(\mathbf{w})
=
\sum_i w_i
\prod_{k=1}^{h}(1+r_{i,t+k}),
\qquad h=1,\ldots,H_t.
\]

生成曲线以后，不允许 Controller 或 Outer 在窗口中再次修改权重。这样可以避免后续多次切换污染当前决策的比较。

### 3.3 交易成本

两条路径均根据当天目标权重相对当前漂移权重的换手率扣除一次调仓成本：

\[
C_t(\mathbf{w})
=
\lambda
\sum_i |w_i-w^{current}_{i,t}|.
\]

因此，Switch 路径通常会承担更高的初始调仓成本，但 Hold 路径如果存在 Inner 调整，也按实际换手扣除成本。

### 3.4 自适应比较窗口

Controller 的最大持有上限为30天。对于决策前已经持有 \(d_t\) 天的样本，比较窗口定义为：

\[
H_t=\max(1,30-d_t).
\]

例如：

| 已持有时间 \(d_t\) | 比较窗口 \(H_t\) |
|---:|---:|
| 1天 | 29天 |
| 10天 | 20天 |
| 20天 | 10天 |
| 29天 | 1天 |

该窗口表示：如果当天不切换，原配置距离30天强制切换上限还剩多少天。

## 4. 收益指标

### 4.1 Switch 相对 Hold 的收益优势

记窗口终点的冻结净值分别为：

\[
W^{switch}_{t,H_t},
\qquad
W^{hold}_{t,H_t}.
\]

当天 Switch 相对于 Hold 的日均对数收益优势定义为：

\[
A_t^{return}
=
\frac{
\log W^{switch}_{t,H_t}
-
\log W^{hold}_{t,H_t}
}{H_t}
\times 10,000.
\]

单位为 bp/日。

- \(A_t^{return}>0\)：Switch 优于 Hold；
- \(A_t^{return}<0\)：Hold 优于 Switch。

### 4.2 实际决策价值

为了统一评价实际动作是否正确，定义：

\[
V_t^{return}
=
(2a_t-1)A_t^{return},
\]

其中 \(a_t=1\) 表示实际 Switch，\(a_t=0\) 表示实际 Hold。

因此：

| 实际动作 | 决策价值 |
|---|---|
| Switch | \(V_t=A_t\) |
| Hold | \(V_t=-A_t\) |

无论实际动作是什么，\(V_t>0\) 都表示实际动作优于同日未选择的动作。

### 4.3 计算示例

假设当前已经持有10天，因此 \(H_t=20\)。未来20天终点：

```text
Hold 冻结净值   = 1.020
Switch 冻结净值 = 1.010
```

则：

\[
A_t^{return}
=
\frac{\log(1.010)-\log(1.020)}{20}
\times10,000
\approx-4.93\text{ bp/日}.
\]

这表示当天 Switch 相比 Hold 平均每天少4.93 bp。如果 Controller 实际选择 Hold，则此次决策价值为：

\[
V_t^{return}=+4.93\text{ bp/日}.
\]

## 5. 最大回撤指标

对冻结净值曲线 \(W_{t,h}\)，最大回撤定义为：

\[
\operatorname{MDD}(W)
=
\max_h
\frac{
\max_{u\le h}W_{t,u}-W_{t,h}
}{
\max_{u\le h}W_{t,u}
}.
\]

Switch 相对 Hold 的回撤优势为：

\[
A_t^{MDD}
=
\operatorname{MDD}(W^{hold})
-
\operatorname{MDD}(W^{switch}).
\]

- \(A_t^{MDD}>0\)：Switch 的最大回撤更小；
- \(A_t^{MDD}<0\)：Hold 的最大回撤更小。

实际动作的回撤决策价值同样定义为：

\[
V_t^{MDD}=(2a_t-1)A_t^{MDD}.
\]

因此 \(V_t^{MDD}>0\) 统一表示实际动作降低了最大回撤。

## 6. 统计推断方法

### 6.1 点估计

Switch 和 Hold 分组的点估计均为组内所有自由决策的等权平均：

\[
\bar A_g
=
\frac{1}{N_g}
\sum_{t\in g}A_t,
\qquad
g\in\{\text{Switch},\text{Hold}\}.
\]

各市场样本构成为：

| 市场 | Switch组 | Hold组 |
|---|---:|---:|
| NASDAQ-100 | 231个决策 | 1,103个决策 |
| CSI-300 | 92个决策 | 1,128个决策 |

### 6.2 置信区间

因为相邻决策的未来窗口重叠，样本不能视为相互独立。报告使用 circular block bootstrap：

- 区块长度：30个连续决策日；
- 重复次数：10,000次；
- 每次抽取连续区块并拼接到原样本长度；
- 使用 bootstrap 分布的2.5%和97.5%分位数构造95%置信区间。

### 6.3 均值显著性

整体决策价值使用 Newey–West HAC 标准误：

- 最大滞后阶数：5；
- 原假设：平均决策价值为0；
- NASDAQ-100 与 CSI-300 同类检验使用 Benjamini–Hochberg 方法修正多重比较。

### 6.4 匹配随机动作置换

为了检验结果是否仅来自“某些持有期或波动状态本来就更有利”，进一步将样本按照以下状态分层：

1. 决策前持有时间三分位；
2. 过去20日市场波动率三分位。

在每个分层内部随机打乱 Switch/Hold 动作，同时严格保持该层的 Switch 数量不变。共重复5,000次。

真实决策价值与置换分布比较得到单侧置换 p 值。全部5,000次置换均未违反层内 Switch 数量约束。

### 6.5 两个市场的成分股一致性

为了检验“NASDAQ-100成分股更加同步，因此切换配置的边际作用较小”这一解释，本报告额外比较两个实验股票池在共同测试日期内的横截面特征。

- 共同日期：2020-04-23至2025-02-27；
- 同时存在于两个市场收益面板的交易日：1,131天；
- NASDAQ-100实验股票池：39只；
- CSI-300实验股票池：53只；
- 收益定义：复权收盘价的日对数收益；
- 每个市场仅使用模型实际交易的股票；
- 日度同向比例和横截面离散度之差使用30日配对circular block bootstrap，重复10,000次。

一致性指标定义如下：

1. **平均成分股相关性**：股票两两日收益相关系数的平均值；
2. **第一主成分解释率**：标准化成分股收益协方差矩阵中最大特征值占全部特征值之和的比例；
3. **日度同向比例**：每天与等权市场收益符号相同的股票比例；
4. **横截面离散度**：每天所有成分股日收益的横截面标准差。

## 7. 统计结果

### 7.1 Switch/Hold 收益分解

| 市场 | 实际动作 | 决策数 | 平均 Switch−Hold | 95%区块置信区间 | 有利动作比例 |
|---|---|---:|---:|---:|---:|
| NASDAQ-100 | Switch | 231 | +0.232 bp/日 | [−0.213, 0.687] | 50.22% |
| NASDAQ-100 | Hold | 1,103 | −0.241 bp/日 | [−3.786, 3.291] | 50.86% |
| CSI-300 | Switch | 92 | +0.460 bp/日 | [−2.128, 2.530] | 54.35% |
| CSI-300 | Hold | 1,128 | **−6.269 bp/日** | **[−11.293, −1.311]** | 58.42% |

解释：

- NASDAQ-100 的 Switch 和 Hold 两组置信区间均包含0，没有检出稳定的动作收益差异；
- CSI-300 实际 Switch 日的平均优势为正，但置信区间包含0，不能认为 Switch 本身具有稳定收益优势；
- CSI-300 实际 Hold 日的 `Switch−Hold` 显著小于0，说明这些日期继续持有优于强行切换；
- 将方向换成实际动作价值后，CSI-300 Hold组相当于平均避免了6.269 bp/日的不利切换损失。

### 7.2 全部实际动作的决策价值

| 市场 | 指标 | 点估计 | 95%区块置信区间 | Newey–West t | 调整后 p |
|---|---|---:|---:|---:|---:|
| NASDAQ-100 | 收益决策价值 | +0.240 bp/日 | [−2.688, 3.298] | 0.207 | 0.8357 |
| NASDAQ-100 | MDD 决策价值 | −0.030 pp | [−0.228, 0.164] | −0.406 | 0.6846 |
| CSI-300 | 收益决策价值 | **+5.831 bp/日** | **[1.213, 10.430]** | 3.481 | 0.0010 |
| CSI-300 | MDD 决策价值 | **+0.371 pp** | **[0.158, 0.595]** | 4.386 | <0.0001 |

另外：

| 市场 | 收益价值为正 | MDD价值为正 | 平均自适应窗口 | 中位自适应窗口 |
|---|---:|---:|---:|---:|
| NASDAQ-100 | 50.75% | 47.60% | 17.90天 | 19天 |
| CSI-300 | 58.11% | 57.38% | 18.02天 | 19天 |

### 7.3 匹配随机动作置换

| 市场 | 指标 | 真实动作 | 匹配置换均值 | 调整后 p | 判断 |
|---|---|---:|---:|---:|---|
| NASDAQ-100 | 收益决策价值 | 0.240 bp/日 | 0.536 bp/日 | 0.962 | 不显著 |
| NASDAQ-100 | MDD 决策价值 | −0.030 pp | −0.023 pp | 0.647 | 不显著 |
| CSI-300 | 收益决策价值 | 5.831 bp/日 | 5.260 bp/日 | 0.113 | 不显著 |
| CSI-300 | MDD 决策价值 | 0.371 pp | 0.306 pp | 0.0316 | 显著 |

解释：

- NASDAQ-100 的收益和MDD真实值均未优于匹配置换分布；
- CSI-300 收益决策价值总体显著为正，但控制持有时间、波动状态和各层Switch数量后，真实动作相对于匹配随机动作的额外收益优势没有达到5%显著性；
- CSI-300 MDD的匹配置换结果仍显著，说明实际时机的回撤改善不能完全由上述状态和Switch数量解释。

### 7.4 市场成分股一致性

| 指标 | NASDAQ-100 | CSI-300 | NASDAQ−CSI |
|---|---:|---:|---:|
| 平均成分股相关性 | 0.359 | 0.242 | +0.117 |
| 第一主成分解释率 | 39.39% | 26.42% | +12.97 pp |
| 平均日度同向比例 | 72.19% | 66.84% | +5.35 pp |
| 平均横截面收益离散度 | 1.487% | 1.986% | −0.499 pp |

日度配对区块自助法结果：

| 比较量 | 均值差 | 95%区块置信区间 |
|---|---:|---:|
| NASDAQ−CSI同向比例 | +5.35 pp | [3.65, 7.01] pp |
| NASDAQ−CSI横截面离散度 | −0.499 pp | [−0.666, −0.346] pp |

四个指标方向一致：NASDAQ-100实验股票池的共同波动成分更强，而CSI-300实验股票池的个股收益分化更明显。这为两个市场Controller结果不同提供了数据层面的机制解释。

## 8. 结果意味着什么

### 8.1 统计证据直接支持的内容

1. CSI-300测试期内，实际Controller动作相对于同日未选择动作具有正的平均收益决策价值；
2. CSI-300实际动作具有正的平均最大回撤决策价值；
3. CSI-300主要的收益贡献来自Hold决策，即避免在不利日期切换；
4. 控制持有时间、市场波动状态和Switch数量后，CSI-300回撤改善仍超过匹配随机动作；
5. NASDAQ-100的各项主要检验均未拒绝零效应，只能报告为“未检出统计证据”；
6. 共同测试日期内，NASDAQ-100实验股票池比CSI-300股票池表现出更高的成分股相关性、同向比例和共同因子解释率，以及更低的横截面收益离散度。

### 8.2 市场特性如何解释结果差异

当天Switch和Hold的组合收益差可以写为：

\[
\Delta R_t
=
(\mathbf w_t^{switch}-\mathbf w_t^{hold})^\top
\mathbf r_{t+1}.
\]

由于两组组合权重和都为1，因此权重差之和为0。若个股收益主要由共同市场成分构成：

\[
\mathbf r_{t+1}
=
r^{market}_{t+1}\mathbf 1
+
\boldsymbol\epsilon_{t+1},
\]

则共同市场项会在权重差中抵消：

\[
(\mathbf w^{switch}-\mathbf w^{hold})^\top
r^{market}\mathbf 1
=0.
\]

Switch和Hold的差异主要取决于个股相对市场的分化项：

\[
\Delta R_t
=
(\mathbf w^{switch}-\mathbf w^{hold})^\top
\boldsymbol\epsilon_{t+1}.
\]

因此：

- 当成分股相关性高、同向比例高、横截面离散度低时，不同配置的未来收益更接近，Controller即使改变配置，也较难产生较大的可识别增量；
- 当个股分化更明显时，不同配置之间的收益和回撤差异更大，Controller是否允许切换更容易影响组合结果。

这与当前数据一致：

1. NASDAQ-100平均成分股相关性为0.359，横截面离散度为1.487%；其Switch和Hold收益差均接近0，整体决策价值不显著；
2. CSI-300平均成分股相关性为0.242，横截面离散度为1.986%；其Hold决策能够过滤明显不利的切换，并表现出显著的MDD改善。

因此，可以将NASDAQ-100结果不显著解释为：该实验股票池在测试期内具有更强的共同波动和更低的横截面分化，使Outer候选配置与当前配置之间的可实现收益差较小，压缩了Controller能够创造或识别的边际价值。

### 8.3 统计证据不能支持的内容

1. 不能声称实际 Switch 日具有显著的正收益优势；
2. 不能声称精确的收益切换时机显著优于匹配随机动作；
3. 不能声称Controller在NASDAQ-100上具有显著决策价值；
4. 不能将单一 seed、单一 checkpoint 的时间序列置信区间解释为跨随机种子的训练稳定性；
5. 不能把冻结反事实比较解释为重新运行两条动态策略后的长期因果效应；
6. 市场一致性与Controller显著性之间是跨市场机制一致性证据，不是随机对照因果证明；模型拟合、样本区间、股票池大小和市场制度差异也可能影响结果。

## 9. 最简结论

> 两个市场呈现不同结果。在NASDAQ-100完整测试集中，Switch、Hold、整体收益决策价值和MDD决策价值的置信区间均包含0，匹配置换检验也不显著，因此当前数据未提供Controller有效性的统计证据。市场横截面统计显示，NASDAQ-100实验股票池具有更高的平均成分股相关性和同向比例、更高的共同因子解释率以及更低的收益离散度；不同配置的未来收益因而更容易接近，这可以解释Controller边际作用为何较难显现。在CSI-300完整测试集中，个股分化更明显，Controller主要通过避免不利切换发挥作用：实际Switch日的收益优势不显著，但在实际Hold日，强制切换平均会损失6.269 bp/日，其95%区块置信区间完全低于0。CSI-300全部实际动作的平均收益和MDD决策价值均显著为正；进一步控制持有期、市场波动状态及切换数量后，收益时机的增量证据不显著，而MDD改善仍达到统计显著。该市场特性解释与数据一致，但仍属于机制假设而非因果证明。

## 10. 统计风险与谬误检查

- 检查覆盖：11/11

| 检查项 | 状态 | 本分析中的处理 |
|---|---|---|
| Simpson's paradox | NOTE | NASDAQ-100 与 CSI-300 分市场报告，不使用合并市场均值 |
| Ecological fallacy | NOTE | 推断单位始终为市场内决策事件，不推断个股层面效果 |
| Berkson's paradox | CAUTION | 仅分析选定 checkpoint 的测试轨迹，存在模型选择条件 |
| Collider bias | CAUTION | 状态匹配置换仅作为稳健性检验，不解释为因果控制 |
| Base-rate neglect | CAUTION | 明确报告NASDAQ-100的17.32%和CSI-300的7.54% Switch基准率，不单独使用AUROC证明有效 |
| Regression to the mean | NOTE | 未按极端未来收益选择样本；但状态条件分析不作因果解释 |
| Survivorship bias | NOTE | 完整纳入两个市场共2,554个自由决策，无事后删除失败决策 |
| Look-elsewhere effect | CAUTION | 报告不显著结果，并对主要同类检验进行BH修正 |
| Garden of forking paths | CAUTION | 分析属于事后解释性验证，并非预注册确认性检验 |
| Correlation ≠ causation | CAUTION | 使用“反事实冻结比较”和“统计支持”，不声称真实动态因果效应 |
| Reverse causality | NOTE | 收益结局发生在动作之后，但模型状态关系仍只作描述性解释 |

市场一致性补充分析未按照结果选择日期，而是使用两个市场共同覆盖的全部1,131个交易日；但该比较仍只有两个市场，不能据此估计一般性的“市场一致性—Controller效果”回归关系。

## 11. 可复现文件

分析代码：

```text
paper_experiments/analyze_controller_adaptive_timing.py
paper_experiments/eval_end_to_end_explain.py
```

结果表：

```text
tables/adaptive_horizon_decision_value.csv
tables/switch_hold_decomposition.csv
tables/matched_action_permutation.csv
tables/decision_audit.csv
```

市场一致性补充分析使用：

```text
DeepAries/data/nas/nas_data.csv
DeepAries/data/sh/sh_data.csv
```

运行参数：

```text
max_horizon=30
block_length=30
bootstrap_reps=10000
placebo_reps=5000
random_seed=20260727
markets=nas sh
checkpoint_seeds=nas:49 sh:90
```

该结果已经重新运行并通过以下审计：

```text
NASDAQ-100有效自由决策：1,334
NASDAQ-100自由Switch：231
CSI-300有效自由决策：1,220
CSI-300自由Switch：92  
两个市场无效曲线：0
两个市场置换层内Switch数量违规：0
```
