# Controller Base 状态解释性验证

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-29
- Verification Status：VERIFIED
- Markets：NASDAQ-100 seed 49；CSI-300 seed 90
- Sample：完整测试集自由决策日，NASDAQ-100 N=1,334，CSI-300 N=1,220
- Probe：固定市场、当前持仓和Outer candidate，单独中和portfolio-state输入

## 1. 结论

当前checkpoint中的Base可以解释为**稳定的保守切换门槛**，但不能解释为
**当前组合质量分数**。

Base logit在NASDAQ-100和CSI-300中分别稳定在约-1.019和-0.991，对应
Base-only Switch概率约26.5%和27.1%，完整测试集中从未单独触发Switch。
它的主要行为作用是要求raw Adv至少达到约0.012，才允许最终logit越过0。

## 2. 全测试集分布

| Market | Base logit mean | Base logit SD | Base-only probability | Base-only Switch | Required raw Adv |
|---|---:|---:|---:|---:|---:|
| NASDAQ-100 | -1.0192 | 0.0018 | 26.52% | 0/1,334 | 0.01199 |
| CSI-300 | -0.9907 | 0.0032 | 27.08% | 0/1,220 | 0.01156 |

## 3. 当前状态受控消融

每次只改变一个显式portfolio-state字段，其他输入全部固定。表中的概率变化为
`sigmoid(Base_original) - sigmoid(Base_ablated)`，单位是百分点。

| Market | Ablated state | Mean Base-logit effect | Mean Base-probability effect (pp) | 95th percentile absolute effect (pp) |
|---|---|---:|---:|---:|
| NASDAQ-100 | Holding age | -0.000661 | -0.0129 | 0.0289 |
| NASDAQ-100 | Drawdown | -0.000119 | -0.0023 | 0.0067 |
| NASDAQ-100 | Segment return | -0.000040 | -0.0008 | 0.0096 |
| NASDAQ-100 | All three | -0.000836 | -0.0163 | 0.0372 |
| CSI-300 | Holding age | +0.000325 | +0.0064 | 0.0146 |
| CSI-300 | Drawdown | -0.001997 | -0.0394 | 0.0831 |
| CSI-300 | Segment return | -0.001170 | -0.0231 | 0.0548 |
| CSI-300 | All three | -0.002900 | -0.0573 | 0.1082 |

三个当前状态字段全部中和后，所需raw Adv门槛的平均变化仅为：

- NASDAQ-100：0.000012；
- CSI-300：0.000042。

相比Adv自身标准差（NASDAQ约0.0104，CSI约0.0057），这一门槛变化很小。

## 4. 为什么简单相关不能作为Base状态解释

未经控制时，Base与持有时长的Spearman相关在NASDAQ-100和CSI-300分别为
0.476和0.557；但Base也同时接收市场序列、持仓和candidate信息，持有时长又由
此前Controller动作决定。因此该相关不能证明Base由持有时长驱动。

在同时控制持有时长、区间收益和回撤的rank-HAC回归中，持有时长系数仍为正，
但受控输入消融显示其对Base概率的实际影响不超过百分之零点几。相关性反映排序，
不代表具有足以改变动作的量级。

当前收益和回撤的关系也不跨市场一致：

- NASDAQ中Base与回撤的未控制相关为正，但控制持有时长后消失；
- CSI中Base与区间收益的未控制相关为正，但控制持有时长后不显著；
- CSI中控制后的回撤系数反而为负。

因此不能将Base稳定解释为“组合越差，Base越高”。

## 5. 可以采用的论文表述

> The Base term acts as a conservative switching prior. It remains close to
> a 27% Base-only switching probability and imposes an approximately 0.012
> raw-Adv threshold before a candidate can trigger reallocation. Portfolio
> status is shown explicitly through holding-period return, drawdown, and
> holding age, rather than treating Base as a calibrated portfolio-quality
> score.

不建议写：

> Base measures the quality of the current portfolio.

代码结构也不支持严格的current-only解释，因为Base和Adv共享同时包含当前组合、
candidate及二者差异的表征。

## 6. 统计解释边界

- 11/11 fallacy scan completed。
- 分市场报告，避免Simpson汇总反转。
- 推断单位是自由决策日，不外推到强制动作或单只股票。
- 输入消融固定其他模型输入，但仅说明模型行为敏感性，不等价于经济因果效应。
- 未依据事后收益选择样本，避免case selection和look-elsewhere偏差。
- 时间序列存在自相关；简单相关仅用于描述，核心结论以全样本分布和受控输入消融为主。
- Base接近初始化偏置-1.0，因此“保守先验”解释强于“学习到的状态质量”解释。

## 7. 复现文件

- `controller_base_state_ablation.csv`：2,554个自由决策日的逐日输入消融结果；
- `scripts/probe_controller_base_state.py`：模型重放和状态消融脚本。
