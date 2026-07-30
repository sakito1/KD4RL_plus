# Controller Gate 与 Advantage 项可解释案例报告

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-28
- Verification Status：VERIFIED
- Version Label：controller_gate_adv_case_v1
- 模型：NASDAQ-100 seed 49；CSI-300 seed 90
- 数据：完整测试集逐日重放
- 代码：`KD_abk/KD4RL_plus`
- Checkpoint：`KD4RL_plus/results/end`

## 1. 公式与代码变量

Controller 的最终切换概率为：

\[
p_L
=
\sigma\left(
\ell^{gate}_t
+
\eta\tanh\left(
\frac{\ell^{adv}_t}{c_{adv}}
\right)
\right).
\]

代码变量对应关系如下：

| 论文符号 | 代码字段 | 含义 |
|---|---|---|
| \(\ell^{gate}_t\) | `base_exit_logit` | Gate输出的基础退出logit |
| \(\ell^{adv}_t\) | `switch_advantage_pred` | Advantage head预测的切换优势 |
| \(c_{adv}\) | `controller_switch_adv_logit_scale=0.02` | Advantage缩放尺度 |
| \(\eta\) | `controller_switch_adv_logit_coef=1.9` | Advantage对策略logit的最大调制强度 |
| \(\eta\tanh(\ell^{adv}_t/c_{adv})\) | `exit_logit-base_exit_logit` | Advantage实际加入策略的logit修正 |
| 最终logit | `exit_logit` / `policy_logit` | Gate与adv修正之和 |
| \(p_L\) | `exit_prob` | 最终切换概率 |

测试阶段使用确定性阈值：

\[
\text{Switch}
\quad\Longleftrightarrow\quad
p_L>0.5
\quad\Longleftrightarrow\quad
\text{exit\_logit}>0.
\]

## 2. 重放一致性

使用 `KD_abk` 实验代码加载 `KD4RL_plus/results/end` 中的checkpoint，重新生成 `base_exit_logit` 和 `switch_advantage_pred`。

| 检查 | NASDAQ-100 | CSI-300 |
|---|---:|---:|
| 重放总行数 | 1,369 | 1,247 |
| 自由决策 | 1,334 | 1,220 |
| `exit_prob` 最大误差 | 0 | 0 |
| `policy_logit` 最大误差 | 0 | 0 |
| 动作不一致数 | 0 | 0 |
| 公式重构最大误差 | \(2.85\times10^{-7}\) | \(2.14\times10^{-7}\) |

这说明新导出的Gate和adv字段与原论文结果使用的Controller轨迹完全一致。

## 3. 全测试集中的 Gate 与 Adv 分工

| 指标 | NASDAQ-100 | CSI-300 |
|---|---:|---:|
| Gate logit范围 | [−1.024, −1.012] | [−1.000, −0.982] |
| Gate单独对应的概率范围 | [0.264, 0.267] | [0.269, 0.273] |
| Gate单独触发的Switch | 0 | 0 |
| Adv将最终logit推过0的Switch | 231 | 92 |
| 正Adv但未越过阈值的Hold | 436 | 789 |
| 负Adv并继续Hold | 667 | 339 |
| 预测Adv与真实Switch优势的Spearman相关 | 0.040 | 0.141 |

因此，当前checkpoint中两项的功能非常清楚：

1. **Gate不是一个单独主动触发切换的模块。** 它在测试集内输出接近 −1 的稳定负logit，相当于约27%的基础切换概率，构成保守的Hold先验；
2. **Adv是决定是否越过0.5阈值的主要动态项。** 测试集中的全部自由Switch都由正adv修正将最终logit从负值推到正值；
3. **Gate保留否决权。** 较弱的正adv不足以抵消约 −1 的Gate基线时，最终动作仍然是Hold；
4. **负adv会强化Hold。** 当候选新组合被Adv head判断为更差时，adv修正进一步降低最终切换概率。

需要如实说明：预测Adv与事后真实优势的对应关系在NASDAQ-100上很弱，在CSI-300上较强但仍属于弱相关。因此，下面的案例用于解释计算机制，不代表每次Adv判断都正确。

## 4. Case A：正 Adv 推翻 Gate 并触发 Switch

这两个日期就是现有论文Controller案例使用的日期，不是根据本次分析重新选择的最佳样本。

| 市场 | 日期 | Gate logit | Gate概率 | Adv预测 | Adv logit修正 | 最终logit | 最终概率 | 动作 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| NASDAQ-100 | 2021-04-19 | −1.020 | 0.265 | +0.0233 | +1.564 | +0.543 | 0.633 | Switch |
| CSI-300 | 2021-07-07 | −0.990 | 0.271 | +0.0123 | +1.042 | +0.052 | 0.513 | Switch |

### NASDAQ-100：2021-04-19

Gate单独给出的切换概率只有26.5%，会选择Hold。正Adv产生+1.564的logit修正，将最终logit从−1.020推到+0.543，最终切换概率上升到63.3%，因此触发Switch。

未来29日冻结反事实：

| 路径 | 累计收益 | MDD |
|---|---:|---:|
| Hold | −1.90% | 8.82% |
| Switch | +0.68% | 5.51% |

Switch相对于Hold的日均优势为+8.94 bp/日。该案例说明Adv项可以克服保守Gate，并且本次修正方向与后续反事实结果一致。

### CSI-300：2021-07-07

Gate单独给出的切换概率为27.1%。Adv产生+1.042的logit修正，将最终logit从−0.990推到+0.052；最终概率为51.3%，刚好越过切换阈值。

未来22日冻结反事实：

| 路径 | 累计收益 | MDD |
|---|---:|---:|
| Hold | −12.11% | 13.69% |
| Switch | +4.62% | 8.23% |

Switch相对于Hold的日均优势为+79.22 bp/日。这个案例最清楚地展示了公式中Adv项的作用：Gate倾向保留旧组合，但候选组合优势足够大时，Adv将动作翻转为Switch。

## 5. Case B：Gate 阻止较弱的正 Adv

这类案例按照“正Adv、最终仍Hold、概率最接近0.5、比较窗口不少于10日”选择，用来解释两项如何在阈值附近竞争。

| 市场 | 日期 | Gate logit | Adv修正 | 最终logit | 最终概率 | 动作 | 真实Switch优势 |
|---|---|---:|---:|---:|---:|---|---:|
| NASDAQ-100 | 2024-04-04 | −1.023 | +1.014 | −0.009 | 0.4977 | Hold | +1.88 bp/日 |
| CSI-300 | 2024-04-12 | −0.994 | +0.994 | −0.00015 | 0.49996 | Hold | +6.14 bp/日 |

两个案例中Adv都认为Switch具有一定吸引力，但修正强度略小于Gate的负基线，最终概率没有严格超过0.5，因此继续Hold。

事后冻结反事实显示Switch收益略好，这说明保守Gate会降低错误切换，同时也可能错过边际切换机会。它不是“Gate正确”的案例，而是展示Gate–Adv阈值权衡及其代价。

## 6. Case C：负 Adv 与 Gate 共同强化 Hold

| 市场 | 日期 | Gate logit | Adv预测 | Adv修正 | 最终概率 | Hold收益 | Switch收益 | Switch−Hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NASDAQ-100 | 2022-06-17 | −1.014 | −0.0218 | −1.514 | 0.0739 | +8.53% | +6.67% | −8.27 bp/日 |
| CSI-300 | 2020-07-16 | −0.989 | −0.0102 | −0.890 | 0.1325 | +10.73% | +3.93% | −45.26 bp/日 |

### NASDAQ-100：2022-06-17

Gate原本已经倾向Hold，负Adv又加入−1.514的修正，使切换概率从Gate基线26.6%下降到7.4%。未来21日中，Hold收益和MDD均优于Switch：

```text
Hold：收益 +8.53%，MDD 2.87%
Switch：收益 +6.67%，MDD 5.70%
```

### CSI-300：2020-07-16

负Adv加入−0.890的修正，使切换概率从27.1%下降到13.2%。未来14日中：

```text
Hold：收益 +10.73%，MDD 3.15%
Switch：收益 +3.93%，MDD 5.34%
```

这两个案例说明：当Adv head认为候选配置劣于当前配置时，它不会单独创造新动作，而是强化Gate的保守Hold倾向。

## 7. 最可靠的解释

当前模型中的Gate和Adv不是两个对等的切换信号，而更接近以下结构：

```text
Gate：提供约27%切换概率的保守Hold先验
                   ↓
Adv > 0：提高最终logit；足够强时触发Switch
Adv < 0：降低最终logit；强化Hold
                   ↓
最终概率严格超过0.5才Switch
```

论文中可以表述为：

> The base gate acts as a conservative hold prior, while the bounded advantage term provides the principal state-dependent modulation. In both test markets, the base gate alone never crosses the switching threshold; all learned free switches occur when a positive advantage correction is sufficiently large to overturn the negative gate logit. Conversely, weak positive corrections remain blocked, and negative corrections reinforce holding.

不建议表述为：

> Gate本身能够根据状态主动识别切换时机。

因为当前checkpoint的 `base_exit_logit` 变化范围很窄，而且单独从未越过切换阈值。真正产生动作差异的是Adv调制项。

## 8. 文件

| 文件 | 内容 |
|---|---|
| `controller_gate_adv_cases_nas.png/.pdf` | NASDAQ-100三个案例 |
| `controller_gate_adv_cases_sh.png/.pdf` | CSI-300三个案例 |
| `controller_gate_adv_case_summary.csv` | 六个案例的全部数值 |
| `controller_gate_adv_audit.csv` | 全测试集Gate/Adv分工统计 |
| `controller_gate_adv_trace_nas.csv` | NASDAQ-100逐日Gate/Adv重放 |
| `controller_gate_adv_trace_sh.csv` | CSI-300逐日Gate/Adv重放 |

## 9. 解释边界

1. 单个case用于说明公式如何产生动作，不能代替全测试集统计；
2. Gate输出接近常数是当前两个checkpoint的实证现象，不是模型架构强制规定；
3. `switch_advantage_pred` 是模型决策时的预测量；图中的未来Hold/Switch曲线是事后冻结反事实，只用于检验该次判断；
4. 正Adv并不保证未来Switch一定更好，负Adv也不保证Hold一定更好；
5. NASDAQ-100中预测Adv与真实优势的相关性仅为0.040，因此不应使用少数成功案例声称Adv具有稳定预测能力；
6. CSI-300对应相关性为0.141，方向更明确，但效应仍较弱。
