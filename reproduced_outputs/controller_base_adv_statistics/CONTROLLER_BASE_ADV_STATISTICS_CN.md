# Controller Base 与 Adv 全测试集统计分析

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-29
- Verification Status：VERIFIED
- Version Label：controller_base_adv_stats_v1
- Markets：NASDAQ-100 seed 49；CSI-300 seed 90
- Sample：完整测试集中的自由 Controller 决策日
- HAC：Newey–West，5阶滞后
- Confidence interval：20日循环区块 bootstrap，2,000次，固定随机种子

## 1. 统计对象

- **Base**：`base_exit_logit`，以及其单独对应的 `sigmoid(base_exit_logit)`；
- **Adv**：`switch_advantage_pred`，实际策略修正为
  `1.9 × tanh(switch_advantage_pred / 0.02)`；
- **训练对齐目标**：决策时 Switch 与 Hold 两个候选组合在剩余持有期内的反事实
  log-return 差，扣除二者增量换手成本；
- **固定期限稳健性**：同一决策日未来20日和30日的冻结组合反事实收益差。

## 2. Base 与 Adv 的机制分工

| Market     |    N |   Base prob mean |   Base prob SD (pp) |   Base-only Switch |   Adv/Base dynamic SD |   Adv-only Switch rate |   Final Switch rate |   Positive Adv blocked |   Median Adv threshold |
|:-----------|-----:|-----------------:|--------------------:|-------------------:|----------------------:|-----------------------:|--------------------:|-----------------------:|-----------------------:|
| NASDAQ-100 | 1334 |           0.2652 |              0.0344 |                  0 |               456.955 |                 0.5    |              0.1732 |                    436 |                 0.012  |
| CSI-300    | 1220 |           0.2708 |              0.0631 |                  0 |               155.005 |                 0.7221 |              0.0754 |                    789 |                 0.0116 |

Base 的概率在两个市场都稳定在约27%，单独从未超过0.5。Adv 的动态标准差是
Base 的约155–457倍，说明最终概率的时间变化几乎完全来自Adv。Base的实际作用是
设置约0.012的正Adv门槛，把“Adv为正就切换”的高频策略过滤为更低频的最终Switch。

## 3. Base 与 Adv 是否对应真实反事实优势

| Market     | Target                            | Signal   |   Spearman rho |   CI low |   CI high |   NW p |   BH q |   Sign AUC |
|:-----------|:----------------------------------|:---------|---------------:|---------:|----------:|-------:|-------:|-----------:|
| NASDAQ-100 | Training-aligned adaptive horizon | Base     |          0     |   -0.088 |     0.09  |  0.996 |  0.996 |      0.503 |
| NASDAQ-100 | Training-aligned adaptive horizon | Adv      |          0.04  |   -0.066 |     0.136 |  0.353 |  0.683 |      0.518 |
| NASDAQ-100 | Fixed 20-day robustness           | Base     |          0.036 |   -0.078 |     0.153 |  0.434 |  0.805 |      0.495 |
| NASDAQ-100 | Fixed 20-day robustness           | Adv      |         -0.007 |   -0.125 |     0.098 |  0.876 |  0.876 |      0.513 |
| CSI-300    | Training-aligned adaptive horizon | Base     |         -0.028 |   -0.144 |     0.083 |  0.512 |  0.683 |      0.485 |
| CSI-300    | Training-aligned adaptive horizon | Adv      |          0.121 |    0.017 |     0.219 |  0.003 |  0.013 |      0.564 |
| CSI-300    | Fixed 20-day robustness           | Base     |         -0.023 |   -0.129 |     0.085 |  0.604 |  0.805 |      0.484 |
| CSI-300    | Fixed 20-day robustness           | Adv      |          0.058 |   -0.048 |     0.161 |  0.177 |  0.708 |      0.542 |

NASDAQ-100中，Base与Adv都没有形成可重复的优势排序；Adv在训练对齐目标上的
Spearman相关仅约0.04，Newey–West检验不显著。CSI-300中，Adv对训练对齐目标存在
弱正相关，且在HAC和BH校正后仍保留统计证据；Base仍接近随机。固定20日目标下，
两个市场的Adv关系都没有通过HAC检验，因此CSI结果应描述为“期限匹配下的弱证据”，
不能描述为跨期限稳定预测。

## 4. 极端分组检验

| Market     | Signal   |   Bottom Q mean |   Top Q mean |   Q4-Q1 bp/day |   CI low |   CI high |   NW p |   BH q |
|:-----------|:---------|----------------:|-------------:|---------------:|---------:|----------:|-------:|-------:|
| NASDAQ-100 | Base     |           0.013 |        2.887 |          2.874 |   -2.663 |     9.321 |  0.367 |  0.49  |
| NASDAQ-100 | Adv      |           0.81  |        1.305 |          0.494 |   -6.514 |     7.232 |  0.879 |  0.879 |
| CSI-300    | Base     |          -4.791 |       -9.206 |         -4.414 |  -15.23  |     7.148 |  0.354 |  0.49  |
| CSI-300    | Adv      |         -13.017 |       -1.423 |         11.595 |    2.325 |    20.26  |  0.002 |  0.008 |

CSI-300的Adv最高四分位相对于最低四分位对应更高的事后优势；NASDAQ-100没有
相同证据。Base的四分位差在两个市场都不稳定。

## 5. 最终 Controller 动作的反事实结果

| Market     | Outcome                           |   Hold mean |   Switch mean |   Switch-Hold bp/day |   CI low |   CI high |   NW p |   BH q |
|:-----------|:----------------------------------|------------:|--------------:|---------------------:|---------:|----------:|-------:|-------:|
| NASDAQ-100 | Training-aligned adaptive horizon |      -0.306 |         0.24  |                0.546 |   -3.041 |     4.11  |  0.706 |  0.706 |
| NASDAQ-100 | Fixed 20-day                      |       1.991 |         0.024 |               -1.966 |   -4.268 |     0.253 |  0.027 |  0.044 |
| NASDAQ-100 | Fixed 30-day                      |       1.691 |         0.09  |               -1.601 |   -3.583 |     0.223 |  0.029 |  0.029 |
| CSI-300    | Training-aligned adaptive horizon |      -6.699 |         0.424 |                7.122 |    1.45  |    12.874 |  0.007 |  0.013 |
| CSI-300    | Fixed 20-day                      |      -2.346 |         1.504 |                3.851 |    0.188 |     7.973 |  0.044 |  0.044 |
| CSI-300    | Fixed 30-day                      |      -1.342 |         2.896 |                4.239 |    1.42  |     7.777 |  0.006 |  0.013 |

- CSI-300：Switch日相对于Hold日在三个窗口中均表现更好；20日结果的
  Newey–West p与两市场BH q均约为0.044，属于边界性证据，30日结果更稳定。
- NASDAQ-100：训练对齐窗口无显著差异；固定20/30日的点估计为负，虽然
  Newey–West p约为0.027/0.029，但20日区块bootstrap置信区间为
  [−4.27, 0.25]、30日为[−3.58, 0.22]，均包含0。两种推断不一致，因此只能
  视为负向警示，不能断言Controller在NASDAQ上稳定改善或稳定损害切换收益。

## 6. 可以支持的结论

1. **结构分工得到强支持**：Base是低波动的保守阈值，Adv承担动态调制；
2. **Base不是独立预测器**：它不单独触发Switch，对真实优势的AUC约为0.5；
3. **Adv的统计解释具有市场差异**：CSI-300存在弱但可检验的期限匹配信号，
   NASDAQ-100没有；
4. **联合Controller的正向证据主要来自CSI-300**，NASDAQ结果只能支持“保守过滤机制”，
   不能支持“稳定改善切换收益”。

## 7. 解释边界与统计谬误检查

- Coverage：11/11。
- Simpson：NASDAQ与CSI分别报告，未用汇总市场结果掩盖方向差异；
- Ecological：推断单位保持为决策日，不外推到单只股票；
- Berkson：只分析自由决策日是模型机制规定的条件样本，结论不外推到强制动作；
- Collider：未加入事后表现作为控制变量；
- Base-rate neglect：同时报告正优势比例、AUC和balanced accuracy；
- Regression to mean：使用全测试集，不按极端事后收益筛选case；
- Survivorship：使用完整测试轨迹；不足20/30个未来交易日的末端样本从对应固定期限检验中排除；
- Look-elsewhere：训练对齐目标预先作为主结果，固定20/30日作为稳健性；
- Forking paths：分析属于事后解释性验证，不能当作预注册确认性证据；
- Correlation/causation：只陈述关联和反事实对齐，不作因果归因；
- Reverse causality：预测量先于未来收益，但模型选择和训练过程仍可能造成样本内偏差。

## 8. 文件

- `controller_base_adv_distribution.csv`：各信号完整分布；
- `controller_base_adv_roles.csv`：Base/Adv阈值与动作分工；
- `controller_base_adv_alignment.csv`：相关、区块CI、HAC、FDR与AUC；
- `controller_base_adv_quartiles.csv`：极端四分位检验；
- `controller_base_adv_decision_effect.csv`：Switch/Hold与Adv正负分组结果；
- `controller_base_adv_statistics.png/.pdf`：核心统计图。
