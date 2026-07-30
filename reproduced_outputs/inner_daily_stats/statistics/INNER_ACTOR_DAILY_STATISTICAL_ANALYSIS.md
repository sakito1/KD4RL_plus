# Inner-Actor 全测试期日级统计分析

> **已废弃：** 本报告使用的是 `results/end` 中的旧归档 checkpoint，不是论文最终选择的模型。论文 selected 模型统计请以 `reproduced_outputs/inner_daily_stats_paper_selected/statistics/` 为准。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Analysis Date: 2026-07-27
- Code Version: `KD_abk/KD4RL_plus` master, commit `c9c779e39f8d3c28f784938841c453fe956adede`
- NAS Model: seed 49, `best_model.pth`, SHA256 `7152fe3588ac3528e7ae54fafe440aeee516293b6b486c9d1420dbf253f4e55e`
- SH Model: seed 90, `best_model.pth`, SHA256 `8022c8cae48be9232fee9dd00337230b2cd88071587a996b0de73d3fed0e6a42`

## 分析问题

检验 Inner-Actor 的日级权重调整是否与 action 之后的单日横截面收益方向一致，以及这些调整是否相对 base portfolio 产生正的日级增量收益。

定义：

```text
inner tilt = executed weight - base weight
```

每天只在实际活跃的10只组合资产中计算：

```text
IC_t = SpearmanCorr(inner_tilt_t, next_day_relative_return_t)
```

经济收益使用两个口径：

```text
Gross alpha
= log(executed_weight × next_day_price_ratio)
  - log(base_weight × next_day_price_ratio)

Fair net alpha
= log(executed gross × (1 - executed turnover × cost))
  - log(base gross × (1 - base turnover × cost))
```

交易成本率使用环境配置中的 `5e-5`。

## 全测试期结果

| Market | Days | Mean Spearman IC | NW(5) p | Block 95% CI | IC > 0 | Gross alpha (bp/day) | Gross p | Fair net alpha (bp/day) | Net p | Net block 95% CI (bp/day) | Net alpha Sharpe | Cumulative net alpha | Positive permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NASDAQ-100 | 1,369 | 0.0099 | 0.313 | [-0.0090, 0.0282] | 51.28% | -0.046 | 0.495 | -0.040 | 0.560 | [-0.162, 0.084] | -0.249 | -0.54% | 0.830 |
| CSI-300 | 1,247 | -0.0101 | 0.313 | [-0.0280, 0.0086] | 47.96% | -0.248 | 0.039 | -0.237 | 0.048 | [-0.443, -0.048] | -0.852 | -2.91% | 0.998 |

Pearson IC 和逐资产方向命中率同样没有形成正向证据：

| Market | Mean Pearson IC | Mean directional hit rate |
|---|---:|---:|
| NASDAQ-100 | -0.0014 | 49.83% |
| CSI-300 | -0.0180 | 48.56% |

环境实际 reward 定义使用执行组合的全部交易成本、但不向 base 收取反事实成本。按这个更保守的口径：

| Market | Environment reward alpha (bp/day) | NW(5) t | p |
|---|---:|---:|---:|
| NASDAQ-100 | -0.107 | -1.581 | 0.114 |
| CSI-300 | -0.327 | -2.717 | 0.0066 |

## Switch 与 Hold 日期

NASDAQ-100：

| Group | Mean IC | IC p | Net alpha (bp/day) | Net p |
|---|---:|---:|---:|---:|
| Hold days | 0.0102 | 0.363 | -0.048 | 0.565 |
| Switch days | 0.0088 | 0.681 | -0.003 | 0.823 |

CSI-300：

| Group | Mean IC | IC p | Net alpha (bp/day) | Net p |
|---|---:|---:|---:|---:|
| Hold days | -0.0068 | 0.534 | -0.257 | 0.053 |
| Switch days | -0.0393 | 0.161 | -0.065 | 0.504 |

没有发现“只在非切仓日有效”或“只在切仓日有效”的稳定证据。

## 与案例图的关系

案例脚本会扫描多个重叠窗口，再选择评分最高的窗口：

- NASDAQ-100 案例：Mean corr = 0.457，positive days = 73.33%。
- CSI-300 案例：Mean corr = 0.331，positive days = 70.00%。

但是完整测试期的 Mean IC 分别只有 `0.0099` 和 `-0.0101`。因此案例图可以展示局部行为，但不能用于证明 Inner-Actor 在完整测试期稳定捕获日级波动收益。

## Trace 时间对齐问题

`eval_end_to_end_explain.py` 当前记录的 `inner_alpha` 不能直接用于统计：

- `base_log_return` 使用 action 后的 `t -> t+1` 收益。
- `exec_log_return` 却通过 `portfolio_value_after / portfolio_value_before` 得到，主要对应此前已实现的 `t-1 -> t` 收益和当前成本。

旧 trace `inner_alpha` 与正确同日 reward alpha 的平均绝对误差：

| Market | MAE |
|---|---:|
| NASDAQ-100 | 0.014882 |
| CSI-300 | 0.014623 |

本分析没有使用该错误列，而是由 base/exec weights、原始价格和环境交易成本公式重新计算。价格和权重对齐验证如下：

| Market | Base return MAE | Turnover MAE | Cost-rate MAE |
|---|---:|---:|---:|
| NASDAQ-100 | 3.41e-08 | 1.67e-08 | 8.37e-13 |
| CSI-300 | 3.53e-08 | 1.86e-08 | 9.29e-13 |

## 解释结论

对于当前归档的 NAS seed 49 和 SH seed 90：

1. NASDAQ-100 的日级 action-return 关系与零无法区分，经济增量收益也接近零。
2. CSI-300 没有正向 IC，日级增量收益反而呈弱负向；考虑多重比较后不宜强调其边界显著性，但明确不存在正向证据。
3. 随机置换检验的正向单侧 p 值分别为 0.830 和 0.998，同样不支持 tilt 优于随机资产匹配。
4. 因而不能用这两个 checkpoint 声称“Inner-Actor 在完整测试期稳定赚取日级波动收益”。
5. 可以保留的谨慎表述是：Inner-Actor 在事后选择的局部窗口中表现出 tilt 与未来相对收益的局部对齐；这种模式没有推广到完整测试期。

## 统计风险检查（11/11）

| 风险 | 结论 |
|---|---|
| Simpson's paradox | 两个市场分别报告；年度方向存在异质性，不应合并宣称全局有效。 |
| Ecological fallacy | 未从市场级均值推断单只股票必然有效。 |
| Berkson/selection bias | 案例窗口由最大评分事后选取，存在明显选择偏差。 |
| Collider bias | 主分析未加入可能的 collider 控制变量。 |
| Base-rate neglect | 不适用于当前连续收益分析。 |
| Regression to mean | 最大评分窗口可能向均值回归；完整测试期结果已用于校正解释。 |
| Survivorship bias | 固定股票池是否包含幸存者偏差尚未审计。 |
| Look-elsewhere effect | `select_window` 扫描多个窗口和资产，案例统计不能视为确认性检验。 |
| Garden of forking paths | 本分析属于事后验证；报告了 Spearman/Pearson、NW(5)/NW(20)、block bootstrap 和置换检验。 |
| Correlation ≠ causation | IC 只能说明关联，不能证明 Inner action 导致未来收益。 |
| Reverse causality/leakage | action 在收益实现前产生；但特征生成全链路的数据泄漏仍需单独审计。 |

## 输出文件

```text
inner_actor_daily_statistics.csv
inner_actor_daily_subgroups.csv
inner_actor_daily_series.csv
inner_actor_alignment_validation.csv
```
