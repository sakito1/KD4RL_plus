# Inner-Actor 全测试期日级统计分析（论文 Selected Models）

## Material Passport

- Verification Status: ANALYZED
- Code: `KD_abk/KD4RL_plus` master `c9c779e39f8d3c28f784938841c453fe956adede`
- NAS checkpoint SHA256: `e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d`
- SH checkpoint SHA256: `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94`

## 模型来源

```text
NASDAQ-100:
/home/tongwenxuan/KD4RL_plus/results/
controller_first_joint_lowlr_retry_20260622_02/
lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/
nas/ppo/seed_49/checkpoints/best_model.pth

CSI-300:
/home/tongwenxuan/KD4RL_plus/results/
e2e_standard_joint_lowlr_20260622_01/
lookback60_hold30_standard_joint_lowlr_nas49_sh90/
sh/ppo/seed_90/checkpoints/best_model.pth
```

## 案例图复现核对

| Market | Selected window Mean r | Positive days | 截图 |
|---|---:|---:|---:|
| NASDAQ-100 | 0.45697 | 73.33% | 0.46 / 73% |
| CSI-300 | 0.43985 | 73.33% | 0.44 / 73% |

两张 Inner-Actor 案例图的统计与截图一致。

## 全测试期日级统计

| Market | Days | Mean Spearman IC | IC p | IC block 95% CI | Fair net alpha (bp/day) | Alpha p | Alpha block 95% CI | Positive permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NASDAQ-100 | 1,369 | 0.0099 | 0.313 | [-0.0090, 0.0282] | -0.040 | 0.560 | [-0.162, 0.084] | 0.830 |
| CSI-300 | 1,247 | -0.0051 | 0.606 | [-0.0229, 0.0135] | -0.199 | 0.098 | [-0.408, -0.006] | 0.993 |

环境 reward 口径（执行组合承担全部交易成本、base 不承担反事实成本）：

| Market | Reward alpha (bp/day) | NW(5) p |
|---|---:|---:|
| NASDAQ-100 | -0.107 | 0.114 |
| CSI-300 | -0.285 | 0.018 |

## 结论

1. 正确的 selected checkpoint 可以复现论文中的局部 Inner-Actor 案例图。
2. 完整测试期的日级 IC 在两个市场都不显著。
3. NAS 的公平净增量 alpha 接近零；SH 为负但 Newey-West p=0.098，不能作为常规5%水平下的显著结果。
4. 因此案例图可以表述为“selected local alignment”，不能扩展成“完整测试期稳定赚取日级波动收益”。
5. `eval_end_to_end_explain.py` 当前保存的旧 `inner_alpha` 存在时间错位，本统计由 weights、原始价格和环境成本公式重新计算。

## 数据文件

```text
inner_actor_daily_statistics.csv
inner_actor_daily_series.csv
inner_actor_alignment_validation.csv
```
