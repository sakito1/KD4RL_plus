# Controller 实际持仓监督与 MSE 简化设计

## 目标

在不改变 Controller 双分支网络尺寸、Outer+Inner checkpoint、30 日强制
切换和 Top-Tail 比例约束的前提下，统一监督数据来源和组合定义：

1. Risk 与 Advantage 使用 MSE；
2. 两个回归目标保留 \(20\) 倍缩放，避免被 BCE、Rate 和 PG 淹没；
3. 监督数据由当前 Controller 探索产生；
4. Risk 和 Advantage 的 Hold 侧均以决策前实际漂移持仓为起点；
5. 正式脚本只保留 `sup_pg`，参数只定义一次。

## 组合与目标定义

在自由决策日 \(t\)，令

\[
\mathbf w_t^{\mathrm{drift}}
\]

表示决策前、经过上一日价格漂移后的实际持仓。该持仓包含此前 Inner Actor
调整所产生的真实配置结果。

### Risk

Risk 分支输入中的持仓聚合和组合状态均使用
\(\mathbf w_t^{\mathrm{drift}}\)。Risk target 为该实际持仓在剩余允许持有期
\(H_t\) 内相对等权市场的最大回撤：

\[
D_t =
\operatorname{MDD}
\left(
\frac{V(\mathbf w_t^{\mathrm{drift}})}
     {V^{\mathrm{market}}}
\right).
\]

回归损失改为

\[
\mathcal L_{\mathrm{risk}}
=
\operatorname{MSE}
\left(
\widehat D_t,20D_t
\right).
\]

### Advantage

Hold 分支直接使用决策前实际漂移持仓
\(\mathbf w_t^{\mathrm{drift}}\)，不再次调用 Inner Actor 生成 Hold 权重。

Switch 分支使用 Manager 候选 active base 经冻结的 Inner Actor
确定性执行后得到的候选执行权重
\(\mathbf w_t^{\mathrm{switch}}\)。

优势定义为

\[
A_t =
R(\mathbf w_t^{\mathrm{switch}})
-R(\mathbf w_t^{\mathrm{drift}})
-C\left(
\mathbf w_t^{\mathrm{drift}}
\rightarrow
\mathbf w_t^{\mathrm{switch}}
\right).
\]

Hold 不产生即时再平衡成本；Switch 成本从决策前实际持仓计算。回归损失为

\[
\mathcal L_{\mathrm{adv}}
=
\operatorname{MSE}
\left(
\widehat A_t,20A_t
\right).
\]

经济标签继续使用未缩放的原始 \(D_t\) 和 \(A_t\)：

\[
y_t =
\mathbb I
\left[
(D_t\ge0.05\land A_t>0)
\lor
(A_t\ge0.05)
\right].
\]

## 监督数据采集

删除 `controller_aux_pretrain_offpolicy`。监督预训练使用当前 Controller 的
随机策略采集轨迹：

1. 从训练期均匀覆盖的 12 个起点分别运行 300 日 Controller rollout；
2. Controller 在所有正常自由决策日按当前概率采样 Hold/Switch；
3. 初始建仓和 30 日上限触发的强制动作不记录为监督决策；
4. 完成全部 12 条轨迹后冻结该批数据；
5. 在同一批轨迹上重新前向并更新 30 次；
6. 监督预训练结束后进行 3 轮 Controller counterfactual PG。

监督采集期间 Outer Actor 与 Inner Actor 均冻结并确定性执行。仅 Controller
参数更新。

## Loss

监督预训练保持

\[
\mathcal L_{\mathrm{pre}}
=
\mathcal L_{\mathrm{risk}}
+\mathcal L_{\mathrm{adv}}
+\mathcal L_{\mathrm{switch}}
+\mathcal L_{\mathrm{rate}},
\]

其中四项系数均为 \(1\)。

PG 阶段保持

\[
\mathcal L_{\mathrm{controller}}
=
\mathcal L_{\mathrm{PG}}
+0.1
\left(
\mathcal L_{\mathrm{risk}}
+\mathcal L_{\mathrm{adv}}
+\mathcal L_{\mathrm{switch}}
+\mathcal L_{\mathrm{rate}}
\right)
-0.001\mathcal H.
\]

Switch 使用不做类别平衡的普通 BCE。Rate 使用 batch 级 5%--15%
Top-Tail logit margin。PG 使用全部日级自由决策 log-prob 的统一平均。

## 正式训练脚本

`train_sh/explore_controller_from_nas45_outer_inner.sh` 仅保留正式
`sup_pg` 流程：

- 删除 `MODE`、`probe`、`pg_only`、`sup_only` 分支；
- 监督预训练轮数固定为 1；
- replay 次数固定为 30；
- Controller PG 固定为 3 轮；
- 删除重复的临时变量赋值；
- 删除 `controller_aux_pretrain_offpolicy`；
- 保留 seed、GPU、输出目录和 checkpoint 的环境变量覆盖能力；
- 保留 `DRY_RUN` 和已有输出保护；
- 保留 `--skip_test`，训练结束后测试由独立测试命令执行。

## 不变范围

- Outer+Inner seed-45 checkpoint 及其参数不变；
- Controller 网络隐藏维度和参数形状不变；
- 30 日训练/验证最大持有期不变；
- Controller 日级自由决策不变；
- 5% 经济标签阈值不变；
- 5%--15% 切换比例约束不变；
- 不增加新分支、新 loss 或额外超参数。

## 验收

自动测试至少验证：

1. Risk target 使用实际 `prev_weights` 漂移，而不是 `prev_base_weight`；
2. Advantage Hold 权重等于决策前 `weights_drift`；
3. Advantage Switch 成本从 `weights_drift` 计算；
4. Risk 和 Advantage 均调用 MSE；
5. 监督预训练走 Controller rollout，而不是固定 30 日 off-policy 路径；
6. 脚本只包含 `sup_pg` 且无重复参数定义；
7. Outer/Inner 参数在 Controller-only 训练中保持冻结；
8. 现有 Controller 定向测试无回归。
