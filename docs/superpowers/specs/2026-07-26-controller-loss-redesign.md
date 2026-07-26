# Controller Loss 简化重设计

## 目标与范围

本设计保留现有 Controller 双分支网络和五阶段训练流程，仅调整监督目标、轨迹
log-prob 聚合和切换比例约束。第 4 阶段冻结 Outer Actor 与 Inner Actor，仅训练
Controller；第 5 阶段解冻三个模块进行联合微调。

Controller 的最终职责保持不变：

1. Risk 分支判断当前 active base 是否正在失效；
2. Advantage 分支估计 Manager 候选组合相对继续持有的成本调整收益优势；
3. 两个 embedding 拼接后，由融合网络输出切换概率；
4. 轨迹级 PG 以完整投资路径的收益增益作为最终优化目标。

不增加新的预测分支、价值网络、逐日 reward 分配或训练后阈值校准。

## 目标定义

对每个自由决策日 \(t\)，在相同起点和相同剩余允许持有期内构造 Hold 与
Switch 两条冻结路径。

当前 active base 的相对市场风险定义为

\[
D_t =
\operatorname{MDD}
\left(
\frac{V^{\mathrm{hold}}_{t:t+H_t}}
     {V^{\mathrm{market}}_{t:t+H_t}}
\right).
\]

候选组合的成本调整相对优势定义为

\[
\Delta_t =
R_t^{\mathrm{switch}}
-R_t^{\mathrm{hold}}
-\Delta \mathrm{Cost}_t.
\]

其中

\[
H_t=H_{\max}-d_t
\]

是当前 active base 达到最大持有期前的剩余交易日数。

经济切换标签保持为

\[
y_t =
\mathbb I
\left[
(D_t\ge 0.05\land \Delta_t>0)
\lor
(\Delta_t\ge 0.05)
\right].
\]

标签仅使用训练期未来数据生成，不参与验证和测试期推理。

## 双分支监督

Risk 分支输出 \(\widehat D_t\)，使用连续回归监督：

\[
\mathcal L_{\mathrm{risk}}
=
\operatorname{Huber}
\left(
\widehat D_t,\;20D_t
\right).
\]

Advantage 分支输出 \(\widehat\Delta_t\)，直接预测相对优势大小：

\[
\mathcal L_{\mathrm{adv}}
=
\operatorname{Huber}
\left(
\widehat\Delta_t,\;20\Delta_t
\right).
\]

Advantage 不再使用 \(\mathbb I[\Delta_t>0]\) 的二分类目标。连续回归使
advantage embedding 同时保留优势方向和幅度，从而支持最终融合层区分微弱正
优势与显著正优势。

两个 embedding 拼接后产生最终 logit：

\[
z_t =
f_{\mathrm{fusion}}
\left(
[e_t^{\mathrm{risk}};e_t^{\mathrm{adv}}]
\right),
\qquad
p_t^{\mathrm{switch}}=\sigma(z_t).
\]

最终决策监督使用不做类别平衡的普通 BCE：

\[
\mathcal L_{\mathrm{switch}}
=
\operatorname{BCEWithLogits}(z_t,y_t).
\]

Switch 是自然少数类，因此保留训练数据中的真实类别比例。现有 Nasdaq-100
训练窗口中，该经济规则产生 \(575/3480=16.52\%\) 的正标签，不属于极端稀疏
情形。

## Top-Tail 切换比例约束

比例约束在一个训练 batch 的全部自由决策上计算，不对每个 300 日窗口分别
施加。设 batch 中共有 \(N\) 个自由决策，将 logit 从大到小排序：

\[
z_{[1]}\ge z_{[2]}\ge\cdots\ge z_{[N]}.
\]

令

\[
k_{\min}=\lceil0.05N\rceil,
\qquad
k_{\max}=\lfloor0.15N\rfloor,
\qquad
m=0.1.
\]

下限约束为

\[
\mathcal L_{\min}
=
\frac{1}{k_{\min}}
\sum_{i=1}^{k_{\min}}
[m-z_{[i]}]_+^2,
\]

上限约束为

\[
\mathcal L_{\max}
=
\frac{1}{N-k_{\max}}
\sum_{i=k_{\max}+1}^{N}
[m+z_{[i]}]_+^2.
\]

最终比例损失为

\[
\mathcal L_{\mathrm{rate}}
=
\mathcal L_{\min}+\mathcal L_{\max}.
\]

该约束要求排名前 5% 的 logit 至少达到 \(0.1\)，排名 15% 之后的 logit
至多为 \(-0.1\)。中间 3%--15% 的决策由经济标签和 PG 自由决定。因此，
普通 BCE 决定哪些日期排名靠前，比例损失只约束越过 \(p=0.5\) 的数量。

排序操作只用于选择参与约束的 logit。梯度通过被选中的 logit 正常反传；
所有违反上下界的样本都会获得梯度，不依赖温度参数或不可导的硬计数。

## 轨迹级策略梯度

Controller 继续使用完整 300 日路径相对固定 30 日 Outer+Inner 基准的收益
增益：

\[
R^{\mathrm{ctrl}}
=
\log V_T^{\mathrm{controller}}
-\log V_T^{\mathrm{fixed}}.
\]

交易成本已包含在两条实际执行路径中。

每个自由决策仍为日级动作，但轨迹 log-prob 改为所有自由决策的统一平均：

\[
\bar{\ell}_{\mathrm{traj}}
=
\frac{1}{T}
\sum_{t=1}^{T}
\log\pi_\theta(a_t\mid s_t).
\]

轨迹级 PG 损失为

\[
\mathcal L_{\mathrm{PG}}
=
-R^{\mathrm{ctrl}}
\bar{\ell}_{\mathrm{traj}}.
\]

不再先计算 segment 内平均再对 segment 平均。持有 segment 的长度由
Controller 动作决定，按 segment 等权会使短 segment 中的日级动作获得更高
权重。统一日级平均在固定长度 rollout 中只是标准轨迹 log-prob 求和的常数
缩放，不引入逐日 reward。

## 训练损失

监督预训练阶段使用

\[
\mathcal L_{\mathrm{pre}}
=
\mathcal L_{\mathrm{risk}}
+\mathcal L_{\mathrm{adv}}
+\mathcal L_{\mathrm{switch}}
+\mathcal L_{\mathrm{rate}}.
\]

PG 阶段使用

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

第 5 阶段联合微调沿用相同 Controller 损失，同时保留 Outer Actor 与 Inner
Actor 原有训练目标。

## 实现约束

1. Risk 输入应与 active-base risk target 的持仓定义保持一致；
2. Advantage target 必须使用 Inner Actor 执行后的 Hold 与 Switch 权重，并
   扣除两条路径的增量交易成本；
3. Top-Tail 约束只统计正常自由决策，不包含起始强制切换和 30 日上限切换；
4. 采集动作与重新计算 log-prob 时必须使用一致的策略分布；Controller
   Dropout 应关闭，或保证两次前向使用相同 mask；
5. 一个 batch 的轨迹全部采集完成后再更新 Controller，避免数据采集过程中
   策略参数变化；
6. 不再启用原 soft switch-rate band loss、类别平衡 BCE 或 advantage-sign
   BCE。

## 训练诊断与验收

每轮至少记录：

- Risk Huber loss，以及 \(\widehat D_t\) 与 \(D_t\) 的相关系数；
- Advantage Huber loss，以及 \(\widehat\Delta_t\) 与 \(\Delta_t\) 的相关系数；
- 普通 BCE、正负标签的平均 switch probability 和 probability gap；
- batch 内 \(p_t>0.5\) 的真实比例；
- Top-3%、Top-15% 边界 logit；
- 训练轨迹的收益增益和自由切换次数；
- 验证集自由切换次数、总收益与最大回撤。

快速验证通过条件：

1. Advantage 预测相关性为正，且随训练提高；
2. 正标签平均概率高于负标签，概率差不再接近零；
3. 验证集自由切换比例处于 3%--15%，不存在全 Hold 或高频切换；
4. Controller 验证收益高于同一 frozen Outer+Inner checkpoint；
5. 三轮训练中至少一轮产生可复现的正收益增益，而不是仅依赖强制 30 日切换。
