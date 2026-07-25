# Controller 经济事件引导设计

## 目标

为双分支 Controller 提供方向明确但不过度重复的训练监督。风险分支判断当前
组合相对整个股票池是否恶化，优势分支判断 Manager Candidate 相对当前组合
是否具有成本调整后的收益优势，最终融合层学习何时执行 Switch。

本设计只使用训练集未来路径构造监督目标。验证和测试阶段只能使用决策时可见
特征，不读取风险、优势或事件标签。

## 决策窗口

Controller rollout 和辅助窗口均使用 300 个交易日。每个自由决策日的比较
期限为当前持仓距离 30 日最大持仓限制的剩余时间：

\[
H_t=H_{\max}-d_t,\qquad H_{\max}=30.
\]

Hold 和 Switch 分支使用相同起点、相同 \(H_t\) 和相同交易成本口径。

## 相对市场风险

以股票池内全部资产的等权买入持有路径作为市场基准。对未来
\(h\le H_t\)，定义：

\[
V_t^{\mathrm{hold}}(h)
=\sum_iw_{i,t}\prod_{k=1}^{h}(1+r_{i,t+k}),
\]

\[
V_t^{\mathrm{mkt}}(h)
=\frac{1}{N}\sum_i\prod_{k=1}^{h}(1+r_{i,t+k}),
\qquad
Q_t(h)=\frac{V_t^{\mathrm{hold}}(h)}{V_t^{\mathrm{mkt}}(h)}.
\]

风险监督目标为相对财富路径的最大回撤：

\[
D_t^{\mathrm{rel}}
=
\max_{h\le H_t}
\left[
1-\frac{Q_t(h)}{\max_{u\le h}Q_t(u)}
\right].
\]

风险分支输入共享时序表示、全股票池特征和当前组合信息，不使用 Candidate。

## Candidate 净优势

优势监督目标为：

\[
A_t
=R_t^{\mathrm{switch}}-R_t^{\mathrm{hold}}
-c\left(TO_t^{\mathrm{switch}}-TO_t^{\mathrm{hold}}\right),
\]

其中 \(c=0.0001\)，收益使用现有冻结路径对数收益口径。优势分支输入共享时序
表示、当前组合和 Candidate 比较信息。

## 5%经济触发规则

风险阈值和显著优势阈值均固定为 5%，两个市场使用相同规则：

\[
g_t=
\mathbb I\left[
\left(D_t^{\mathrm{rel}}\ge0.05\land A_t>0\right)
\lor
\left(A_t\ge0.05\right)
\right].
\]

第一项表示当前组合相对市场明显恶化且存在更好的 Candidate；第二项表示当前
组合未必恶化，但 Candidate 提供显著机会。阈值在验证和测试前锁定，不根据
后续结果调整。

## 连续事件标签

连续满足 \(g_t=1\) 的日期视为同一经济事件：

- 连续事件首日：\(y_t=1,m_t=1\)；
- 同一事件后续日期：\(m_t=0\)，不进入 Label loss；
- 非事件日期：\(y_t=0,m_t=1\)。

任一实际或强制 Switch 都会改变持仓状态并重置事件连续性。后续状态重新计算
风险和优势，因此可以形成新事件。

seed 77 的 12 个 300 日训练窗口产生 127 个独立事件，监督正标签比例为
3.97%。风险主导、优势主导和双重满足事件分别为 69、48 和 10 个。

## 类别平衡 Label loss

最终融合层输出 \(z_t\) 和 \(p_t=\sigma(z_t)\)。对未屏蔽样本分别计算：

\[
\mathcal L_{\mathrm{pos}}
=
\frac{1}{N_+}\sum_{t:m_ty_t=1}\operatorname{softplus}(-z_t),
\]

\[
\mathcal L_{\mathrm{neg}}
=
\frac{1}{N_-}\sum_{t:m_t=1,y_t=0}\operatorname{softplus}(z_t),
\]

\[
\mathcal L_{\mathrm{label}}
=\frac{1}{2}\mathcal L_{\mathrm{pos}}
+\frac{1}{2}\mathcal L_{\mathrm{neg}}.
\]

若一个训练批次只有一个有效类别，仅使用该类别均值。实现通过监督权重保证
正负类别各贡献总 Label loss 的一半。

## 最终目标

辅助预训练不使用策略梯度和熵奖励：

\[
\mathcal L_{\mathrm{pretrain}}
=
\mathcal L_{\mathrm{label}}
+0.1\mathcal L_{\mathrm{risk}}
+\mathcal L_{\mathrm{adv}}.
\]

Controller PG 阶段使用：

\[
\mathcal L_{\mathrm{final}}
=
\mathcal L_{\mathrm{PG}}
+0.1\mathcal L_{\mathrm{label}}
+0.1\mathcal L_{\mathrm{risk}}
+\mathcal L_{\mathrm{adv}}
-0.01\mathcal H.
\]

风险和优势项使用 Smooth-L1。Label loss 更新最终融合层、两个 embedding
分支和共享时序编码器；风险与优势回归分别更新对应分支和共享编码器；PG
负责根据轨迹收益校准最终切换行为。

## 验收标准

1. 相对风险使用全股票池等权买入持有路径，期限与 Hold/Switch 比较完全一致。
2. 风险事件必须同时满足 \(A_t>0\)，不会将更差 Candidate 标为 Switch。
3. 连续事件只有首日为正标签，后续日期监督权重为零。
4. 实际或强制 Switch 后事件连续性重置。
5. 正负类别同时存在时，各贡献 Label loss 的 50%。
6. 辅助预训练和 PG 阶段的 Label 系数分别为 1.0 和 0.1。
7. PG 阶段风险、优势和熵系数分别为 0.1、1.0 和 0.01。
8. 验证和测试阶段不构造或读取未来标签。
9. 日志分别报告 Label、风险、优势和熵的原始及加权损失。

