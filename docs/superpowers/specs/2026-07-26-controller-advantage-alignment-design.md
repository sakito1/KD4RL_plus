# Controller 优势监督对齐设计

## 目标

修正优势分支的输入与监督目标不一致问题。Controller 在训练和推理时均使用
冻结 Inner Actor 分别生成 Hold 与 Switch 的实际执行权重，并据此预测成本
调整后的相对优势。

## 方案选择

考虑过三种方案：

1. 只把 Smooth-L1 改成符号分类：改动最小，但仍保留执行权重错位。
2. 将优势目标改为 Outer base 之间的比较：输入与目标一致，但不再对应最终
   实际执行组合。
3. **采用方案**：向优势分支传入 `hold_exec`、`switch_exec` 和剩余持有期限，
   并使用按优势绝对值加权的符号分类。该方案保持监督目标的经济含义，同时
   消除主要的信息缺失。

## 数据流

每个自由决策日先由冻结 Outer Actor 产生 Candidate，再用冻结 Inner Actor
分别计算：

\[
\mathbf w_t^{\mathrm{hold}},\qquad
\mathbf w_t^{\mathrm{switch}}.
\]

Controller 的风险分支保持不变。优势分支使用两组实际执行权重构造当前组合
表示、Candidate 差分表示、换手率、集中度和重合度。为保持现有
Outer+Inner checkpoint 的 Controller 层形状可加载，剩余期限不扩展 MLP
输入维度，而是作为 Candidate 差分表示的门控信号：

\[
h_t=(H_{\max}-d_t)/H_{\max}.
\]

收益期限相关的两组 Candidate 差分表示乘以 \(h_t\)，静态换手率、集中度和
重合度保持不变。

优势目标仍为：

\[
A_t=R_t^{\mathrm{switch}}-R_t^{\mathrm{hold}}
-c(TO_t^{\mathrm{switch}}-TO_t^{\mathrm{hold}}).
\]

## 优势损失

优势 head 输出 logit \(q_t\)，分类标签为
\(a_t=\mathbb I[A_t>0]\)。单样本权重使用
\(\lvert A_t\rvert\)，并在批次内归一到均值 1：

\[
\mathcal L_{\mathrm{adv}}
=
\frac{\sum_t \bar w_t\,
\operatorname{BCEWithLogits}(q_t,a_t)}
{\sum_t\bar w_t}.
\]

零优势样本保留一个数值下限，避免整批权重为零。最终 switch label 的经济
规则与类别平衡方式本次不修改。

## 兼容性和验收

- `decision_stats` 的新参数均为可选；未提供时回退到现有权重，避免破坏已有
  测试和非 Controller 路径。
- 训练记录必须保存 Hold/Switch 实际执行权重与剩余期限。
- PG 重算 loss 时使用记录中的相同输入，不重新调用会变化的 Actor。
- 单元测试验证实际执行权重改变优势预测输入、期限进入优势分支、优势 loss
  使用符号标签和绝对值权重。
- 不执行 Git commit，由用户自行提交。
