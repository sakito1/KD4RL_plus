# Controller 双分支联合训练实施方案

## 目标

将当前由单一共享表示和固定优势修正公式组成的 Controller，改为：

1. 资产历史只经过一套共享时序编码器；
2. 风险分支仅使用当前持仓、持仓状态和共享时序信号，预测未来持有风险；
3. 优势分支使用当前持仓、候选组合及共享时序信号，预测扣除即时交易成本后的 switch--hold 相对优势；
4. 两个分支的隐向量拼接后由可学习 MLP 输出最终切换概率；
5. 风险监督、优势监督和轨迹策略梯度在同一次更新中联合反传，不截断任一分支到最终策略的梯度。

## 实施步骤

### 1. 用测试约束结构和梯度

- 新增 `tests/test_controller_dual_branch.py`。
- 检查两个分支隐向量及预测输出的形状。
- 固定当前持仓而改变候选组合，风险隐向量应保持不变，优势隐向量应变化。
- 仅对最终 `policy_logit` 反传时，风险分支、优势分支和共享 LSTM 均应获得非零梯度。

### 2. 重构 `MonitorAC`

- 保留现有 LSTM 与两级时序注意力，避免改变输入数据接口。
- 构造当前持仓风险特征：
  `portfolio_last + portfolio_ctx + holding_state`。
- 构造候选相对优势特征：
  `portfolio_last + portfolio_ctx + candidate-current temporal differences + action_state`。
- 分别通过 `risk_mlp` 和 `advantage_mlp` 得到两个分支隐向量。
- 使用 `switch_mlp([risk_embedding, advantage_embedding])` 直接产生切换 logit。
- 风险预测头只连接风险分支，优势预测头只连接优势分支。
- 保留原有返回字段，以兼容训练、验证和测试代码。

### 3. 保持训练目标语义一致

- `hold_risk_pred` 继续使用未来 hold 路径最大回撤的 Smooth-L1 监督。
- `switch_advantage_pred` 使用连续的、成本调整后的 switch--hold 收益差进行 Smooth-L1 监督。
- 最终切换概率只由可学习融合网络产生，不再使用
  `coef * tanh(pred / scale)` 的固定人工融合。
- 轨迹策略梯度通过融合网络同时更新两个分支及共享时序编码器。

### 4. 验证

- 运行新增结构与梯度测试。
- 运行 Controller 相关既有测试。
- 对修改文件执行 Python 语法检查。

## 兼容说明

旧的 `controller_switch_adv_logit_*` 参数暂时保留在命令行和构造接口中，避免历史脚本立即报错，但不再参与最终切换概率计算。当前正在运行的训练进程已经载入旧模型定义，必须重新启动训练才会使用新结构。
