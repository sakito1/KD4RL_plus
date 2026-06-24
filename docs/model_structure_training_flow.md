# HRL + Controller 模型结构与训练流程说明

本文是一份面向论文写作的实现说明，用来解释当前 KD4RL_plus 中 HRL + Controller 的模型结构、动作生成方式、训练集组织、训练链路、loss 设计和验证测试协议。它的目标不是重新提出一个抽象算法，而是把当前代码中真实执行的流程写清楚，作为论文方法部分、实验设置部分和消融解释部分的参考依据。

对应的主要实现位置如下：

- `Components/PPO_model.py`: Outer、Inner、Controller/Monitor 的神经网络结构。
- `agent/PPO_agent.py`: 三个模块的动作组合、PPO 更新、buffer 和多模块冻结/解冻。
- `env/PPO_env.py`: observation 构造、episode 切分、交易执行、交易成本、reward 和辅助 target。
- `Train/PPO_train.py`: fixed HRL warmup、joint finetune、controller counterfactual policy gradient。
- `Train/controller_pg.py`: controller 反事实 reward 和 policy-gradient loss。
- `run_hrl_training.py`: 训练入口参数、市场/种子循环、checkpoint 管理和不同训练模式。
- `train_sh/run_hrl_fixed60_inner_noaux_retrain.sh`: 训练 fixed HRL 的复现实验入口。
- `train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh`: 在 frozen HRL 上训练 controller 的入口。
- `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`: HRL warmup + controller PG + controller-active joint 的完整训练入口。

## 0. 写作口径

论文里可以把该方法描述为一个三层层级强化学习框架：

```text
Outer policy:        switch 日生成新的基准投资组合
Controller policy:   每个可决策日判断是否退出当前持仓段
Inner policy:        每日围绕基准组合做连续权重微调
```

更具体地说，系统不是让一个网络直接输出最终股票权重，而是把交易决策拆成三个相互配合的层次：

1. **配置层 / Outer**: 当系统决定切仓时，Outer 生成新的 top-K 基准组合。
2. **时机层 / Controller**: 判断今天继续持有还是切换到 Outer 给出的新组合。
3. **执行层 / Inner**: 在当天选定的基准组合内部进行连续调权，得到最终执行权重。

因此，论文里可以强调两个核心设计点：

- 第一，持仓时机不是固定规则，而是通过 counterfactual policy gradient 训练出来的 daily exit policy。
- 第二，controller 的 switch 信号不是单独学一个辅助头后闲置，而是将 `switch_advantage_pred` 接入 `exit_logit`，让“切仓相对持有更优”的判断能够直接影响最终 switch 概率。

## 1. 符号和变量

下面的符号用于把代码里的变量转成论文里的统一表述。

```text
t                 当前交易日
N                 股票数量
K                 Outer 选择的 top-K 股票数，代码中为 trade_num
T_o               Outer lookback window，当前好配置为 60
T_i               Inner lookback window，默认来自 config，通常为 10
H                 固定持仓周期 / max_hold，当前好配置为 30
w_t^drift         昨日实际持仓经价格变动后的漂移权重，对应 weights_drift
b_t^drift         当前基准组合经价格变动后的漂移权重，对应 base_drift
b_t               当天选定的基准组合，对应 base_used
w_t               当天最终执行权重，对应 weights_exec
u_t               Outer 生成的候选 switch 组合，对应 act_out
z_t,h_t,p_t       SSM 状态，对应 obs["ssm"]
s_t^port          当前持仓段状态，对应 port_state
a_t^c             Controller 动作，0=hold, 1=switch
pi_c              Controller 的 switch probability，对应 exit_prob
```

环境中的漂移权重按如下方式理解：

```text
w_t^drift = normalize(w_{t-1} * r_{t-1 -> t})
b_t^drift = normalize(b_{t-1} * r_{t-1 -> t})
```

这里的 `*` 是逐资产相乘，`normalize` 表示归一化到权重和为 1。代码对应 `env/PPO_env.py::_get_observation()`。

## 2. 数据、状态和 episode

### 2.1 市场划分

环境从配置文件中读取训练集、验证集和测试集日期范围。当前项目里有 NAS 和 SH 两套配置：

```text
NAS:
  train: 2000-04-07 到 2017-12-29
  val:   2018-01-02 到 2020-04-22
  test:  2020-04-23 到 2025-10-03

SH:
  train: 2000-04-07 到 2017-12-28
  val:   2018-01-02 到 2019-12-31
  test:  2020-01-02 到 2025-02-28
```

训练和验证/测试的 episode 组织不同：

- 训练时通常使用固定窗口池，从训练区间构造多个固定长度 episode。
- 验证和测试时按时间顺序从验证/测试起点跑到区间终点。

### 2.2 每日 observation

环境每天返回一个 observation，主要包含：

- `outer_state`: `[1, N, T_o, F]`，每只股票过去 `outer_window` 天的特征窗口，经 rolling z-score 标准化。
- `inner_state`: `[1, N, T_i, F]`，每只股票短窗口特征，给 Inner 做每日调权。
- `ssm`: 包括 `z, h, p, q_bear, q_bull`。
- `weights_drift`: `[1, N]`，实际持仓漂移权重。
- `base_drift`: `[1, N]`，当前基准组合漂移权重。
- `port_state`: `[1, 6]`，当前持仓段状态。
- `held_p`: 当前持仓组合加权后的 SSM `p` 值，用于规则分析和诊断。

`port_state` 的 6 个元素是：

```text
time_norm          当前持仓天数 / max_hold
drawdown           当前组合相对历史峰值的回撤
seg_return         当前持仓段内收益
cumulative_alpha   持仓段内累计 inner alpha
cumulative_risk    持仓段内累计 downside risk
cost_feat          交易成本特征
```

Controller 实际在 `MonitorAC._state_features()` 中使用其中一部分构造更紧凑的 5 维状态：

```text
time_norm
remaining_norm = 1 - time_norm
seg_return_norm
drawdown_norm
concentration = sum_i weights_i^2
```

### 2.3 价格、交易成本和组合收益

环境执行一天交易时，先根据昨日持仓和过去一天价格变化得到漂移组合，再根据当天最终执行权重计算换手和交易成本：

```text
turnover_t = sum_i |w_t[i] - w_t^drift[i]|
cost_t = value_t * turnover_t * transaction_cost_pct
```

当前默认交易成本为：

```text
TRANSACTION_COST_RATE = 5e-5
```

当日组合 log return 由最终权重和下一日价格比计算，并扣除交易成本影响。代码对应 `env/PPO_env.py::step()`。

## 3. 整体模型结构

整体结构可以写成：

```text
observation_t
  -> Outer: 生成候选 switch 组合 u_t
  -> Controller: 生成 a_t^c in {hold, switch}
  -> base selector:
       if a_t^c = switch: b_t = u_t
       if a_t^c = hold:   b_t = b_t^drift
  -> Inner: 基于 b_t 生成最终执行权重 w_t
  -> Env: 执行 w_t，得到 portfolio return 和训练信号
```

这里三层模块共享一个环境，但优化目标不同：

- Outer 学习“切仓时买什么”。
- Controller 学习“什么时候切仓”。
- Inner 学习“当前股票池内怎么调权”。

从论文叙述上，Outer 和 Inner 构成 fixed HRL backbone；Controller 则是在该 backbone 上学习动态持仓段退出策略。

## 4. Outer actor-critic

### 4.1 输入和编码

Outer 的输入是：

```text
outer_state:    [B, N, T_o, F]
weights_drift:  [B, N]
```

网络结构：

```text
AssetLSTMATTN:
  每只股票独立 LSTM 编码时间窗口
  HiddenAttn 对 LSTM 所有时间步做注意力池化

CAAN:
  对 N 只股票的 embedding 做 cross-asset attention

Fusion:
  concat(CAAN embedding, last_day_feature, weights_drift)
  -> MLP -> per-asset feature
```

代码对应 `OuterAC.encode()`。

### 4.2 Outer 动作生成

Outer actor 对每只股票输出一个 Normal 分布：

```text
x_i ~ Normal(mu_i, sigma_i)
score_i = tanh(x_i)
```

然后取 top-K 分数对应的股票，并对 top-K 分数做 softmax：

```text
I_K = TopK(score)
u_t[i] = softmax(score_i for i in I_K), if i in I_K
u_t[i] = 0, otherwise
```

其中 `u_t` 是候选 switch 组合，也就是代码中的 `act_out`。这不是最终执行权重；只有当 Controller 决定 switch 时，`u_t` 才会成为当天 `base_used`。

Outer 的 log-prob 基于采样前的连续变量 `raw_action` 计算，并包含 `tanh` 变换的 log-Jacobian 校正。执行组合中 top-K 操作本身不是通过环境收益直接反传的；PPO 更新使用采样动作的 log-prob ratio。

### 4.3 Outer value 和辅助预测

Outer critic 做法：

```text
per-asset feature
  -> market attention pooling
  -> concat(current weights representation)
  -> value head
```

Outer 辅助头 `pred_stock_return` 预测每只股票未来 `max_hold` 内的累计 log return。target 来自：

```text
env._future_stock_return_target(start_day=t, horizon=max_hold)
```

该辅助损失帮助 Outer 的 per-asset representation 更贴近未来收益排序。

## 5. Controller / Monitor

### 5.1 Controller 的输入

Controller 接收的信息比 Outer 更偏向“是否应该退出当前持仓段”：

```text
SSM state:       z, h, p, q_bear, q_bull
current weights: weights_drift
portfolio state: port_state
candidate action: switch_action = Outer candidate u_t
asset sequence: outer_state 最近 controller_window 天
```

当前好配置：

```text
CONTROLLER_WINDOW = 30
CONTROLLER_HIDDEN_DIM = 64
CONTROLLER_INIT_EXIT_BIAS = -1.0
CONTROLLER_EVAL_SWITCH_THRESHOLD = 0.5
```

初始 exit bias 为负值，意味着训练初期 switch 概率偏低，有利于避免一开始过度切仓。

### 5.2 Controller 的状态构造

Controller 先编码资产最近窗口：

```text
asset_state [B, N, T_c, F]
  -> one-layer LSTM
  -> asset_seq [B, N, T_c, H]
```

然后构造 hold 权重和 switch 权重：

```text
hold_weights   = normalize(weights_drift)
switch_weights = normalize(switch_action)
```

随后计算：

```text
portfolio_last = sum_i hold_weights_i   * last_emb_i
switch_last    = sum_i switch_weights_i * last_emb_i
```

再通过两层 temporal attention 得到：

```text
portfolio_ctx = sum_i hold_weights_i   * asset_ctx_i
switch_ctx    = sum_i switch_weights_i * asset_ctx_i
```

Controller 还显式构造了候选切仓动作与当前持仓的关系：

```text
turnover              = sum_i |switch_weights_i - hold_weights_i|
switch_concentration  = sum_i switch_weights_i^2
overlap               = sum_i min(switch_weights_i, hold_weights_i)
```

这些信息拼接后进入 head MLP：

```text
value_feat = concat(
  portfolio_last,
  portfolio_ctx,
  switch_last - portfolio_last,
  switch_ctx - portfolio_ctx,
  state_features,
  action_state_features
)
```

这个设计的含义是：Controller 不是只看“当前持仓好不好”，也看“Outer 给出的候选新组合和当前组合差在哪里”。

### 5.3 Controller 输出头

Controller 输出如下变量：

```text
base_exit_logit
switch_advantage_pred
exit_logit
exit_prob = sigmoid(exit_logit)
hold_return_pred
hold_risk_pred
value
```

其中 `exit_logit` 是最终用于策略分布的 logit：

```text
exit_logit = base_exit_logit
           + controller_switch_adv_logit_coef
             * tanh(switch_advantage_pred / controller_switch_adv_logit_scale)
```

当前好配置：

```text
controller_switch_adv_logit_coef = 1.9
controller_switch_adv_logit_scale = 0.02
controller_switch_adv_logit_detach = 1
```

`detach=1` 的含义是：PG 主损失不会通过这条 logit 加法支路反向更新 `switch_advantage_pred` head；该 head 主要由辅助 loss 学习。但在 forward 时，它的输出仍然会影响 `exit_logit`，进而影响 `exit_prob`。

这一点可以作为论文里的关键实现细节：辅助信号不只是用于 representation learning，而是作为策略 logit 的结构化调制项。

### 5.4 Controller 动作生成

Controller 的动作空间是二元离散动作：

```text
a_t^c = 0: hold
a_t^c = 1: switch
```

策略分布为：

```text
pi(a_t^c | s_t) = Categorical([0, exit_logit])
```

训练时：

```text
a_t^c ~ pi(a_t^c | s_t)
```

评估和测试时：

```text
a_t^c = 1 if exit_prob > 0.5 else 0
```

需要注意：Controller 的 switch/hold 是离散动作，不是通过环境收益对动作本身求导。它遵循标准 policy gradient 的 score-function estimator：

```text
gradient proportional to grad log pi(a_t^c | s_t) * reward
```

因此，Controller 不要求环境转移或 switch 动作本身可导。可训练性来自动作 log-prob 和 episode-level reward。

## 6. Inner actor-critic

### 6.1 输入和编码

Inner 的输入是：

```text
inner_state:    [B, N, T_i, F]
base_used:      [B, N]
weight_drift:   [B, N]
```

如果启用 `inner_use_topk`，Inner 会只在当前 `base_used` 的 top-K 股票上运行；当前主配置中 Inner 的最终执行权重仍被 scatter 回完整股票池。

网络结构：

```text
per-asset LSTM
  -> temporal attention layer 1
  -> temporal attention layer 2
  -> concat(node_feat, base_used_i, weight_drift_i)
  -> MLP fusion
  -> actor / critic / pred heads
```

代码对应 `InnerAC.encode()`。

### 6.2 Inner 动作生成

当前实现中 Inner 不输出离散买卖动作，而是输出连续 score：

```text
y_i ~ Normal(mu_i, sigma_i)
```

然后只在当前 base 中非零权重的资产上做 masked softmax：

```text
mask_i = 1 if base_used_i > 0 else 0
target_i = softmax(y_i over mask)
```

最终执行权重是 base 和 target 的凸组合：

```text
w_t = (1 - alpha) * b_t + alpha * target
```

其中 `alpha = inner_max_boundary`。当前脚本通过 `inner_max_boundary` 控制 Inner 每天可以偏离 base 的程度。

这个设计让 Inner 的行为更稳定：

- 不扩大 Outer 选出的股票池。
- 不直接输出任意全市场组合。
- 只在当前基准组合内部调节权重。

### 6.3 Inner value 和辅助预测

Inner critic 使用两种聚合：

```text
base-weighted representation
attention-pooled representation
```

然后拼接后输出 value。Inner 辅助头预测每只股票下一日收益：

```text
inner_stock_return_target = env._future_stock_return_target(t, horizon=1)
```

Inner reward 是实际执行组合相对 base 组合的超额 log return：

```text
inner_reward = (portfolio_return - base_return) * reward_scale_inner
```

因此 Inner 学到的是：在 Outer/Controller 已经决定股票池和持仓段后，如何通过日频调权贡献 alpha。

## 7. 每日动作链路

每日动作链路在 `agent/PPO_agent.py::get_action()` 中完成。

### 7.1 强制决策逻辑

先由 trainer 判断当天是否被强制 hold/switch：

```text
第 0 天: 必须 switch，用于建仓
fixed HRL 模式: 未到固定周期 hold，到固定周期 switch
controller 模式: 未被强制时交给 Controller
```

对于当前好用 controller 配置：

```text
controller_no_hold_constraints = True
controller_decision_mode = daily
controller_train_max_hold = 0
controller_eval_max_hold = -1
max_hold = 30
```

含义是：

- 训练 controller PG 时，不再用 min-hold 限制自由决策。
- `controller_train_max_hold=0` 表示训练 rollout 内基本关闭 forced max-hold，让 controller 自己探索切仓。
- 评估和测试时 `controller_eval_max_hold=-1` 回到全局 `max_hold=30`，保留最大持仓期保护。
- switch 次数的主要约束由 reward 里的 `controller_max_switches` 和 overflow penalty 控制。

### 7.2 动作组合步骤

每天的动作组合如下：

1. 判断 `force_switch`。
2. 如果需要候选 switch 动作，则 Outer 生成 `act_out = u_t`。
3. 如果 `force_switch is None`，Controller 根据当前状态和 `act_out` 采样/判断 hold 或 switch。
4. 选择当天 base：

```text
if switch:
    base_used = act_out
else:
    base_used = base_drift
```

5. Inner 根据 `base_used` 生成最终执行权重 `weights_exec`。
6. 环境执行 `weights_exec`，并记录 reward、target、switch event 和 portfolio value。

### 7.3 Fixed HRL 与 Controller HRL 的区别

Fixed HRL：

```text
Controller 不参与
每 30 天强制 switch
Outer 只在 switch 日生成新组合
Inner 每日调权
```

Controller HRL：

```text
Controller daily 判断 hold/switch
switch 时采用 Outer 候选新组合
hold 时沿用 base_drift
Inner 每日调权
测试时仍保留 max_hold=30 的强制保护
```

因此，Controller 改变的是持仓段边界，而不是直接替代 Outer 或 Inner。

## 8. 训练集和 episode 组织

### 8.1 固定窗口池

Inner 和 Controller 都使用固定窗口池组织训练 episode。核心函数是 `env._build_fixed_train_pool()`：

```text
输入:
  raw_indices: 训练区间所有交易日 index
  episode_len: 每个 episode 的长度
  stride_days: offset 间隔
  start_offsets: 从训练区间开头取多少个候选 offset

过程:
  1. 取 raw_indices 前 start_offsets 个点作为候选起点区域
  2. 按 stride_days 取 offset: 0, stride, 2*stride, ...
  3. 每个 offset 起点向后按 episode_len 连续切 episode
```

用例子说明 controller 当前配置：

```text
episode_len = 600
start_offsets = 30
stride_days = 5
```

候选 offset 为：

```text
0, 5, 10, 15, 20, 25
```

每个 offset 都从训练区间对应起点开始，按 600 天一段往后切 episode。

如果设置：

```text
controller_fixed_pool_limit = 12
```

则从完整 episode pool 中均匀抽取 12 个窗口。因此当前 controller 训练中，一个 epoch 使用 12 个 600 天 episode。

### 8.2 Inner 训练集组织

当前 HRL 复现脚本默认：

```text
INNER_EPISODE_LEN = MAX_HOLD * INNER_SEGMENTS_PER_EPISODE
                  = 30 * 20
                  = 600

INNER_TRAIN_EPISODES_PER_EPOCH = 30
INNER_START_STRIDE_DAYS = 1
INNER_EPISODE_BATCH_SIZE = 12
INNER_EPISODE_PARALLEL_WORKERS = 12
INNER_ROLLOUT_UPDATE_STEPS = 600
INNER_PPO_EPOCHS = 1
```

也就是：

- 一个 Inner epoch 有 30 个 episode。
- 一个 episode 有 600 个交易日。
- 每次最多并行跑 12 个 episode。
- 12 个 episode 的 buffer 合并后做一次 Inner PPO update。

实现位置：

- `Train/PPO_train.py::_reserve_train_episode_windows()`
- `Train/PPO_train.py::_run_inner_episode_worker()`
- `Train/PPO_train.py::_run_inner_episode_batch()`

### 8.3 Controller 训练集组织

当前好用 controller 配置：

```text
CONTROLLER_ROLLOUT_LEN = 600
CONTROLLER_WINDOWS_PER_EPOCH = 30
CONTROLLER_START_STRIDE_DAYS = 5
CONTROLLER_FIXED_POOL_LIMIT = 12
CONTROLLER_EPISODE_BATCH_SIZE = 12
CONTROLLER_EPISODE_PARALLEL_WORKERS = 12
```

所以 controller 每个 epoch 使用 12 个固定窗口 episode。每个 batch 里并行跑 12 个窗口。每个窗口都跑一组反事实对照：

```text
baseline:   fixed HRL，固定 30 天切仓
controlled: HRL + controller，由 controller daily 判断是否 switch
```

这就是 counterfactual PG 的训练样本。它不是一个“单日有监督标签数据集”，而是 episode-level counterfactual dataset。

### 8.4 一个 controller batch 里发生什么

对每个窗口 `(start, stop)`：

1. 拷贝环境，跑 fixed HRL baseline。
2. 再拷贝环境，跑 controller-controlled episode。
3. controlled episode 中，记录所有自由决策日的：
   - state
   - candidate switch action
   - sampled hold/switch action
   - log-prob 所需信息
   - 辅助 target
4. episode 结束后，计算 controlled 相对 baseline 的 reward。
5. 12 个 episode 的 loss 累积后，做一次 optimizer step。

该设计与 Inner 的 batch 思路一致：都是并行跑多个固定窗口 episode，再合并为一次更新。

## 9. 训练流程

### 9.1 Fixed HRL 训练

入口：

```bash
bash train_sh/run_hrl_fixed60_inner_noaux_retrain.sh
```

默认关键参数：

```text
OUTER_WINDOW = 60
MIN_HOLD = 30
MAX_HOLD = 30
WARMUP_OUTER_EPOCHS = 2
WARMUP_INNER_EPOCHS = 2
INNER_EPISODE_LEN = 600
INNER_TRAIN_EPISODES_PER_EPOCH = 30
INNER_EPISODE_BATCH_SIZE = 12
JOINT_LR_MULT = 0.001
OUTER_PRED_COEF = 0.1
INNER_PRED_COEF = 0.05
INNER_PRED_TARGET_SCALE = 10
```

训练阶段：

```text
Stage 1: Warmup Outer
  - 固定 30 天切仓
  - 只更新 Outer
  - 保存 temp_warmup_outer.pth

Stage 2: Warmup Inner
  - 使用固定 600 天窗口池
  - 固定 30 天切仓
  - 并行 episode batch
  - 只更新 Inner
  - 保存 temp_warmup_inner.pth

Stage 3: Outer + Inner joint
  - Controller 不参与
  - Outer 和 Inner 联合微调
  - 保存 hrl_fixed_best.pth
```

Fixed HRL 的作用是先得到稳定 backbone：

- Outer 学会固定切仓时选择哪些股票。
- Inner 学会在 Outer 选出的 top-K 内进行日频调权。

### 9.2 Controller-only 训练

入口：

```bash
bash train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh
```

该脚本加载已有 fixed HRL：

```text
SOURCE_ROOT/.../{market}/ppo/seed_{seed}/checkpoints/hrl_fixed_best.pth
```

然后：

```text
冻结 Outer 和 Inner
只训练 Controller/Monitor
使用 counterfactual policy gradient
```

当前好用配置：

```text
CONTROLLER_EPOCHS = 3
CONTROLLER_ROLLOUT_LEN = 600
CONTROLLER_EPISODE_BATCH_SIZE = 12
CONTROLLER_FIXED_POOL_LIMIT = 12
CONTROLLER_PG_LOGPROB_REDUCTION = sum
CONTROLLER_RETURN_COEF = 1.0
CONTROLLER_MAX_SWITCHES = 30
CONTROLLER_MAX_SWITCH_PENALTY_COEF = 0.001
CONTROLLER_VALUE_COEF = 0.0
CONTROLLER_ENTROPY_COEF = 0.0
CONTROLLER_VALUE_NORMALIZE_ADVANTAGE = 0
```

这表示：

- 每个 epoch 使用 12 个 600 天 episode。
- 一个 batch 是 12 个 episode。
- reward 不做 batch normalization。
- 不使用 value baseline。
- 不额外加 entropy bonus。
- 主优化信号是 raw counterfactual reward。

### 9.3 Controller 辅助预训练

脚本里有：

```text
CONTROLLER_SUP_PRETRAIN_EPOCHS = 1
CONTROLLER_SUP_PRETRAIN_ROLLOUT_LEN = 240
CONTROLLER_AUX_PRETRAIN_OFFPOLICY = 1
CONTROLLER_AUX_REPLAY_EPOCHS = 3
```

这个阶段虽然变量名里有 `SUP`，但当前好配置里：

```text
CONTROLLER_SUP_COEF = 0.0
```

因此它不是用最终 switch label 去监督 `exit_logit`。它主要用于辅助头：

- `hold_return_pred`
- `hold_risk_pred`
- `switch_advantage_pred`

尤其是 `switch_advantage_pred`，会通过 weighted BCE 学习 switch advantage 的正负和强度。

### 9.4 完整端到端训练

入口：

```bash
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

流程：

```text
HRL warmup outer
  -> HRL warmup inner
  -> fixed HRL outer+inner joint
  -> controller counterfactual PG
  -> controller-active HRL joint finetune
```

最后一步 joint finetune：

```text
CONTROLLER_JOINT_EPOCHS = 1
JOINT_LR_MULT = 0.0001
```

这一步同时解冻 Controller、Outer、Inner，但学习率极低。论文里可以解释为：

> After learning the exit policy on top of a fixed HRL backbone, we optionally perform a low-learning-rate joint fine-tuning stage to align the controller, allocator, and executor without overwriting the learned fixed-HRL behavior.

注意：当前论文展示时不需要把 `best_model` checkpoint 单独命名成一个新方法。更清楚的做法是把最终启用 controller 的策略统一称为 `Full Controller` 或 `Controller-PG`，避免把 checkpoint 名误解成一个新方法。

## 10. Loss 和 reward

### 10.1 PPO 基础形式

Outer、Inner 和普通 joint 阶段使用 PPO-style clipped objective。形式上可以写为：

```text
L_policy = - E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
r_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))
```

代码中 advantage 对 Inner 和 Outer 的 PPO 更新会做标准化，这是常见 PPO 技巧。但 Controller PG 的 counterfactual reward 在当前好配置中不做 batch normalization。

### 10.2 Outer loss

Outer 只在 switch 日更新。总体 loss：

```text
L_outer =
  L_outer_policy
  + vf_coef * L_outer_value
  + outer_pred_coef * L_outer_pred
  - ent_coef * H_outer
```

其中：

- `L_outer_policy`: PPO clipped policy loss。
- `L_outer_value`: switch segment value loss。
- `L_outer_pred`: 每只股票未来 `max_hold` 累计收益预测 SmoothL1 loss。
- `H_outer`: action distribution entropy。

Outer 的 return 是 segment-level 的。buffer 在 switch boundary 上把这一段的 daily outer reward 聚合给该 switch 决策。

### 10.3 Inner loss

Inner 每天更新。总体 loss：

```text
L_inner =
  L_inner_policy
  + vf_coef * L_inner_value
  + inner_pred_coef * L_inner_pred
  - ent_coef * H_inner
```

Inner reward：

```text
r_inner,t = (portfolio_return_t - base_return_t) * reward_scale_inner
```

其中：

- `portfolio_return_t`: 使用 Inner 最终执行权重后的组合 log return。
- `base_return_t`: 使用 base 权重的组合 log return。

这让 Inner 直接优化相对 base 的超额收益。

### 10.4 Controller counterfactual reward

Controller 的主 reward 来自反事实对照：

```text
baseline   = fixed HRL on the same window
controlled = HRL + controller on the same window
```

episode reward：

```text
return_uplift = controlled.log_return - baseline.log_return
overflow = max(0, controlled.segment_count - max_switches)
overflow_penalty = overflow^2 / max_switches^2

R_controller =
  controller_return_coef * return_uplift
  - controller_max_switch_penalty_coef * overflow_penalty
```

当前好配置：

```text
controller_return_coef = 1.0
controller_max_switches = 30
controller_max_switch_penalty_coef = 0.001
```

注意几个排除项：

- 当前好配置不使用 MDD reward 项。
- 当前好配置不使用 turnover reward 项。
- 当前好配置不使用 minimum switch count penalty。
- 当前好配置不对 batch reward 做 normalization。

这是为了让 controller 训练保持标准 policy-gradient 逻辑：每个 episode 的 raw counterfactual reward 决定该 episode 中采样动作的更新方向。

### 10.5 Controller PG loss

当前好配置：

```text
controller_pg_logprob_reduction = sum
```

因此一个 episode 的 log-prob 是所有自由决策日 log-prob 之和：

```text
LogP_episode = sum_t log pi_c(a_t^c | s_t)
```

如果不使用 value baseline：

```text
A_episode = R_controller
L_pg = - A_episode * LogP_episode
```

如果 batch 里有 12 个 episode，则：

```text
L_batch = mean_j L_pg,j
```

当前好配置中：

```text
controller_value_coef = 0
controller_value_normalize_advantage = 0
controller_entropy_coef = 0
```

所以 Controller 的主项非常干净：

```text
L_controller_main = - R_controller * sum_t log pi_c(a_t^c | s_t)
```

### 10.6 Controller 辅助 target

环境为 Controller 提供三个重要辅助 target。

#### hold_return target

继续持有当前漂移组合到剩余 max-hold 的未来 log return：

```text
hold_return_target =
  future_log_return(current_holdings_drift, start_day=t, horizon=max_hold - t_held)
```

对应 `controller_hold_return_target`。

#### hold_risk target

继续持有当前漂移组合到剩余 max-hold 的未来最大回撤：

```text
hold_mdd_target =
  future_max_drawdown(current_holdings_drift, start_day=t, horizon=max_hold - t_held)
```

对应 `controller_hold_mdd_target`。

#### switch_advantage target

候选 switch 组合相对继续持有组合的未来收益优势，扣除切仓交易成本：

```text
switch_advantage =
  future_log_return(candidate_switch_weight)
  - future_log_return(current_holdings_drift)
  - turnover_to_candidate * transaction_cost_pct
```

对应 `controller_switch_advantage`。

这个 target 是当前 controller 成功的关键，因为它直接描述了“如果今天切仓，相对继续持有是否更好”。

### 10.7 Controller 辅助 loss

当前好配置：

```text
controller_aux_return_coef = 0.1
controller_aux_mdd_coef = 0.1
controller_aux_switch_adv_coef = 1.0
controller_aux_switch_adv_loss_type = weighted_bce
```

辅助 loss 包括：

```text
L_aux_return = SmoothL1(hold_return_pred, hold_return_target)
L_aux_mdd    = SmoothL1(hold_risk_pred, hold_mdd_target)
L_aux_sw     = weighted BCE on sign/sized switch_advantage
```

总的 Controller 更新可写成：

```text
L_controller =
  L_pg
  + aux_return_coef * L_aux_return
  + aux_mdd_coef * L_aux_mdd
  + aux_switch_adv_coef * L_aux_sw
```

当前好配置中不使用：

```text
supervised BCE on exit_logit
local_adv_loss
expected_switch_loss
overflow_action_loss
value_loss
entropy_loss
```

需要特别说明：`weighted_bce` 是作用在 `switch_advantage_pred` 这个辅助头上的，不是用一个人工 switch label 直接监督最终 `exit_logit`。最终 switch 行为仍由 counterfactual PG 主目标训练。

## 11. 为什么 switch_advantage_pred 接到 exit_logit 很关键

早期 controller 效果不明显的原因之一是：辅助头即使学到了“这一天 switch 可能更好”，这个信号也不一定能推动最终 `exit_prob` 超过 0.5。于是模型可能出现：

```text
辅助头有信息
但 exit head 不动
最终策略仍然 hold
```

现在的结构把 `switch_advantage_pred` 直接接入 `exit_logit`：

```text
exit_logit = base_exit_logit + f(switch_advantage_pred)
```

这样当辅助头判断 switch advantage 为正时，会直接提高 switch logit。论文里可以把它描述为：

> We use the predicted switch advantage as a logit-level modulation term for the exit policy, so that the learned local counterfactual signal can directly influence the binary switching probability.

这不是把规则硬编码进去，因为：

- `switch_advantage_pred` 是由模型从状态中预测的。
- 它的 target 来自环境中的未来反事实收益估计。
- 最终策略仍然由 PG reward 决定采样动作的长期更新方向。

## 12. 验证和测试协议

### 12.1 验证集

验证时环境切到 `val`，以 eval 模式运行。主要模式：

```text
Fixed HRL:
  fixed_cycle = 30
  controller disabled

Controller:
  fixed_cycle = None
  controller enabled
  action = switch if exit_prob > 0.5 else hold
```

checkpoint 选择：

- Fixed HRL 阶段默认根据 Sharpe 或脚本指定指标选最优。
- Inner warmup 当前脚本使用 return 作为选择指标。
- Controller 当前好配置使用 return 作为选择指标。
- Controller-active joint 也按 controller 选择指标验证。

### 12.2 测试集

测试时环境切到 `test`，从测试起点顺序跑到测试终点。输出包括：

- `test_s3_AllModules.csv`: 最终模型组合价值。
- `*_portfolio.csv`: 每日组合价值、收益、drawdown、inner alpha。
- `*_actions.csv`: 每日 action、switch/hold、exit_prob、switch advantage 等。
- `*_switch_events.csv`: switch event 列表。
- `*_metrics.json`: 总收益、Sharpe、MDD、switch count 等指标。

论文图表当前主要使用：

```text
paper_experiments_outputs/end_to_end_explain/figures
paper_experiments_outputs/end_to_end_explain/tables
```

当前图表展示中不再把 `best_model` checkpoint 单独命名成一个新方法，避免读者把 checkpoint 名误解为额外模型。

## 13. 当前复现实验入口

### 13.1 只训练 fixed HRL

```bash
bash train_sh/run_hrl_fixed60_inner_noaux_retrain.sh
```

用途：

- 训练 Outer + Inner backbone。
- 生成 `hrl_fixed_best.pth`。
- Controller 不参与。

### 13.2 在 frozen HRL 上训练 controller

```bash
bash train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh
```

用途：

- 加载 fixed HRL。
- 冻结 Outer 和 Inner。
- 用 counterfactual PG 训练 Controller。
- 可选低学习率 joint。

### 13.3 完整端到端训练流程

```bash
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

用途：

- 从 HRL warmup 开始。
- 接 controller PG。
- 最后做低学习率 controller-active joint。
- 不覆盖归档好模型。

### 13.4 当前归档结果

当前归档好模型和结果整理在：

```text
results/end
```

复现实验说明：

```text
train_sh/README_hrl_controller_reproduce.md
```

论文图表输出：

```text
paper_experiments_outputs/end_to_end_explain
```

## 14. 论文中可以怎么表述

下面这段可以作为方法部分的基础表述：

> We formulate portfolio rebalancing as a hierarchical decision process with three coupled policies. The outer policy proposes a sparse top-K base portfolio at switching times. The controller policy decides whether to continue the current holding segment or switch to the newly proposed base portfolio. The inner policy then refines the selected base portfolio through continuous weight adjustment within the active asset set. This decomposition separates asset selection, switching timing, and execution-level allocation.

Controller 训练可以这样写：

> The controller is trained with counterfactual policy gradient. For each sampled training window, we run both a fixed-period HRL baseline and a controller-enabled rollout using the same frozen outer and inner policies. The episode reward is defined as the log-return improvement over the fixed-period baseline, penalized only when the number of holding segments exceeds a pre-specified switch budget. The controller update uses the sum of log-probabilities over all free switching decisions in the episode, without batch reward normalization.

switch advantage 辅助头可以这样写：

> To make local switching evidence directly actionable, the controller predicts a switch advantage, defined as the future return advantage of switching to the outer candidate portfolio over continuing the current holdings after transaction costs. This predicted advantage is used not only as an auxiliary learning target but also as a bounded additive modulation to the exit logit, allowing positive local switch evidence to increase the final switching probability.

Inner 可以这样写：

> The inner policy does not choose a new asset universe. Instead, it produces continuous scores over the currently active base portfolio, converts them into a masked softmax target, and forms the final executable allocation as a convex combination of the base portfolio and the target allocation. This design stabilizes daily execution while preserving the sparse asset selection made by the outer policy.

## 15. 实现与论文术语对应表

```text
论文术语                         代码变量 / 文件
------------------------------------------------------------
Outer policy                     net.outer / OuterAC
Inner policy                     net.inner / InnerAC
Controller / exit policy         net.mon / MonitorAC
candidate switch portfolio       act_out / switch_action
base portfolio                   base_used
drifted current weights          weights_drift
drifted base portfolio           base_drift
final executable weights         weights_exec
switch probability               exit_prob
switch logit                     exit_logit / policy_logit
hold return prediction           hold_return_pred
hold risk prediction             hold_risk_pred
switch advantage prediction      switch_advantage_pred
fixed HRL checkpoint             hrl_fixed_best.pth
controller checkpoint            controller_best.pth
full controller evaluation       full_controller scenario
```

## 16. 一句话总结

当前模型的核心是“先学稳定的固定周期 HRL backbone，再用反事实 policy gradient 学 daily switch controller”。Outer 决定切仓后买什么，Controller 决定今天是否切仓，Inner 决定在当前股票池里如何调权。Controller 的成功关键在于：用 controlled rollout 相对 fixed HRL baseline 的收益提升作为主 PG reward，同时把 `switch_advantage_pred` 接到 `exit_logit` 上，让局部反事实优势能够真正推动 `exit_prob` 超过 0.5 并触发有效切仓。
