# HRL Controller Final Plan

本文档整理当前最终版 HRL + embedding + controller 训练方案。核心目标是：

- 用趋势状态 embedding 提供股票级状态表征。
- 用 controller 判断是否触发 outer actor 重新选股。
- outer actor 直接最大化组合收益。
- inner actor 最大化相对 base portfolio 的超额收益，并放大该相对收益 reward。
- 所有收益型 reward 使用 log return，保证时间维度上可加。

## 1. 总体结构

最终系统由四个部分组成：

```text
Embedding Generator
    生成每只股票 t 时刻 final_emb_t 和 p_next_t

Outer Actor
    在 switch 时重新生成 base portfolio

Inner Actor
    在 base portfolio 内部做权重微调

Controller
    判断当前持仓是否继续 hold，还是 switch 后交给 outer actor 重新决策
```

职责划分：

```text
embedding:
    学股票状态，不直接交易

outer:
    决定买哪些股票，目标是组合收益最大化

inner:
    在 outer base 上微调权重，目标是相对 base 的超额收益

controller:
    决定什么时候重新调用 outer，目标是组合收益最大化
```

## 2. Embedding Generator

### 2.1 输入

每只股票在 t 时刻使用当前可见特征：

```text
features_t
```

这里的 t 时刻信息只能来自 t 及 t 之前，不能使用未来数据。

### 2.2 obs embedding

当前观测 embedding 使用 MLP 生成：

```text
obs_emb_t = MLP(features_t)
```

这部分只负责把当前时刻的特征压缩成一个股票状态向量。

### 2.3 alpha

alpha 是重要变量，表示趋势持续权重。alpha 不用普通 MLP 直接生成，而是用 LSTM-attention 结构生成，因为它需要根据过去一段序列判断趋势是否延续。

```text
alpha_t = LSTM_Attn_AlphaNet(features_{t-L+1:t})
```

alpha 的语义：

```text
alpha_t 越大：
    趋势大概率延续，final_emb_t 更接近 final_emb_{t-1}

alpha_t 越小：
    趋势可能发生切换，final_emb_t 更接近当前 obs_emb_t
```

建议第一版使用 scalar alpha：

```text
alpha_t shape = [B, 1]
```

### 2.4 final embedding

最终 embedding 由 t-1 时刻 final embedding 和 t 时刻 obs embedding 通过 alpha 加权得到：

```text
final_emb_t =
    alpha_t * final_emb_{t-1}
    + (1 - alpha_t) * obs_emb_t
```

这里 final_emb_t 是截至 t 时刻可用的信息状态。

### 2.5 prediction target

使用 final_emb_t 预测下一时刻上涨概率：

```text
p_next_t = P(up_{t+1} | final_emb_t)
```

主训练目标：

```text
loss_pred = BCE(p_next_t, y_{t+1})
```

可以加一个较小的 alpha 辅助监督，让 alpha 更接近趋势延续概率：

```text
alpha_target_t = 1 - abs(y_{t+1} - y_t)
loss_alpha = BCE(alpha_t, alpha_target_t)
```

总 loss：

```text
loss_emb =
    loss_pred
    + lambda_alpha * loss_alpha
    + lambda_obs_aux * loss_obs_aux
```

建议默认：

```text
lambda_alpha = 0.05
lambda_obs_aux = 0.01
```

### 2.6 导出内容

embedding generator 训练完成后导出：

```text
final_emb / state_emb
obs_emb
alpha
update_rate = 1 - alpha
p_next
```

给 HRL 使用时：

```text
z = final_emb
h = final_emb
p = p_next
```

## 3. Controller

### 3.1 输入

controller 输入保持简洁：

```text
all_final_emb_t: 所有股票当前 final_emb, shape [N, E]
weights_t: 当前真实组合权重, shape [N]
hold_days: 当前持仓已经持续天数
max_hold: 最大持仓天数
```

其中 weights_t 建议使用经历价格漂移后的真实权重：

```text
weights_t = weights_drift
```

这样 controller 判断的是当前真实持仓，而不是刚调仓时的静态 base 权重。

### 3.2 网络结构

每只股票的 final_emb 先过共享 MLP：

```text
asset_repr_i = MLP(final_emb_i)
```

然后做两个聚合：

```text
weighted_emb = sum_i weights_i * asset_repr_i
mean_emb = mean_i asset_repr_i
diff_emb = weighted_emb - mean_emb
```

时间输入：

```text
hold_ratio = hold_days / max_hold
free_ratio = clamp((hold_days - min_hold) / (max_hold - min_hold), 0, 1)
```

controller state：

```text
controller_state = concat(
    weighted_emb,
    mean_emb,
    diff_emb,
    hold_ratio,
    free_ratio
)
```

输出：

```text
switch_logit
value
```

其中：

```text
switch_logit:
    actor 输出，用于 hold/switch

value:
    critic 输出，用于 PPO
```

controller 不选股、不输出权重，只决定是否触发 switch。

### 3.3 持仓天数约束

固定使用：

```text
min_hold = 20
max_hold = 40
```

规则：

```text
hold_days < min_hold:
    强制 hold，不采样 controller action

min_hold <= hold_days < max_hold:
    controller 决定 hold 或 switch

hold_days >= max_hold:
    强制 switch，不采样 controller action
```

这里不是 rule switch。controller 在 20 到 40 天之间自由决策，天数只作为硬约束，避免过短持仓或无限持仓。

## 4. Reward 设计

收益率 reward 统一使用 log return。原因是普通收益率不能直接跨时间相加，而 log return 满足：

```text
log(V_T / V_0) = sum_t log(V_{t+1} / V_t)
```

因此 PPO 中的 step reward 使用 log return 更一致。

设：

```text
gross_portfolio_t = V_{t+1} / V_t
gross_base_t = V_base_{t+1} / V_base_t
```

则：

```text
log_ret_portfolio_t = log(gross_portfolio_t + eps)
log_ret_base_t = log(gross_base_t + eps)
```

建议：

```text
eps = 1e-8
```

### 4.1 Outer Actor reward

outer actor 负责生成 base portfolio，目标是直接最大化真实组合收益。

```text
outer_reward_t =
    reward_scale_outer * log_ret_portfolio_t
```

建议：

```text
reward_scale_outer = 100
```

outer 的辅助监督项：

```text
outer_pred_loss:
    股票级未来收益预测 loss
```

最终 outer loss：

```text
loss_outer =
    PPO_loss_outer
    + outer_pred_coef * outer_pred_loss
```

建议：

```text
outer_pred_coef = 0.1
```

### 4.2 Inner Actor reward

inner actor 负责在 base portfolio 上做权重微调。它不直接和 outer 抢目标，而是优化相对 base 的超额收益。

普通相对收益数值较小，因此需要提高 inner actor 的 reward 系数。

log return 形式：

```text
relative_log_ret_t =
    log_ret_portfolio_t - log_ret_base_t
```

inner reward：

```text
inner_reward_t =
    reward_scale_inner * relative_log_ret_t
```


inner 的辅助监督项：

```text
inner_pred_loss:
    股票级下一日 log return 预测 loss
```

最终 inner loss：

```text
loss_inner =
    PPO_loss_inner
    + inner_pred_coef * inner_pred_loss
```

建议：

```text
inner_pred_coef = 0.05
```

### 4.3 Controller reward

controller 的 RL reward 直接使用每日真实组合 log return：

```text
controller_reward_t =
    reward_scale_controller * log_ret_portfolio_t
```

建议：

```text
reward_scale_controller = 100
```

controller 不使用未来 20 天收益作为 RL reward。未来 20 天收益只用于辅助监督标签。

## 5. Controller 辅助监督

controller 在可自由决策区间内有一个小的监督学习项：

```text
min_hold <= hold_days < max_hold
```

监督信号比较未来 20 天：

```text
ret_hold_20:
    如果继续持有当前真实组合，未来 20 天累计 log return

ret_switch_20:
    如果今天 switch 到 outer actor 新组合，未来 20 天累计 log return
```

计算：

```text
ret_hold_20 = sum_{k=1}^{20} log(gross_hold_{t+k} + eps)
ret_switch_20 = sum_{k=1}^{20} log(gross_switch_{t+k} + eps)
```

标签：

```text
label_switch = 1 if ret_switch_20 > ret_hold_20 else 0
```

监督 loss：

```text
loss_controller_sup =
    BCEWithLogitsLoss(switch_logit, label_switch)
```

可以按收益差加权：

```text
sup_weight = clamp(abs(ret_switch_20 - ret_hold_20) / scale, 0, max_weight)
```

最终 controller loss：

```text
loss_controller =
    PPO_loss_controller
    + controller_sup_coef * sup_weight * loss_controller_sup
```

建议：

```text
controller_sup_coef = 0.05
```

含义：

```text
如果 switch 未来 20 天收益更高，且 controller 选择 switch，监督惩罚很小。
如果 hold 更好，且 controller 选择 hold，监督惩罚很小。
只有 controller 和 20 天收益比较结果相反时，监督项才会明显惩罚。
```

注意：这个监督标签只用于训练，不用于测试或真实决策。

## 6. HRL 训练流程

### 6.1 阶段 1：训练 embedding generator

对 NAS 和 A 股分别训练或分别导出 embedding：

```text
input:
    每只股票特征

target:
    y_{t+1}

output:
    final_emb, obs_emb, alpha, p_next
```

训练完成后冻结 embedding generator。

### 6.2 阶段 2：controller-only 训练

加载已有或预训练好的 HRL outer/inner 权重。

冻结：

```text
embedding generator
outer actor
inner actor
```

训练：

```text
controller actor/critic
```

这一阶段 controller 学：

```text
什么时候继续 hold
什么时候 switch 后让 outer 重新选股
```

训练目标：

```text
maximize daily portfolio log return
+ small supervised switch loss
```

### 6.3 阶段 3：joint 微调

继续冻结：

```text
embedding generator
```

正常训练：

```text
controller
```

小学习率微调：

```text
outer actor
inner actor
```

目标是让 controller、outer、inner 适应彼此，而不是大幅破坏已有 HRL 策略。

## 7. Episode 和更新节奏

### 7.1 episode 起点

每个 epoch 使用多个训练起点。建议变量名明确区分：

```text
train_episode_start_count
```

每个 episode：

```text
从选定起点开始，一直跑到训练集结尾
```

不是固定长度 episode。

建议默认：

```text
train_episode_start_count = 5
```

### 7.2 PPO 更新节奏

outer/controller 的 rollout 按持仓段计数，不把整个训练集压进一个大 buffer。

建议：

```text
outer_rollout_segments = 30
max_hold = 40
rollout_update_steps = 30 * 40 = 1200
```

即大约每累计 30 个外层持仓段就更新一次。

inner actor 的 batch/update 尺度跟 outer 对齐：

```text
inner_batch_size = 1200
```

PPO 参数：

```text
ppo_epochs = 3
```

## 8. 推荐默认参数

```text
outer_window = 40
min_hold = 20
max_hold = 40

train_epochs = 15
train_episode_start_count = 5

outer_rollout_segments = 30
rollout_update_steps = 1200
inner_batch_size = 1200
ppo_epochs = 3

outer_pred_coef = 0.1
inner_pred_coef = 0.05
controller_sup_coef = 0.05

reward_scale_outer = 100
reward_scale_inner = 2000
reward_scale_controller = 100

seeds = 42 43 44 45 46
```

## 9. 评估和保存

训练结果按市场优先保存：

```text
results/hrl_controller/<run_name>/nas/seed_42/...
results/hrl_controller/<run_name>/nas/seed_43/...

results/hrl_controller/<run_name>/sh/seed_42/...
results/hrl_controller/<run_name>/sh/seed_43/...
```

每个 seed 至少保存：

```text
best checkpoint
train log
validation metrics
test backtest curve
test backtest metrics
switch history
controller action statistics
```

回测指标需要和 AlphaStock baseline 保持一致，包括：

```text
ARR
AVol
ASR
MDD
CR
SoR
```

如已有 result.xlsx 汇总逻辑，则 HRL controller 结果也写入同一套指标格式。

## 10. 主要代码改动位置

预计需要改动：

```text
SSM_pipeline.py
    新增或改造 alpha-state embedding generator
    MLP 生成 obs_emb
    LSTM-attn 生成 alpha
    final_emb_t 预测 p_{t+1}
    导出 final_emb / alpha / p_next

utils/PriceMatrix.py
    读取新的 final_emb / alpha / p_next
    将 final_emb 对齐到 HRL 的 z/h 输入

Components/PPO_model.py
    替换或新增 ControllerAC
    输入 all_final_emb + weights_drift + hold_days/max_hold
    输出 switch_logit + value

env/PPO_env.py
    reward 改成 log return
    inner reward 改成 relative log return
    加 min_hold/max_hold 门控
    计算 controller 未来 20 天监督标签

agent/PPO_agent.py
    controller PPO loss
    controller supervised switch loss
    outer stock-level pred loss
    inner next-day stock-level pred loss

Train/PPO_train.py
    controller-only 阶段
    joint 微调阶段
    取消 fixed_cycle 绕过 controller 的逻辑

run_hrl_training.py
    NAS/SH 一次运行入口
    seeds
    路径按 market/seed 保存
    默认参数整理
```

## 11. 关键注意事项

1. 训练和测试时 controller 都不能看到未来 20 天收益。
2. 未来 20 天收益只用于训练时构造 supervised switch label。
3. embedding generator 的 p_next 必须是 t 时刻状态预测 t+1，不能对齐到 t。
4. controller 的输入使用当前真实权重 weights_drift。
5. inner actor 使用相对 log return，并提高 reward_scale_inner。
6. joint 阶段不能继续使用 fixed_cycle，否则 controller 不会真正参与训练。
7. 强制 hold 和强制 switch 的动作不应作为自由 controller action 训练样本。

