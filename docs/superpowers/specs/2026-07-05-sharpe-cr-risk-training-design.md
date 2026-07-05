# Sharpe/CR 风险增强训练设计（最小侵入版）

日期：2026-07-05
状态：按现有 end-to-end 链路简化，等待实现计划

## 目标

沿现有训练链路：

`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`

新增一条独立风险增强训练流程。目标是在尽量不改动原训练结构的前提下，让模型在收益率优先的基础上改善 Sharpe、最大回撤和 CR，从而更有机会超过 DeepAries、DeepTrader 等 baseline。

原脚本不改默认行为。新实验通过新脚本和新增 reward 参数显式开启。

## 最小侵入原则

本设计只改三件事：

1. outer actor 的 reward 从 segment return 切换为 segment Sharpe。
2. controller 的 reward 从 return uplift 切换为相对 CR uplift。
3. seed 范围扩大，模型选择先按收益率最高。

不引入 rank score，不引入复杂多目标 selection，不引入 fixed-horizon reward，不对 reward 做 clip。只保留必要的 `eps` 防止除零。

## 新增参数

新增参数默认保持旧行为：

```text
--outer_reward_mode return|segment_sharpe
--controller_reward_mode return_uplift|relative_cr
```

风险增强脚本中使用：

```text
outer_reward_mode=segment_sharpe
controller_reward_mode=relative_cr
model_selection_metric=return
inner_selection_metric=return
controller_selection_metric=return
```

controller 保持自由切换：

```text
controller_no_hold_constraints=1
controller_train_max_hold=0
controller_eval_max_hold=0
controller_decision_mode=daily
controller_eval_decision_mode=daily
controller_eval_force_max_hold=0
```

这里 `controller_train_max_hold=0` 和 `controller_eval_max_hold=0` 表示禁用 forced max-hold。除了第一个交易日必须建仓，之后是否 switch 完全由 controller policy 决定。

## Outer Actor Reward

现有 outer reward 是每个 segment 内 daily log return 的求和。风险增强版只把这个 segment 聚合方式改成 Sharpe。

设某个 outer decision 产生一个实际持仓段：

```text
segment_returns = 该 segment 内的 daily portfolio log returns
```

当 `outer_reward_mode=return`：

```text
outer_reward = sum(segment_returns)
```

当 `outer_reward_mode=segment_sharpe`：

```text
outer_reward =
  mean(segment_returns)
  / (std(segment_returns) + eps)
  * sqrt(252)
```

这里的 segment 是 controller 实际决定出来的 segment。它会受到 controller stopping rule 的影响，这是可以接受的，因为在联合训练时我们希望 outer actor 在整个 controller-aware 框架下学习：controller 切到哪里，outer 就评价这段实际持仓结果的相对强弱。

不同 segment 的步数可以不同。outer actor 使用 PPO，训练比较的是 advantage 的相对强弱，不要求每个 segment 的长度完全一致。短 segment 的 Sharpe 会更噪声，但第一版不做复杂处理，只使用 `eps` 保证数值稳定。

## Controller Reward

controller reward 使用当前 controller 策略相对于“训练好的 HRL”的 CR 改善。

训练好的 HRL 指前面 warmup outer、warmup inner、固定周期 HRL joint 后得到的 HRL 策略。controller 学的是：在这套已训练 HRL 基础上，加入自由 switch 后，整段 rollout 的 CR 是否提升。

对同一个 rollout window，计算两条路径：

```text
baseline_path   = 训练好的 HRL，不使用自由 controller
controlled_path = 训练好的 HRL + 当前 controller policy
```

CR 定义：

```text
CR = annualized_return / (max_drawdown + eps)
```

当 `controller_reward_mode=return_uplift`，保持旧逻辑。

当 `controller_reward_mode=relative_cr`：

```text
controller_reward =
  CR(controlled_path) - CR(baseline_path)
  - switch_penalty
```

其中 `switch_penalty` 继续使用现有 soft max-switch penalty。它只作为软惩罚，不作为 hard cap。

## 各阶段训练设计

### 1. Warmup Outer

沿原链路训练 outer。

风险增强脚本中：

```text
outer_reward = segment_sharpe
```

controller 仍由固定 schedule 控制，因此这个阶段的 segment 长度相对稳定，outer 先学习风险调整后的组合选择。

### 2. Warmup Inner

保持原逻辑不变：

```text
inner_reward = executed_return - base_return
```

不把 Sharpe 或 CR 加到 inner actor 上，避免一次性改动过多。

### 3. 固定周期 HRL Joint

沿原链路做 outer + inner joint。

```text
outer_reward = segment_sharpe
inner_reward = executed_return - base_return
```

这个阶段得到“训练好的 HRL”，作为后续 controller 相对 CR reward 的 baseline。

### 4. Controller PG

冻结或近似冻结训练好的 HRL，训练 controller。

```text
controller_reward = CR(controlled_path) - CR(trained_HRL_baseline_path) - switch_penalty
```

controller 不考虑最大持仓期，不考虑最小持仓期。除首日建仓外，每天是否 switch 都由 controller policy 决定。

### 5. Controller-active Joint

沿原链路做最终 controller-active joint。

```text
outer_reward      = segment_sharpe
controller_reward = relative_cr
inner_reward      = executed_return - base_return
```

这里 outer 的 segment Sharpe 接受 controller stopping rule 的影响。这个设计符合联合训练逻辑：整个框架共同决定实际持仓段，outer 在这个实际框架下学习每次 switch 后组合的相对强弱。

## 模型选择与评估

第一版不做复杂多指标选择。

训练中 checkpoint selection 先按收益率最高：

```text
model_selection_metric=return
controller_selection_metric=return
inner_selection_metric=return
```

最终报告仍然统计：

```text
total_return
sharpe
max_drawdown
cr
```

如果收益率最高的模型风险指标不理想，再考虑第二轮加权 selection 或 CR selection。第一轮先保持简单。

## Seed Sweep

新增独立脚本：

`train_sh/run_end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep.sh`

第一轮 seeds：

```text
NAS_SEEDS="41 42 43 44 45 46 47 48 49 50"
SH_SEEDS="82 83 84 85 86 87 88 89 90 91"
```

算力允许后扩展到 20 seeds：

```text
NAS_SEEDS="40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"
SH_SEEDS="80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
```

输出目录独立：

`results/end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep`

保留现有 protected-output 检查，避免覆盖已有好模型。

## 实现要点

1. 在 `run_hrl_training.py` 增加 `outer_reward_mode` 和 `controller_reward_mode` 参数。
2. 在 outer segment 聚合处增加 `segment_sharpe` 模式。
3. 在 controller reward 处增加 `relative_cr` 模式。
4. controller 的 baseline 使用训练好的 HRL，不使用自由 controller。
5. controller-active 阶段确保 `controller_train_max_hold=0` 和 `controller_eval_max_hold=0` 真正禁用 forced max-hold。
6. 新增 seed-sweep 脚本，复制现有 e2e 链路，只改输出目录、seeds、reward mode 和 selection metric。
7. 原脚本默认行为必须不变。

## 测试计划

需要覆盖：

- 原 `run_end_to_end_hrl_controller_joint_nas49_sh90.sh` echo 测试仍通过。
- `outer_reward_mode=return` 时 outer reward 与旧逻辑一致。
- `outer_reward_mode=segment_sharpe` 时 outer reward 使用 segment returns 计算 Sharpe。
- `controller_reward_mode=return_uplift` 时 controller reward 与旧逻辑一致。
- `controller_reward_mode=relative_cr` 时 controller reward 等于 controlled path 和 trained-HRL baseline path 的 CR 差值再减 soft switch penalty。
- `controller_train_max_hold=0` 和 `controller_eval_max_hold=0` 时不会因为达到全局 `max_hold` 被 forced switch。
- 新 seed-sweep 脚本使用独立 output root 和扩大 seeds。

## 关于“短 segment Sharpe 噪声”的说明

之前提到“短 segment Sharpe 噪声大”，意思是：segment 天数越少，Sharpe 的估计方差越高。这个问题存在，但不意味着不能训练。PPO 比较的是 batch 内相对 advantage，第一版可以先接受这个噪声，只用 `eps` 做数值保护。若实验中出现极端不稳定，再考虑加 clip 或最短天数 fallback。

## 风险

第一版风险增强训练可能出现两类问题：

- controller 过度频繁 switch，导致成本上升。
- segment Sharpe 噪声较大，outer reward 方差上升。

先不引入复杂约束。通过 seed sweep 和最终收益率选择观察效果。如果收益提升且风险指标改善，则说明最小侵入方案有效；如果不稳定，再进入第二轮 reward shaping。
