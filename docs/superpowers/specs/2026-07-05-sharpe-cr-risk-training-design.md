# Sharpe/CR 风险增强训练设计

日期：2026-07-05
状态：已根据 PPO 下 variable-length segment 可比较的讨论修订，等待实现计划

## 目标

设计一条独立的风险增强训练流程，让 KD4RL+ 尽量在总收益、Sharpe、最大回撤和 CR/Calmar ratio 上同时超过 DeepAries 和 DeepTrader。

第一原则是不影响当前稳定训练流程：

`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`

所有新 reward、新选择指标和新 seed sweep 都必须通过新参数和新脚本显式开启。默认参数保持旧行为。

## 当前问题

现有训练流程主要还是 return-oriented：

- `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh` 中 controller 使用 `--controller_return_coef`，并把 `--controller_mdd_coef` 固定为 `0.0`。
- `Train/controller_pg.py::controller_reward` 虽然保留了 `mdd_coef` 等参数，但当前实现有意忽略 MDD、turnover 和 minimum-count penalty，只使用相对 log-return uplift 加 max-switch overflow penalty。
- `env/PPO_env.py::step` 中 outer reward 目前是 daily log return，outer actor 没有直接优化 Sharpe。
- 现有 validation selection 支持 `return`、`mdd`、`sharpe` 和 `risk_return`，但不支持 CR/Calmar，也不支持综合多指标排名。

因此，单纯调脚本里的系数无法实现“outer 用 Sharpe、controller 用相对 CR”的风险训练目标，需要新增 opt-in reward mode。

## 核心设计原则

三个模块的职责必须分清：

- controller 负责“什么时候换仓”。
- outer actor 负责“换仓时选什么 base portfolio”。
- inner actor 负责“持仓期间如何微调执行权重”。

对应的 reward 也必须分清：

```text
controller reward = 同一 rollout 上 controlled 策略相对 baseline 的 CR 改善
outer reward      = 每个实际持仓 segment 上 base portfolio 的 realized Sharpe
inner reward      = 当前 alpha reward，暂时保持不变
```

关键修正：联合训练时，outer actor 可以使用 Sharpe，但必须把它作为 PPO 中的相对强弱信号来用，而不是把不同长度 segment 的原始 Sharpe 当成绝对可比的最终指标。PPO 的 advantage/normalization 允许不同 segment 长度的样本共同训练；真正需要控制的是短 segment Sharpe 的高方差，以及 controller 切换频率对 outer 更新次数的影响。

## 新参数

新增参数默认保持旧行为：

```text
--outer_reward_mode return|segment_sharpe|fixed_horizon_sharpe
--controller_reward_mode return_uplift|relative_cr
--model_selection_metric sharpe|return|mdd|cr|rank_score
--inner_selection_metric sharpe|return|mdd|cr|rank_score
--controller_selection_metric risk_return|return|mdd|sharpe|cr|rank_score
```

默认值：

```text
outer_reward_mode=return
controller_reward_mode=return_uplift
model_selection_metric=sharpe
inner_selection_metric=return
controller_selection_metric=risk_return
```

新增 reward 系数建议：

```text
outer_sharpe_horizon=30
outer_sharpe_coef=1.0
outer_sharpe_clip=5.0
outer_sharpe_min_days=5
outer_sharpe_length_weight=1
outer_return_floor_coef=0.05
outer_turnover_coef=0.0

controller_cr_coef=1.0
controller_cr_clip=5.0
controller_return_floor_coef=0.05
controller_max_switch_penalty_coef=0.001
```

这些值作为第一轮实验起点，后续可以做小网格搜索。

## Outer Actor Reward

### 旧模式

当 `outer_reward_mode=return` 时，保持当前逻辑不变。outer 的 segment reward 继续来自 daily log return 的聚合。

### 风险增强主模式：Segment Sharpe

当 `outer_reward_mode=segment_sharpe` 时，每个 outer decision 对应一个实际持仓 segment。outer reward 使用该 segment 内 base portfolio 的 realized Sharpe。

设某次 outer 在第 `t_s` 天输出 base portfolio `w_s`，该组合实际被 controller 持有到 `t_e`：

```text
base_path_s = 从 t_s 到 t_e，使用 w_s 作为 base portfolio 得到的 base-only 净值路径

base_daily_log_returns_s = base_path_s 的逐日 log return

segment_sharpe_s =
  mean(base_daily_log_returns_s)
  / (std(base_daily_log_returns_s) + eps)
  * sqrt(252)

segment_log_return_s = log(base_path_s[-1] / base_path_s[0])
```

outer reward：

```text
outer_reward_s =
  outer_sharpe_coef * clip(segment_sharpe_s, -outer_sharpe_clip, outer_sharpe_clip)
+ outer_return_floor_coef * segment_log_return_s
- outer_turnover_coef * turnover_to_w_s
```

这里的 `turnover_to_w_s` 可以先设为 0，不作为第一版核心约束。

必须使用 base-only path，而不是 inner 调整后的 executed path。这样 outer 的 credit 只来自它选出的 base portfolio，inner actor 的局部执行能力不会污染 outer reward。

### 为什么 variable-length segment 仍然可以用 Sharpe

outer actor 使用 PPO，训练时比较的是 action 的相对 advantage，而不是把每个 segment Sharpe 当作最终表格指标直接比较。因此 segment 步数不同不是原则性问题。

但短 segment 的 Sharpe 方差更大，所以实现时需要三层保护：

```text
1. 对 segment_sharpe 做 clip。
2. 对 outer reward/advantage 做 batch normalization 或沿用 PPO 的 advantage normalization。
3. 对过短 segment 使用 length reliability weight，或在天数不足 outer_sharpe_min_days 时退化为 return floor。
```

可选的 length reliability weight：

```text
length_weight = min(1, sqrt(segment_days / outer_sharpe_horizon))
outer_reward_s = length_weight * clipped_segment_sharpe + return_floor
```

这个权重不是因为不同长度不能训练，而是为了降低短段 Sharpe 的估计噪声。

### 可选对照模式：Fixed-Horizon Sharpe

`outer_reward_mode=fixed_horizon_sharpe` 保留为 ablation 或诊断模式。它在每个 outer action day 使用固定 `H` 天的反事实 Sharpe：

```text
path_H(w_t) = 从第 t 天开始，使用 w_t 买入并持有 H 天得到的反事实净值路径
```

该模式的优点是评价窗口完全一致，缺点是更像监督式未来标签，和实际 controller 持仓路径的耦合较弱。第一版主实验使用 `segment_sharpe`，fixed-horizon 作为稳定性对照。

## Controller Reward

controller 使用 rollout 级别的相对 CR 改善。

对于同一个训练窗口，构造两条路径：

```text
baseline   = 无自由 controller / 固定规则 / 继续原策略的 counterfactual path
controlled = 当前 controller 策略产生的 path
```

两条路径必须覆盖同一个 rollout window，因此长度一致，可以比较 CR。

CR 定义：

```text
annualized_return = mean(daily_returns) * 252
CR = annualized_return / max(max_drawdown, eps)
```

controller reward：

```text
controller_reward =
  controller_cr_coef * clip(CR_controlled - CR_baseline, -controller_cr_clip, controller_cr_clip)
+ controller_return_floor_coef * (log_return_controlled - log_return_baseline)
- normalized_max_switch_overflow_penalty
```

使用相对 CR uplift，而不是绝对 CR，原因是 controller 应学习“当前切换策略相对同一窗口 baseline 是否改善了风险收益效率”。

return floor 很小，只用于防止 controller 学到“收益很低但回撤也很低”的保守路径。

switch penalty 仍然由 controller 承担，因为切换频率是 controller 的职责，不应该通过 outer reward 间接惩罚。

## Inner Actor Reward

第一版不改 inner actor reward：

```text
inner_reward = executed_return - base_return
```

inner actor 的任务仍然是持仓期内相对 base portfolio 做局部增强。暂时不把 Sharpe 或 CR 直接塞给 inner，避免三个模块同时改变导致无法定位实验效果。

## 各阶段训练设计

### 阶段 1：Warmup Outer

controller 固定周期，inner 关闭或固定。

目标是让 outer actor 学会选择实际持仓段内风险收益质量更高的 base portfolio。

```text
outer_reward = base-only segment Sharpe
             + small_return_floor
```

warmup 阶段 controller 固定，因此 segment 长度基本一致，segment Sharpe 的噪声较低。这个阶段可以让 outer 先建立 Sharpe-oriented 的选组合能力。

### 阶段 2：Warmup Inner

outer 和 controller 不作为主要训练对象，inner 保持旧目标：

```text
inner_reward = executed_return - base_return
```

这个阶段只让 inner 学习在已有 base portfolio 上做局部执行增强。

### 阶段 3：固定周期 HRL Joint

controller 仍然由固定 schedule 控制，outer 和 inner 可以一起训练。

```text
outer_reward = base-only segment Sharpe
inner_reward = executed_return - base_return
```

此阶段的作用是让 outer 的 segment Sharpe 目标和 inner 的局部 alpha 目标先共同稳定下来。

### 阶段 4：Controller PG

冻结或近似冻结 outer/inner，单独训练 controller。

```text
controller_reward = rollout_level_relative_CR_uplift
```

controlled 和 baseline 在同一个 rollout window 上比较，因此 CR 可比。这个阶段只学习“什么时候切换”。

### 阶段 5：Controller + Outer 联合训练

这是最关键阶段，reward 必须保持职责分离：

```text
controller_reward = rollout-level relative CR uplift
outer_reward      = base-only realized segment Sharpe of selected base portfolio
inner_reward      = executed_return - base_return
```

不要让 outer 使用 controller 的 rollout CR reward。outer 使用自身 segment 的 Sharpe，相当于评价“这次换过去的 base portfolio 在实际持仓期间质量如何”。

controller 决定切换频率，outer 评价每次切换时给出的组合在实际持仓段内的相对强弱。

实现时还要注意：outer loss 应按 outer decision 数量做平均或保持现有 PPO 归一化，避免“切换更多导致 outer 梯度次数更多”成为隐式奖励。切换频率只应由 controller 的 CR reward 和 switch penalty 管。短 segment 的 Sharpe 只作为带 clip/weight 的相对 reward 使用，不作为最终评测指标。

## 模型选择

训练 reward 负责优化模块行为，最终模型选择使用多指标。

验证指标：

```text
total_return: 越高越好
sharpe:       越高越好
max_drawdown: 越低越好
cr:           越高越好
```

新增 `rank_score`：

```text
rank_score =
  rank(total_return, higher better)
+ rank(sharpe, higher better)
+ rank(cr, higher better)
+ rank(max_drawdown, lower better)
```

保存以下 checkpoint：

```text
best_return
best_sharpe
best_mdd
best_cr
best_rank_score
```

如果某个模型在所有指标上都是 best，则直接作为主结果。如果没有，则使用 `best_rank_score` 作为综合主结果，同时保留单指标冠军用于表格和消融分析。

## Seed Sweep

新增独立脚本：

`train_sh/run_end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep.sh`

第一轮 seed：

```text
NAS_SEEDS="41 42 43 44 45 46 47 48 49 50"
SH_SEEDS="82 83 84 85 86 87 88 89 90 91"
```

算力允许且训练稳定后扩展到 20 seeds：

```text
NAS_SEEDS="40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"
SH_SEEDS="80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
```

默认输出目录：

`results/end_to_end_hrl_controller_joint_sharpe_cr_seed_sweep`

该脚本必须保留现有 protected-output 检查，拒绝写入 archived good-model roots。

## 实现数据流

1. `run_hrl_training.py` 解析新增 reward mode、horizon、clip 和系数参数。
2. runtime config 保存这些参数，默认值保持旧行为。
3. `PPO_Env.step` 继续输出 daily log return、portfolio value、base weight、outer action 等信息。
4. 新增 segment-level base-only Sharpe 计算函数，用于根据 outer action `w_s` 和实际持仓段 `[t_s, t_e]` 计算 `segment_sharpe_s` 和 `segment_log_return_s`。
5. buffer 或 trainer 在 outer decision segment 结束时记录 segment Sharpe outer reward。
6. `HRL_Buffer.finish_episode` 在 `outer_reward_mode=return` 时保持旧聚合逻辑，在 `segment_sharpe` 时使用对应 segment 的 base-only Sharpe reward；`fixed_horizon_sharpe` 作为可选对照模式。
7. controller counterfactual rollout 计算 baseline 和 controlled 的 log return、MDD、annualized return、CR。
8. `controller_reward` 根据 `controller_reward_mode` 选择旧 return uplift 或新 relative CR uplift。
9. validation 计算 total return、annualized return、Sharpe、MDD 和 CR。
10. trainer 保存单指标 best checkpoint 和综合 rank-score checkpoint。
11. 新 seed-sweep 脚本运行 NASDAQ 和 CSI-300 的扩大 seed 实验。

## 测试计划

实现前先加 focused tests：

- 旧默认参数下，`controller_reward` 保持 return uplift 行为不变。
- `controller_reward_mode=relative_cr` 时，reward 使用 clipped CR uplift。
- CR 在 max drawdown 极小或为 0 时使用 epsilon 和 clip，避免爆炸。
- `_validation_score` 支持 `cr` 和 `rank_score`。
- `outer_reward_mode=return` 时，outer reward 聚合与旧逻辑一致。
- `outer_reward_mode=segment_sharpe` 时，outer reward 使用 base-only realized segment Sharpe，不使用 inner executed return。
- 两个相同 base-only return 序列应得到相同 segment Sharpe reward，即使它们来自不同 rollout。
- 过短 segment 应触发 clip、length weight 或 return-floor fallback，避免 Sharpe 爆炸。
- `outer_reward_mode=fixed_horizon_sharpe` 作为可选对照时，应使用固定 horizon 反事实 Sharpe。
- 原始 `run_end_to_end_hrl_controller_joint_nas49_sh90.sh` echo 测试仍然通过。
- 新 seed-sweep 脚本 echo 测试检查 seeds、reward modes、selection metrics 和 safe output root。

## 风险与缓解

segment Sharpe 在短持仓段上可能有噪声。使用 `outer_sharpe_clip`、`eps`、length reliability weight 和 small return floor 控制。

CR 在极低回撤时可能爆炸。使用 `controller_cr_clip` 和 `eps`。

所有指标同时 best 不一定存在，因为总收益和回撤天然可能冲突。通过 `best_rank_score` 和单指标 best checkpoint 同时保存来避免只押一个指标。

outer 使用 segment Sharpe 会受到 controller 切换频率影响。通过按 outer decision 归一化 loss、controller switch penalty、以及短段 Sharpe 降权来缓解。

fixed-horizon Sharpe 可作为 ablation。如果实际 segment Sharpe 训练不稳定，再切换到 fixed-horizon 模式做稳定性对照。

现有好模型必须可复现。通过默认参数保持旧行为、保留旧脚本、测试旧脚本 echo 输出进行保护。
