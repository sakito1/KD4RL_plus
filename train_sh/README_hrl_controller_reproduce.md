# HRL + Controller 复现说明

本文档记录当前可用的训练入口、这轮通过验证的 SH/NAS 结果位置，以及需要注意的复现方式。

## 1. 一体化训练入口

完整流程入口：

```bash
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

默认流程是：

```text
HRL warmup -> fixed HRL joint -> controller PG -> controller + HRL joint
```

当前脚本默认配置里，最终 controller+HRL joint 使用低学习率：

```bash
JOINT_LR_MULT=0.0001
CONTROLLER_JOINT_EPOCHS=1
```

建议复现实验时显式指定新的输出目录，避免覆盖已经验证过的结果：

```bash
REPRODUCE_BEST_MODE=0 \
OUTPUT_ROOT=results/e2e_standard_joint_lowlr_reproduce_$(date +%Y%m%d_%H%M%S) \
RUN_NAME=lookback60_hold30_standard_joint_lowlr_nas49_sh90 \
JOINT_LR_MULT=0.0001 \
CONTROLLER_JOINT_EPOCHS=1 \
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

## 2. 历史最好/保底复现入口

同一个入口也可以切到历史最好复现模式：

```bash
REPRODUCE_BEST_MODE=1 \
OUTPUT_ROOT=results/e2e_reproduce_best_$(date +%Y%m%d_%H%M%S) \
bash train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh
```

这个模式会委托到：

```text
scripts/run_reproduce_hrl_controller_nas49_sh90.sh
```

默认会读取 `results/end` 里的 archived best 作为 floor，避免最终结果低于已经保存的好模型。它不会写入 `results/end`。

## 3. 单独 controller-first 入口

如果只想从已有 fixed HRL checkpoint 训练 controller，再做 controller+HRL joint，可以用：

```bash
CONTROLLER_ONLY=0 \
NAS_SEEDS=49 \
SH_SEEDS= \
OUTPUT_ROOT=results/controller_first_joint_lowlr_retry_$(date +%Y%m%d_%H%M%S) \
RUN_NAME=lookback60_hold30_controller_first_joint_lowlr_nas49 \
JOINT_LR_MULT=0.0001 \
bash train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh
```

这个脚本默认从下面目录读取 fixed HRL：

```text
results/hrl_lookback60_hold30_inner_noaux_retrain/lookback60_hold30_inner_noaux_retrain
```

注意：controller PG 存在随机性；同样 seed 下也可能因为 rollout 采样轨迹差异出现小幅波动。NAS49 这次需要 retry2 才完全复现到历史最好。

## 4. 当前已验证结果

这轮最终验证通过的 SH 和 NAS 不是同一个输出总目录里的一次完整运行结果，而是从两次运行中选择：

### SH90

结果目录：

```text
results/e2e_standard_joint_lowlr_20260622_01/lookback60_hold30_standard_joint_lowlr_nas49_sh90/sh
```

关键文件：

```text
results/e2e_standard_joint_lowlr_20260622_01/lookback60_hold30_standard_joint_lowlr_nas49_sh90/sh/ppo/seed_90/checkpoints/best_model.pth
results/e2e_standard_joint_lowlr_20260622_01/lookback60_hold30_standard_joint_lowlr_nas49_sh90/sh/ppo/seed_90/test_s3_AllModules.csv
results/e2e_standard_joint_lowlr_20260622_01/lookback60_hold30_standard_joint_lowlr_nas49_sh90/sh/seed_90.log
```

测试结果：

```text
Scenario 3 All Modules
last value = 3401.254150390625
Total Ret  = 240.13%
Switches   = 119
free       = 92
forced_s   = 27
```

历史最好 SH90：

```text
last value = 3049.942626953125
```

因此这轮 SH90 高于历史最好。

### NAS49

结果目录：

```text
results/controller_first_joint_lowlr_retry_20260622_02/lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/nas
```

关键文件：

```text
results/controller_first_joint_lowlr_retry_20260622_02/lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/nas/ppo/seed_49/checkpoints/best_model.pth
results/controller_first_joint_lowlr_retry_20260622_02/lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/nas/ppo/seed_49/test_s3_AllModules.csv
results/controller_first_joint_lowlr_retry_20260622_02/lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/nas/seed_49.log
```

测试结果：

```text
Scenario 3 All Modules
last value = 3655.282470703125
Total Ret  = 265.53%
Switches   = 266
free       = 231
forced_s   = 35
```

历史最好 NAS49：

```text
last value = 3655.282470703125
```

因此这轮 NAS49 等于历史最好。

## 5. 当前结论

当前已验证的最终选择是：

```text
SH90  : 使用 end-to-end low-lr joint 结果
NAS49 : 使用 controller-first low-lr retry2 结果
```

这两个结果都不低于各自历史最好。后续论文实验撰写如果需要一个统一归档目录，建议另建新目录复制这两组 selected artifacts，不要直接覆盖 `results/end`。
