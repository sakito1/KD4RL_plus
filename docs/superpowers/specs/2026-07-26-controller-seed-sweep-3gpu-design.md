# Controller 多种子三 GPU 训练设计

## 目标

在已有 Outer+Inner checkpoint 上训练同种子的 Controller，并自动运行测试集评估：

- NAS：44、45、47、50、56、57、58；
- SH：44、46、49、54；
- 三张 GPU，每张 GPU 同时运行两个任务。

训练沿用当前 `explore_controller_from_nas45_outer_inner.sh` 的监督预训练和 PG 参数，不重新训练 Outer 或 Inner，也不修改模型实现。

## 结构

### 通用单任务脚本

将单任务入口参数化，接收：

- `MARKET`：`nas` 或 `sh`；
- `CONTROLLER_SEED`：同时作为源 checkpoint 种子和 Controller 训练种子；
- `GPU_ID`；
- `SOURCE_ROOT`、`OUTPUT_ROOT` 和 `RUN_NAME`。

源 checkpoint 固定解析为：

```text
${SOURCE_ROOT}/${MARKET}/ppo/seed_${CONTROLLER_SEED}/checkpoints/hrl_fixed_best.pth
```

脚本复用当前 Controller 配置，但不传入 `--skip_test`。训练结束后，
`run_hrl_training.py` 使用验证集选出的 `best_ckpt` 运行测试集评估。

### 三 GPU 调度脚本

调度器生成 11 个 `market:seed` 任务，并建立六条 lane：

- GPU 0：lane 0、lane 1；
- GPU 1：lane 0、lane 1；
- GPU 2：lane 0、lane 1。

任务按轮询方式进入六条 lane。六条 lane 并行运行，每条 lane 内的任务串行执行，因此每张 GPU 最多同时有两个训练进程。

默认种子可通过 `NAS_SEEDS` 和 `SH_SEEDS` 环境变量覆盖；GPU 编号可通过 `GPU0`、`GPU1`、`GPU2` 覆盖。

## 输出

默认输出根目录与源 Outer+Inner 结果分离。每个任务具有唯一运行目录和控制台日志，日志名包含市场、种子和 GPU。

训练全部结束后，调度器从各任务日志提取测试阶段指标，写入统一汇总文件。原始测试日志仍完整保留，汇总失败不覆盖或删除原始结果。

## 错误处理

- `DRY_RUN=1` 时只打印任务分配、checkpoint 和最终命令，不要求 checkpoint 在当前机器存在；
- 正式运行时，单任务在启动前检查 Python 和 checkpoint；
- 某个任务失败时记录失败状态，该 lane 继续执行后续任务；
- 所有 lane 结束后，只要存在失败任务，调度器返回非零状态并提示检查对应日志。

## 验证

测试覆盖：

1. 默认生成指定的 11 个市场—种子任务；
2. 每张 GPU 恰有两条并发 lane；
3. 每个任务加载同市场、同种子的 `hrl_fixed_best.pth`；
4. 训练命令保留当前预训练/PG 参数；
5. 命令不包含 `--skip_test`；
6. shell 语法检查通过；
7. `DRY_RUN=1` 可以在没有全部 checkpoint 的本机完成。

