# SH Relative Inner 多种子训练脚本设计

**目标：** 在 SH 市场对 seed 44、46、49、54 分别从头训练 Outer 和 Relative Inner。

## 执行方式

- 四个 seed 在同一 GPU 上顺序执行，避免并发争用显存。
- 每个 seed 从头训练，不加载冻结 checkpoint。
- 日程为 Outer warmup 4 epoch、Relative Inner warmup 5 epoch、Outer+Inner joint 2 epoch。
- Controller 全程关闭。
- 沿用 NAS44 Relative Inner 的 `close_anchor`、`relative_tcn_attn`、ASU 系数和固定 30 日持有设置。
- 每个 seed 使用独立 run name 和日志文件。

## 可配置项

- `SH_SEEDS`：默认 `44 46 49 54`。
- `OUTER_EPOCHS`、`INNER_EPOCHS`、`JOINT_EPOCHS`：默认分别为 `4`、`5`、`2`。
- `GPU_ID` / `CUDA_VISIBLE_DEVICES`：默认 GPU 0。
- `OUTPUT_ROOT`、`PYTHON_BIN`：允许环境变量覆盖。
- `DRY_RUN=1`：只打印四条训练命令，不启动训练。

## 失败策略

- 正式运行前检查 Python 可执行文件。
- 任一 seed 训练失败时立即退出，保留此前日志。
