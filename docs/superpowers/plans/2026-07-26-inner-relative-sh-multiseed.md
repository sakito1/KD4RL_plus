# SH Relative Inner Multi-seed Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 新增一个可直接运行 SH seed 44、46、49、54 的 Outer + Relative Inner 训练脚本。

**Architecture:** 脚本复用 NAS44 的 Relative Inner 参数，在单 GPU 上逐 seed 从头训练 Outer 和 Inner。标准日程为 Outer 4 epoch、Inner 5 epoch、Outer+Inner joint 2 epoch；每个 seed 使用独立 run name 与日志路径。

**Tech Stack:** Bash、现有 `run_hrl_training.py`。

---

### Task 1: 新增多种子训练脚本

**Files:**

- Create: `train_sh/run_inner_relative_sh_multiseed.sh`

- [ ] **Step 1: 运行缺失脚本测试并确认失败**

Run:

```bash
test -f train_sh/run_inner_relative_sh_multiseed.sh
```

Expected: exit code 1。

- [ ] **Step 2: 编写脚本**

脚本必须实现：

- 默认 `SH_SEEDS="44 46 49 54"`；
- 默认 `OUTER_EPOCHS=4`、`INNER_EPOCHS=5`、`JOINT_EPOCHS=2`；
- 使用 `--markets sh --seeds <seed>`；
- 沿用 `relative_tcn_attn`、`close_anchor`、`inner_asu_coef=0.05`；
- 不传入冻结 checkpoint 或 `inner_only_finetune`；
- 使用 `--warmup_outer_epochs "$OUTER_EPOCHS"`、`--warmup_inner_epochs "$INNER_EPOCHS"`、`--joint_epochs "$JOINT_EPOCHS"`；
- 每个 seed 写入单独日志；
- 支持 `DRY_RUN=1`。

- [ ] **Step 3: 验证 shell 语法**

Run:

```bash
bash -n train_sh/run_inner_relative_sh_multiseed.sh
```

Expected: exit code 0。

- [ ] **Step 4: 验证 dry-run 行为**

Run:

```bash
DRY_RUN=1 bash train_sh/run_inner_relative_sh_multiseed.sh
```

Expected:

- exit code 0；
- 输出恰好包含 seed 44、46、49、54 四条训练命令；
- 每条命令包含 `--warmup_outer_epochs 4 --warmup_inner_epochs 5 --joint_epochs 2`；
- 每条命令不包含冻结 checkpoint 和 `inner_only_finetune`。

## Verification

执行 `bash -n`、默认 dry-run 和训练日程断言。

## Next skill

`$superpower-verification`
