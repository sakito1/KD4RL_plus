# Controller Normalized Supervised Pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 单独验证 Controller 的风险、候选优势和切换标签监督能否被学习，避免策略梯度采样干扰。

**Architecture:** 复用现有连续目标缩放参数，将风险和优势目标按各自 5% 阈值缩放到单位尺度；将快速验证的初始切换 logit 设为 0。增加监督预训练后直接保存并退出的配置，使实验不进入 PG。

**Tech Stack:** Python 3.10、PyTorch、pytest、Bash。

---

### Task 1: 监督预训练后退出

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `Train/PPO_train.py`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_run_hrl_training_command.py`

- [ ] 写失败测试：配置 `controller_pretrain_only=True` 时，监督更新完成后保存 `save_name`，返回 `pretrain_only=True`，且不执行 PG rollout。
- [ ] 写失败测试：`--controller_pretrain_only` 能从父进程命令转发至子进程配置。
- [ ] 运行两个定向测试，确认因功能缺失而失败。
- [ ] 实现命令参数、配置转发以及监督阶段后的保存和提前返回。
- [ ] 重新运行定向测试并确认通过。

### Task 2: 快速监督脚本

**Files:**
- Modify: `train_sh/controller_5pct_outer_sh77_quick.sh`
- Test: `tests/test_end_to_end_hrl_controller_joint_script.py`

- [ ] 写失败测试，要求脚本包含：`controller_pretrain_only`、风险/优势 target scale 均为 20、初始 bias 为 0、5 个监督 epoch 和 3 次 replay。
- [ ] 运行脚本测试并确认失败。
- [ ] 修改脚本，仅运行监督预训练；保留 seed 77、12 个300日窗口和冻结 Outer。
- [ ] 运行脚本测试和 `bash -n`。

### Task 3: 回归验证

**Files:**
- Verify only.

- [ ] 运行 Controller、命令转发和脚本相关测试。
- [ ] 运行 Python 编译检查、Bash 语法检查和 `git diff --check`。
- [ ] 不执行正式训练，不提交或推送仓库。
