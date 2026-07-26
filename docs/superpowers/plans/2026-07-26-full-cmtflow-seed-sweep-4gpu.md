# 完整 CMTFlow 多种子四 GPU 训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 为指定 NAS/SH 种子提供从零完成 Outer、Inner、Outer+Inner joint、Controller 和测试的四 GPU 单进程调度入口。

**Architecture:** 单任务 shell 只负责一个市场—种子的完整 `run_hrl_training.py` 命令；调度 shell 只负责把 11 个任务轮询分配给 4 GPU × 1 lane，并汇总测试日志。所有模型训练保持在同一 Python 进程内，从而直接复用阶段 checkpoint，不做额外加载或 Controller 后联合微调。

**Tech Stack:** Bash、现有 PyTorch 训练入口、Python pytest。

---

### Task 1: 完整单种子训练入口

**Files:**
- Create: `train_sh/train_full_cmtflow_seed.sh`
- Create: `tests/test_full_cmtflow_seed_sweep_4gpu_script.py`

- [ ] **Step 1: 写失败测试**

用 `DRY_RUN=1`、`MARKET=sh`、`SEED=54` 运行尚不存在的单任务脚本，断言命令包含：

```python
assert "--markets sh" in output
assert "--seeds 54" in output
assert "--warmup_outer_epochs 4" in output
assert "--warmup_inner_epochs 2" in output
assert "--joint_epochs 1" in output
assert "--controller_sup_pretrain_epochs 3" in output
assert "--controller_aux_replay_epochs 30" in output
assert "--controller_epochs 5" in output
assert "--controller_sup_coef 0.01" in output
assert "--controller_aux_mdd_coef 0.01" in output
assert "--controller_aux_switch_adv_coef 0.01" in output
assert "--controller_switch_rate_penalty_coef 0.01" in output
assert "--frozen_hrl_checkpoint" not in output
assert "--controller_only_finetune" not in output
assert "--end_to_end_controller_joint" not in output
assert "--skip_test" not in output
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  -q tests/test_full_cmtflow_seed_sweep_4gpu_script.py -k single
```

Expected: FAIL，单任务脚本不存在。

- [ ] **Step 3: 实现单任务脚本**

脚本接收 `MARKET`、`SEED`、`GPU_ID`、`OUTPUT_ROOT`、`RUN_NAME`，验证市场和种子，
并构造一条包含以下阶段的命令：

```text
Outer 4 / Inner 2 / Outer+Inner joint 1
Controller supervised pretrain 3 × replay 30
Controller PG 5 / automatic test
```

Outer/Inner 其余参数复制 `run_outer_inner_seed_sweep_3gpu.sh`；Controller 参数复制
`train_controller_from_outer_inner.sh`。正式运行使用
`CUDA_VISIBLE_DEVICES=$GPU_ID`，dry-run 只打印命令。

- [ ] **Step 4: 运行 GREEN 验证**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  -q tests/test_full_cmtflow_seed_sweep_4gpu_script.py -k single
bash -n train_sh/train_full_cmtflow_seed.sh
```

Expected: 测试 PASS，shell exit 0。

### Task 2: 四 GPU 四 lane 调度与汇总

**Files:**
- Create: `train_sh/run_full_cmtflow_seed_sweep_4gpu.sh`
- Modify: `tests/test_full_cmtflow_seed_sweep_4gpu_script.py`

- [ ] **Step 1: 写失败调度测试**

以 `DRY_RUN=1` 运行调度器，断言 NAS `44 45 47 50 56 57 58`、SH
`44 46 49 54` 各出现一次，并断言：

```python
assert output.count("lane 0 queue:") == 4
assert "lane 1 queue:" not in output
assert "Concurrent jobs per GPU: 1" in output
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  -q tests/test_full_cmtflow_seed_sweep_4gpu_script.py -k scheduler
```

Expected: FAIL，调度脚本不存在。

- [ ] **Step 3: 实现四 lane 调度**

使用：

```bash
GPU_IDS=("${GPU0:-0}" "${GPU1:-1}" "${GPU2:-2}" "${GPU3:-3}")
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
slot_count=$((4 * JOBS_PER_GPU))
slot=$((index % slot_count))
gpu_index=$((slot % 4))
lane=$((slot / 4))
```

四条 queue 后台并行，每条 queue 内串行调用单任务脚本；单任务失败时继续该 lane
后续任务，最终统一返回失败状态。

- [ ] **Step 4: 实现测试汇总**

从 `scheduler_logs/${market}_seed${seed}_gpu${gpu}.log` 提取：

```text
TEST REPORT
Controller eval exit_prob
Switches / Switch detail
Total Ret / Ann Ret / Ann Vol
Sharpe / Max DD
```

写入 `${OUTPUT_ROOT}/test_results_summary.txt`。

- [ ] **Step 5: 运行 GREEN 验证**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  -q tests/test_full_cmtflow_seed_sweep_4gpu_script.py -k scheduler
bash -n train_sh/run_full_cmtflow_seed_sweep_4gpu.sh
```

Expected: 测试 PASS，shell exit 0。

### Task 3: 回归和完整 dry-run

**Files:**
- Verify: `train_sh/train_full_cmtflow_seed.sh`
- Verify: `train_sh/run_full_cmtflow_seed_sweep_4gpu.sh`
- Verify: `tests/test_full_cmtflow_seed_sweep_4gpu_script.py`

- [ ] **Step 1: 运行新测试全集**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  -q tests/test_full_cmtflow_seed_sweep_4gpu_script.py
```

- [ ] **Step 2: 运行相关训练入口回归**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_dual_branch.py \
  tests/test_full_cmtflow_seed_sweep_4gpu_script.py
```

- [ ] **Step 3: 验证 11 个任务和四条 lane**

```bash
DRY_RUN=1 bash train_sh/run_full_cmtflow_seed_sweep_4gpu.sh
```

Expected: 11 个任务、GPU 0--3 各一条 lane，无 GPU 训练进程启动。

- [ ] **Step 4: 检查格式与工作区**

```bash
git diff --check
git status --short
```

按用户要求不执行 commit 或 push。
