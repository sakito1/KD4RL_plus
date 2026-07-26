# Controller 多种子三 GPU 训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 从指定 NAS/SH Outer+Inner checkpoint 训练同种子 Controller，在三张 GPU 上每卡并发两个任务，并自动测试和汇总结果。

**Architecture:** 新增一个只负责单个市场—种子训练的参数化 shell 入口，以及一个只负责构建任务、分配六条 lane、收集状态和汇总测试日志的调度入口。训练参数复用已确认的 NAS-45 配置，模型与 Python 训练代码不变。

**Tech Stack:** Bash、Python `pytest`（脚本行为测试）、现有 `run_hrl_training.py`。

---

## 文件结构

- Create: `train_sh/train_controller_from_outer_inner.sh`  
  参数化单任务入口；解析同种子 checkpoint，启动 Controller 训练和测试。
- Create: `train_sh/run_controller_seed_sweep_3gpu.sh`  
  默认 11 个任务，六条 lane 调度，失败收集，测试日志汇总。
- Create: `tests/test_controller_seed_sweep_3gpu_script.py`  
  通过 `DRY_RUN=1` 验证参数、checkpoint、GPU 分配和测试开关。

### Task 1: 单任务入口

- [ ] **Step 1: 写失败测试**

在 `tests/test_controller_seed_sweep_3gpu_script.py` 中运行尚不存在的
`train_controller_from_outer_inner.sh`，传入：

```python
env.update({
    "DRY_RUN": "1",
    "PYTHON_BIN": "/bin/echo",
    "MARKET": "sh",
    "CONTROLLER_SEED": "54",
    "GPU_ID": "2",
})
```

断言输出包含：

```python
assert "--markets sh" in output
assert "--seeds 54" in output
assert "sh/ppo/seed_54/checkpoints/hrl_fixed_best.pth" in output
assert "--controller_guidance_pretrain_coef 1.0" in output
assert "--controller_aux_mdd_coef 0.01" in output
assert "--controller_aux_switch_adv_coef 0.01" in output
assert "--controller_sup_coef 0.01" in output
assert "--skip_test" not in output
```

- [ ] **Step 2: 验证测试因脚本缺失而失败**

Run:

```bash
pytest -q tests/test_controller_seed_sweep_3gpu_script.py -k single
```

Expected: FAIL，原因是单任务脚本不存在。

- [ ] **Step 3: 实现最小单任务脚本**

脚本验证 `MARKET` 为 `nas|sh`，默认 checkpoint：

```bash
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_ROOT}/${MARKET}/ppo/seed_${CONTROLLER_SEED}/checkpoints/hrl_fixed_best.pth}"
```

构造当前已确认的 Controller 参数；正式执行时设置
`CUDA_VISIBLE_DEVICES=$GPU_ID` 并用 `tee` 写任务日志。命令不得包含
`--skip_test`，从而触发 `trainer.test(result["best_ckpt"])`。

- [ ] **Step 4: 验证单任务测试通过**

Run:

```bash
pytest -q tests/test_controller_seed_sweep_3gpu_script.py -k single
bash -n train_sh/train_controller_from_outer_inner.sh
```

Expected: pytest PASS，shell 语法检查 exit 0。

### Task 2: 六 lane 调度

- [ ] **Step 1: 写失败测试**

在同一测试文件中以 `DRY_RUN=1` 运行尚不存在的调度器，断言：

```python
expected = {
    "nas": ["44", "45", "47", "50", "56", "57", "58"],
    "sh": ["44", "46", "49", "54"],
}
for market, seeds in expected.items():
    for seed in seeds:
        assert f"market={market} seed={seed}" in output
assert output.count("lane 0 queue:") == 3
assert output.count("lane 1 queue:") == 3
assert "Concurrent jobs per GPU: 2" in output
```

- [ ] **Step 2: 验证测试因调度器缺失而失败**

Run:

```bash
pytest -q tests/test_controller_seed_sweep_3gpu_script.py -k scheduler
```

Expected: FAIL，原因是调度脚本不存在。

- [ ] **Step 3: 实现六 lane 轮询**

调度器使用默认值：

```bash
NAS_SEEDS="${NAS_SEEDS-44 45 47 50 56 57 58}"
SH_SEEDS="${SH_SEEDS-44 46 49 54}"
JOBS_PER_GPU=2
GPU_IDS=("${GPU0:-0}" "${GPU1:-1}" "${GPU2:-2}")
```

按 `slot = index % 6` 分配任务，`gpu_index = slot % 3`，
`lane = slot / 3`。启动六个后台 `run_queue`，每条 queue 内依次调用
单任务脚本；记录每条 lane 的退出状态。

- [ ] **Step 4: 实现测试结果汇总**

每个任务的固定调度日志位于：

```text
${OUTPUT_ROOT}/scheduler_logs/${market}_seed${seed}_gpu${gpu}.log
```

所有任务结束后，在 `test_results_summary.txt` 中为每个日志写市场、种子，
并提取：

```text
TEST REPORT
Controller eval exit_prob
Switches
Switch detail
Total Ret
Ann Ret
Ann Vol
Sharpe
Max DD
```

- [ ] **Step 5: 验证调度测试通过**

Run:

```bash
pytest -q tests/test_controller_seed_sweep_3gpu_script.py -k scheduler
bash -n train_sh/run_controller_seed_sweep_3gpu.sh
```

Expected: pytest PASS，shell 语法检查 exit 0。

### Task 3: 回归与最终验证

- [ ] **Step 1: 运行新测试全集**

Run:

```bash
pytest -q tests/test_controller_seed_sweep_3gpu_script.py
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行 Controller 脚本相关回归测试**

Run:

```bash
pytest -q \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_dual_branch.py \
  tests/test_controller_report_logging.py \
  tests/test_explore_controller_from_nas45_outer_inner_script.py \
  tests/test_controller_seed_sweep_3gpu_script.py
```

Expected: 全部 PASS；若既有 NAS-45 测试与当前工作树参数不一致，只报告该既有失败，不修改与本功能无关的训练参数。

- [ ] **Step 3: 执行完整 dry run**

Run:

```bash
DRY_RUN=1 bash train_sh/run_controller_seed_sweep_3gpu.sh
```

Expected: 输出 11 个任务、六条 lane、同市场同种子 checkpoint，且不访问 GPU。

- [ ] **Step 4: 检查变更范围**

Run:

```bash
git diff --check
git status --short
```

Expected: 无新增格式错误；只新增本规格、计划、两个脚本和一个测试文件。

本任务按用户要求不执行 `git commit` 或 `git push`。

