# Controller 60 日强制切仓批量测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 新增一个纯测试脚本，在四张 GPU 上测试结果目录下所有已完成种子，并仅将 Controller 测试强制切仓上限改为 60 日。

**Architecture:** 扫描训练保存的 `seed_*_command.json`，从相同运行目录定位 `best_model.pth`，再重放原始子进程命令。重放前覆盖测试输出目录与 `controller_eval_max_hold`，并追加 `test_only_checkpoint`，从而保留 checkpoint 所需的其他全部配置。

**Tech Stack:** Bash、Python JSON 标准库、现有 `run_hrl_training.py` 测试入口、pytest。

---

### Task 1: 定义批量测试脚本行为

**Files:**
- Create: `tests/test_full_cmtflow_controller_maxhold60_script.py`
- Create: `train_sh/test_full_cmtflow_controller_maxhold60.sh`

- [ ] **Step 1: 写失败测试**

测试在临时目录构造一个完整运行和一个缺失 checkpoint 的运行，使用
`DRY_RUN=1` 调用目标脚本，并断言输出命令：

```python
assert "--max_hold 30" in result.stdout
assert "--controller_eval_max_hold 60" in result.stdout
assert "--test_only_checkpoint" in result.stdout
assert "seed_44" in result.stdout
assert "seed_45" not in executed_commands
```

- [ ] **Step 2: 运行测试并确认因脚本不存在而失败**

Run:

```bash
pytest -q tests/test_full_cmtflow_controller_maxhold60_script.py
```

Expected: FAIL，指出 `train_sh/test_full_cmtflow_controller_maxhold60.sh`
不存在。

- [ ] **Step 3: 实现最小批量测试脚本**

脚本完成以下处理：

```bash
SOURCE_ROOT="${SOURCE_ROOT:-results/full_cmtflow_seed_sweep_4gpu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/full_cmtflow_test_controller_maxhold60}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
```

对每个 `seed_*_command.json`：

1. 用 Python 标准库读取 `.command` 数组；
2. 从命令文件所在运行目录查找唯一的
   `ppo/seed_*/checkpoints/best_model.pth`；
3. 缺少 checkpoint 时记录为 skipped；
4. 将 `--run_root` 改到独立输出目录；
5. 保持 `--max_hold 30`；
6. 将 `--controller_eval_max_hold` 改为 `60`；
7. 移除已有 `--test_only_checkpoint` 及其值，再追加当前 checkpoint；
8. 默认创建 GPU 0--3 四个并行队列，每个队列内部逐个串行执行；
9. 将每个种子的控制台输出写入单独日志并生成总汇总。

- [ ] **Step 4: 运行目标测试并确认通过**

Run:

```bash
pytest -q tests/test_full_cmtflow_controller_maxhold60_script.py
```

Expected: PASS。

### Task 2: 回归验证与静态检查

**Files:**
- Verify: `train_sh/test_full_cmtflow_controller_maxhold60.sh`
- Verify: `tests/test_full_cmtflow_controller_maxhold60_script.py`

- [ ] **Step 1: Shell 语法检查**

Run:

```bash
bash -n train_sh/test_full_cmtflow_controller_maxhold60.sh
```

Expected: exit code 0。

- [ ] **Step 2: 运行相关测试**

Run:

```bash
pytest -q \
  tests/test_full_cmtflow_controller_maxhold60_script.py \
  tests/test_run_hrl_training_command.py
```

Expected: PASS。

- [ ] **Step 3: 检查工作区范围**

Run:

```bash
git status --short
```

Expected: 仅新增本次脚本、测试与文档；不修改模型和训练实现。
