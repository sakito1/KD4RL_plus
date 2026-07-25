# Controller Guidance Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 在不训练 Controller 的情况下，用 seed 77 Outer checkpoint 和训练集生成 300 日双信号 Top-20 标签，并输出可审计的中文统计报告。

**Architecture:** 将标签生成和类别平衡 BCE 实现为独立纯函数；Trainer 只负责复用现有反事实路径收集风险与优势目标；`run_hrl_training.py` 增加 probe-only 入口，加载 Outer 后直接分析并退出。正式 Controller loss 暂不接入该标签。

**Tech Stack:** Python 3.10、PyTorch、NumPy、pandas、pytest。

---

### Task 1: Top-20 标签纯函数

**Files:**
- Create: `Train/controller_guidance.py`
- Create: `tests/test_controller_guidance.py`

- [ ] **Step 1: 写标签生成失败测试**

测试覆盖：

```python
def test_topk_requires_positive_advantage_and_never_exceeds_budget():
    risk = torch.tensor([0.9, 0.1, 0.8, 0.2])
    advantage = torch.tensor([-0.5, 0.1, 0.2, 0.3])
    result = build_topk_guidance_labels(risk, advantage, topk=2)
    assert result.labels.sum().item() == 2
    assert torch.all(advantage[result.labels.bool()] > 0)

def test_topk_does_not_fill_with_negative_advantages():
    risk = torch.tensor([0.9, 0.8, 0.7])
    advantage = torch.tensor([-0.1, 0.2, -0.3])
    result = build_topk_guidance_labels(risk, advantage, topk=20)
    assert result.labels.tolist() == [0.0, 1.0, 0.0]
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q tests/test_controller_guidance.py
```

Expected: FAIL，`Train.controller_guidance` 不存在。

- [ ] **Step 3: 实现稳定百分位排名和 Top-20 标签**

提供：

```python
@dataclass
class GuidanceLabels:
    labels: torch.Tensor
    risk_percentile: torch.Tensor
    advantage_percentile: torch.Tensor
    priority: torch.Tensor
    advantage_only_labels: torch.Tensor

def build_topk_guidance_labels(
    risk: torch.Tensor,
    advantage: torch.Tensor,
    *,
    topk: int = 20,
) -> GuidanceLabels:
    risk = risk.detach().view(-1).float()
    advantage = advantage.detach().view(-1).float()
    if risk.shape != advantage.shape:
        raise ValueError("risk and advantage must have identical shapes")
    risk_percentile = stable_percentile_rank(risk)
    advantage_percentile = stable_percentile_rank(advantage)
    priority = torch.maximum(risk_percentile, advantage_percentile)
    eligible = torch.isfinite(risk) & torch.isfinite(advantage) & (advantage > 0)
    labels = stable_topk_mask(priority, eligible, topk=topk).float()
    advantage_only_labels = stable_topk_mask(
        advantage_percentile, eligible, topk=topk
    ).float()
    return GuidanceLabels(
        labels=labels,
        risk_percentile=risk_percentile,
        advantage_percentile=advantage_percentile,
        priority=priority,
        advantage_only_labels=advantage_only_labels,
    )
```

要求：

- 过滤非有限值；
- 仅允许 `advantage > 0`；
- `priority=max(risk_percentile, advantage_percentile)`；
- 稳定降序选择至多 `topk`；
- 同时返回仅按 advantage 排序的对照标签。

- [ ] **Step 4: 运行标签测试并确认通过**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q tests/test_controller_guidance.py
```

Expected: PASS。

### Task 2: 类别平衡 BCE

**Files:**
- Modify: `Train/controller_guidance.py`
- Modify: `tests/test_controller_guidance.py`

- [ ] **Step 1: 写类别平衡和梯度方向失败测试**

```python
def test_balanced_bce_gives_equal_class_mass_and_correct_gradients():
    logits = torch.zeros(5, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    loss = balanced_guidance_bce(logits, labels)
    loss.backward()
    assert torch.isclose(loss, torch.tensor(math.log(2.0)))
    assert logits.grad[0] < 0
    assert torch.all(logits.grad[1:] > 0)
    assert torch.isclose(logits.grad[0].abs(), logits.grad[1:].sum())
```

- [ ] **Step 2: 运行测试并确认函数缺失导致失败**

- [ ] **Step 3: 实现正负类分别平均的 BCE**

```python
def balanced_guidance_bce(logits, labels):
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    class_means = []
    if torch.any(labels > 0.5):
        class_means.append(raw[labels > 0.5].mean())
    if torch.any(labels <= 0.5):
        class_means.append(raw[labels <= 0.5].mean())
    if not class_means:
        raise ValueError("balanced_guidance_bce requires at least one label")
    return torch.stack(class_means).mean()
```

- [ ] **Step 4: 运行测试并确认通过**

### Task 3: Trainer 历史信号采集与报告

**Files:**
- Modify: `Train/PPO_train.py`
- Modify: `tests/test_controller_guidance.py`

- [ ] **Step 1: 写汇总统计失败测试**

构造两个合成窗口，验证：

- 每窗标签数不超过 Top-K；
- Switch 优势全部为正；
- 输出包含标签比例、优势均值、风险/优势主导数、相邻间隔和双方案重合率；
- 报告明确标记数据范围为 train。

- [ ] **Step 2: 运行测试并确认汇总函数缺失**

- [ ] **Step 3: 实现 `analyze_controller_guidance_labels`**

流程：

1. 使用 `_controller_train_start_pool(300)` 获取训练集窗口；
2. 截取前 12 个固定窗口；
3. 使用 `_run_controller_aux_fixed_windows` 采集 `target_mdd` 和
   `switch_advantage`；
4. 当 `controller_pg_disable_inner=True` 时，Hold/Switch 执行权重均绕过
   Inner，保证探测的是 Outer + Controller；
5. 每个窗口独立生成 Top-20 标签；
6. 写出：
   - `controller_guidance_probe.csv`
   - `controller_guidance_probe.md`

报告只描述标签结构，不进行模型参数更新。

- [ ] **Step 4: 运行汇总测试并确认通过**

### Task 4: Probe-only 命令入口

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `tests/test_run_hrl_training_command.py`

- [ ] **Step 1: 写命令转发与互斥检查失败测试**

验证新参数：

```text
--controller_guidance_probe_only
--controller_guidance_topk 20
--controller_rollout_len 300
```

Probe-only 必须提供 `--frozen_hrl_checkpoint`，加载 Outer/Inner 后调用分析方法，
不进入 Controller 训练和测试。

- [ ] **Step 2: 运行测试并确认参数不存在**

- [ ] **Step 3: 实现 parser、子进程转发和 probe-only 控制流**

输出结果字典包含：

```python
{
    "guidance_probe_only": True,
    "report": str(Path(trainer.run_dir) / "controller_guidance_probe.md"),
    "csv": str(Path(trainer.run_dir) / "controller_guidance_probe.csv"),
}
```

- [ ] **Step 4: 运行命令相关测试并确认通过**

### Task 5: seed 77 离线检测

**Files:**
- Generate: `results/controller_guidance_probe_sh77/controller_guidance_probe_300d_sh77/sh/ppo/seed_77/controller_guidance_probe.md`
- Generate: `results/controller_guidance_probe_sh77/controller_guidance_probe_300d_sh77/sh/ppo/seed_77/controller_guidance_probe.csv`

- [ ] **Step 1: 运行 probe-only 命令**

使用：

```text
checkpoint=results/cmtflow_5stage_sh77/cmtflow_4_2_1_3_1_sh77/
sh/ppo/seed_77/checkpoints/temp_warmup_outer.pth
market=sh
seed=77
rollout_len=300
windows=12
topk=20
disable_inner=true
```

- [ ] **Step 2: 检查报告验收条件**

必须确认：

- 只使用 train split；
- 每窗 Switch 标签不超过20；
- 所有 Switch 标签 advantage 为正；
- CSV 行数等于全部有效自由决策数；
- Markdown 和 CSV 的总计一致。

- [ ] **Step 3: 解释标签合理性**

报告以下证据而不夸大：

- 标签是否达到预期稀疏度；
- 是否主要由连续日期重复占据；
- 被选日期是否具有更高优势或风险；
- 双信号标签与 advantage-only 标签差异；
- 类别平衡 BCE 的基线规模。

### Task 6: 回归验证

**Files:**
- Test: `tests/test_controller_guidance.py`
- Test: `tests/test_controller_counterfactual_pg.py`
- Test: `tests/test_run_hrl_training_command.py`

- [ ] **Step 1: 运行定向测试**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q \
  tests/test_controller_guidance.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_run_hrl_training_command.py
```

- [ ] **Step 2: 运行语法检查**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m py_compile \
  Train/controller_guidance.py Train/PPO_train.py run_hrl_training.py
```

- [ ] **Step 3: 核对工作区**

只报告本任务新增和修改文件，不提交、不推送，保留用户现有其他修改。
