# Final-Model Module Coordination Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** 从当前 Nasdaq-100 seed 49 与 CSI-300 seed 90 最终模型 trace 生成 Controller、Manager--Controller 和 Trader 的中文模块协同统计报告与可复现 CSV。

**Architecture:** 新增一个独立的 replay-analysis 模块，只读取既有 action/portfolio CSV。模块分离事件构建、统计汇总、时间依赖 bootstrap 和中文报告渲染，命令行入口一次生成全部九个产物，不触碰训练、checkpoint 或论文 LaTeX。

**Tech Stack:** Python 3.10、pandas、NumPy、SciPy、pytest、现有 `paper_experiments` 包。

---

## 文件结构

- Create: `paper_experiments/analyze_final_model_module_coordination.py`
  - 解析 trace、构建事件、计算统计、bootstrap、写 CSV 和中文 Markdown。
- Create: `tests/test_final_model_module_coordination.py`
  - 用小型合成 trace 验证所有统计口径和输出确定性。
- Create: `paper_experiments_outputs/final_model_module_coordination/*`
  - 由命令行生成，不手工填写数值。

### Task 1: 事件构建与基础定义

**Files:**
- Create: `tests/test_final_model_module_coordination.py`
- Create: `paper_experiments/analyze_final_model_module_coordination.py`

- [ ] **Step 1: 写 JSON 解析、Trader 修正强度和 shifted volatility 的失败测试**

测试使用6行合成 action/portfolio DataFrame，断言：

```python
events = build_daily_events(actions, portfolio, market="Nasdaq-100", seed=49)
assert events.loc[0, "refinement_l1"] == pytest.approx(0.20)
assert np.isnan(events.loc[2, "recent_volatility_10"])
```

另用12行已知回报序列验证第11行的 volatility 只使用前10行，不包含当前行。

- [ ] **Step 2: 运行聚焦测试并确认 RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_final_model_module_coordination.py -q
```

Expected: FAIL，因为分析模块尚不存在。

- [ ] **Step 3: 最小实现 `parse_vector`、`age_bucket`、`build_daily_events`**

`build_daily_events` 必须：

```python
merged = actions.merge(portfolio[["date", "holding_duration"]], on="date", validate="one_to_one")
merged["refinement_l1"] = merged["inner_tilt_json"].map(
    lambda value: np.abs(parse_vector(value)).sum()
)
merged["recent_volatility_10"] = (
    merged["base_log_return"].shift(1).rolling(10, min_periods=10).std(ddof=1)
)
```

年龄分箱固定为 `0-1`、`2-5`、`6-10`、`11-20`、`21+`。

- [ ] **Step 4: 运行测试并确认 GREEN**

运行 Step 2 命令，预期基础事件测试通过。

### Task 2: Controller 与底仓转换事件

**Files:**
- Modify: `tests/test_final_model_module_coordination.py`
- Modify: `paper_experiments/analyze_final_model_module_coordination.py`

- [ ] **Step 1: 写 chosen-action advantage 符号测试**

构造相同 `switch_advantage_20=0.03` 的两个 free decisions：Switch 行预期
`chosen_action_advantage_20=0.03`，Hold 行预期 `-0.03`。

- [ ] **Step 2: 写 support transition 失败测试**

旧 support 为 `{0,1,2}`，新 support 为 `{1,2,3}`，断言：

```python
assert row["retained_assets"] == 2
assert row["added_assets"] == 1
assert row["removed_assets"] == 1
assert row["support_jaccard"] == pytest.approx(0.5)
```

- [ ] **Step 3: 运行聚焦测试并确认 RED**

运行 Task 1 Step 2 命令，预期缺少 Controller/transition API 导致失败。

- [ ] **Step 4: 实现 `build_controller_events` 与 `build_base_transition_events`**

Controller 事件只保留 `decision_type == "free_decision"`，使用已有
`switch_advantage_20`，若该列缺失则由 switch/hold future return 相减得到。

底仓转换只处理 `is_free_switch == 1` 且存在前一行的事件；support 阈值固定为
`weight > 1e-8`。Weight overlap 为逐资产最小权重之和，L1 distance 为绝对差之和。

- [ ] **Step 5: 运行测试并确认 GREEN**

运行聚焦测试，预期事件构建测试全部通过。

### Task 3: 汇总统计与 moving-block bootstrap

**Files:**
- Modify: `tests/test_final_model_module_coordination.py`
- Modify: `paper_experiments/analyze_final_model_module_coordination.py`

- [ ] **Step 1: 写年龄/波动汇总和概率分箱测试**

断言输出包含固定年龄组顺序和 `vol_q1` 至 `vol_q4`；概率分箱对重复边界使用
rank-based quintiles，保证非空输入总能产生稳定的1--5标签。

- [ ] **Step 2: 写 deterministic circular block bootstrap 失败测试**

对固定数组调用两次：

```python
ci_a = circular_block_bootstrap_ci(x, statistic, block_length=3, reps=200, seed=7)
ci_b = circular_block_bootstrap_ci(x, statistic, block_length=3, reps=200, seed=7)
assert ci_a == ci_b
```

并断言下界不大于点估计、点估计不大于上界。

- [ ] **Step 3: 运行聚焦测试并确认 RED**

运行聚焦测试，预期汇总/bootstrap API 尚未实现。

- [ ] **Step 4: 实现汇总函数**

实现：

- `summarize_controller_decisions`
- `summarize_probability_bins`
- `summarize_base_transitions`
- `summarize_holding_age`
- `summarize_volatility_quartiles`
- `summarize_trader_correlations`

所有表均包含 `market` 与 `seed`。分组统计使用 `count/mean/median/q25/q75`。

- [ ] **Step 5: 实现 circular moving-block bootstrap**

每次从长度为 `block_length` 的循环连续区块中采样，拼接到原样本长度。成对统计
必须以相同索引同时重采样两个变量。使用 `np.nanpercentile(samples, [2.5, 97.5])`。

- [ ] **Step 6: 运行测试并确认 GREEN**

运行聚焦测试，预期所有统计测试通过。

### Task 4: 中文报告和命令行生成器

**Files:**
- Modify: `tests/test_final_model_module_coordination.py`
- Modify: `paper_experiments/analyze_final_model_module_coordination.py`
- Create: `paper_experiments_outputs/final_model_module_coordination/*`

- [ ] **Step 1: 写中文报告与输出文件失败测试**

在临时目录调用 `write_outputs`，断言九个文件全部存在，Markdown 包含：

```text
# 最终模型模块协同统计
## Controller 决策行为
## Manager--Controller 底仓转换
## Trader 跨时间尺度修正
## 可安全用于论文的解释
## 局限
```

- [ ] **Step 2: 运行聚焦测试并确认 RED**

运行聚焦测试，预期报告/CLI 尚未实现。

- [ ] **Step 3: 实现 `render_chinese_report`、`write_outputs` 和 CLI**

CLI 默认参数：

```text
--input-dir paper_experiments_outputs/end_to_end_explain/traces
--output-dir paper_experiments_outputs/final_model_module_coordination
--bootstrap-reps 5000
--block-length 20
--bootstrap-seed 20260717
```

报告中的数字必须从 DataFrame 渲染，不能硬编码。较弱或反向统计使用中性描述，
不输出“显著”一词，除非相应95%区间明确不跨零。

- [ ] **Step 4: 运行测试并确认 GREEN**

运行聚焦测试，预期全部通过。

- [ ] **Step 5: 运行生产统计**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  -m paper_experiments.analyze_final_model_module_coordination
```

Expected: 输出九个文件，并打印两个市场的输入行数与输出目录。

### Task 5: 完整验证

**Files:**
- Verify: `paper_experiments/analyze_final_model_module_coordination.py`
- Verify: `tests/test_final_model_module_coordination.py`
- Verify: `paper_experiments_outputs/final_model_module_coordination/*`

- [ ] **Step 1: 运行聚焦测试**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_final_model_module_coordination.py -q
```

- [ ] **Step 2: 验证生产事件数和有限值**

使用只读 Python 检查：

- Nasdaq-100 daily rows = 1369；
- CSI-300 daily rows = 1247；
- Nasdaq-100 free decisions = 1334；
- CSI-300 free decisions = 1220；
- Nasdaq-100 free switches = 231；
- CSI-300 free switches = 92；
- 汇总表的核心点估计与置信区间均有限。

- [ ] **Step 3: 验证确定性**

连续运行生成器两次，对九个输出执行 `sha256sum`，两次校验和必须相同。

- [ ] **Step 4: 运行差异检查**

```bash
git diff --check -- \
  paper_experiments/analyze_final_model_module_coordination.py \
  tests/test_final_model_module_coordination.py \
  paper_experiments_outputs/final_model_module_coordination \
  docs/superpowers/plans/2026-07-17-final-model-module-coordination-statistics-implementation.md
```

- [ ] **Step 5: 检查工作区范围**

确认未修改 `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`、
训练代码或 checkpoint，并在最终交付中列出生成文件和主要统计结论。
