## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-06-27
- Verification Status: VERIFIED
- Version Label: data_consistency_audit_v1

## Data Consistency Audit

### Root Cause

这次“对不上”主要不是模型结果变了，而是汇总脚本混用了两种口径：

1. Ours 的 trace 记录了 `portfolio_value_before=1000` 和第 0 天 `daily_simple_return`，但旧的 `compute_financial_metrics()` 只用 `portfolio_value.pct_change()`，会把第 0 天收益漏掉。因此表格中 Ours 被算成 NASDAQ `266.37%`、CSI-300 `200.73%`，而曲线按初始资金 1000 画出来对应 NASDAQ `265.53%`、CSI-300 `204.99%`。
2. CSI-300 AlphaStock 有可匹配的表格指标，但没有可恢复的 seed-72 action trajectory。旧脚本把 `curve_status == available` 同时用于曲线和指标柱状图，导致 AlphaStock 出现在论文表格里，却没有出现在主实验指标 CSV/柱状图里。
3. Controller 解释性图当前使用 learned controller 与 5/10/20/30/60 日固定持仓窗口对比。

### Fixes Applied

- `paper_experiments/metrics.py`：当 trace 中存在 `daily_simple_return`/`portfolio_value_before` 时，用完整日收益序列计算 AR、Vol、Sharpe，并用初始资金计算总收益。
- `paper_experiments/run_paper_experiments_final.py`：主实验指标包含 metric-only baseline；收益曲线仍只画可复现 trajectory subset。
- `paper_experiments/run_paper_experiments_final.py`：Ours、消融和 fixed-window 指标从 trace/cache 重新计算，避免读取旧口径 CSV。
- `paper/anonymous-submission-latex-2026.tex`：同步主表、消融表、正文结论和 fixed-window 解释性图。
- `paper_experiments_outputs/paper_experiments_final/FIGURE_INTERPRETATION.md` 和 `PAPER_EXPERIMENT_CONCLUSIONS.md`：同步结论文字。

### Current Checked Values

主实验 Ours：

| Market | TR | AR | Vol | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|
| NASDAQ | 265.53% | 26.50% | 23.04% | 1.15 | 18.62% |
| CSI-300 | 204.99% | 24.95% | 21.95% | 1.14 | 22.78% |

消融关键值：

| Market | Outer-only TR | Outer + Inner TR | Outer + Controller TR | Ours TR |
|---|---:|---:|---:|---:|
| NASDAQ | 220.42% | 227.43% | 237.50% | 265.53% |
| CSI-300 | 147.05% | 158.99% | 237.77% | 204.99% |

Fixed-window 对比：

| Market | Method | TR | Sharpe | MDD | CR |
|---|---|---:|---:|---:|---:|
| NASDAQ | Ours | 265.53% | 1.15 | 18.62% | 1.42 |
| NASDAQ | Best fixed-window TR | 30d, 227.43% | 1.11 | 31.73% | 0.76 |
| CSI-300 | Ours | 204.99% | 1.14 | 22.78% | 1.09 |
| CSI-300 | Best fixed-window TR | 30d, 158.99% | 0.99 | 20.85% | 1.04 |

### Remaining Note

DeepTrader NASDAQ 的原日志/`result.xlsx` 记录为 `196.52%`，当前可复现 replay curve 重算为 `196.27%`，差异约 `0.25` 个百分点。论文当前采用 replay curve 对应的 `196.27%`，这样表格、柱状图和曲线终点一致；该差异已记录在 `paper_experiments_outputs/baseline_matched/manifest/baseline_sources.csv`。

### Verification

- `python tests/test_paper_experiments.py`：通过。
- `run_paper_experiments_final.py`：已重跑，主实验 20 行，CSI-300 AlphaStock 已进入指标柱状图。
- LaTeX 主表 vs final CSV：OK。
- 消融表 vs final CSV：OK。
- 可画曲线终点 vs final CSV：OK。
- `pdflatex` 编译：通过；无 undefined citation/reference，无 overfull；PDF 字体已嵌入。
