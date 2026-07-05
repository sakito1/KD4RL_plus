# Baseline Matched Results

这个目录只管理和论文表格数值能对应上的 baseline 结果，不混入消融实验。

## 目录

- `nas/curves/`: NAS 市场可用曲线。
- `sh/curves/`: SH 市场可用曲线。
- `log_snippets/`: 用于定位表格数值的关键日志片段。
- `manifest/baseline_sources.csv`: 每个 baseline 的来源、表格指标、重算指标和备注。

## 当前状态

- 可用曲线数量: 17
- 仅指标/暂缺曲线数量: 1
- AlphaStock NAS 对应 seed 46，可用当前 action 曲线。
- AlphaStock SH 对应 seed 72，但当前 `actions/test.csv` 已被后续 seed 覆盖，所以先标记为 `metric_only_missing_curve`。
- DeepTrader 不复制大 checkpoint，只记录 checkpoint 路径并通过 eval-only replay 导出曲线。

## 使用建议

主实验画总收益曲线时优先读取 `curve_status=available` 的记录；若某个 baseline 是
`metric_only_missing_curve`，只用于柱形指标或表格，不要画成收益曲线。
