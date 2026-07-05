# Main Experiment

这个实验比较 Ours（HRL + controller）和能够与论文表格对齐的 baseline。`main_equity_*.png` 看累计财富曲线，`main_metrics_*.png` 看总收益、Sharpe、最大回撤和 CR（年化收益率/最大回撤）。

读图时重点看红色 Ours 曲线和柱子：若收益更高且最大回撤更低，说明 controller 参与的完整框架不仅提高收益，也改善了风险控制。SH 的 AlphaStock 因历史 action 被覆盖，进入指标柱状图和表格，但不进入收益曲线图。
