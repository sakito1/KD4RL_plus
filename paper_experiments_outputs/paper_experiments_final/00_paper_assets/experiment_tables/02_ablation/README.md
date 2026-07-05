# Ablation Experiment

这个实验拆开组件：Outer-only、Outer + Inner、Outer + Controller、不同固定窗口切仓，以及 Ours。`ablation_equity_*.png` 展示各变体累计财富，`ablation_metrics_*.png` 展示收益、Sharpe、最大回撤和 CR。

读图时比较三条主线：Outer + Inner 相对 Outer-only 体现 inner actor 的边际贡献；Outer + Controller 相对 Outer-only 体现 controller 的动态切仓贡献；Ours 同时结合 inner 和 controller。固定窗口曲线用于说明 controller 不是简单固定周期切仓。
