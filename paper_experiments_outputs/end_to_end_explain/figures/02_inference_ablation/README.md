# Inference Ablation 实验说明

## 这个实验是做什么的

该实验是最核心的消融实验。它在同一个测试集上比较四种推理设置，用于拆分 inner、controller 和 outer 的贡献：

- No-Inner Fixed：固定 HRL，但关闭 inner 调整。
- Fixed HRL：固定 HRL，启用 inner。
- Controller+Outer：controller 可以决定是否 switch，并使用 outer 组合动作，但关闭 inner。
- Full Controller：controller、outer 和 inner 全部启用。

## 图怎么看

- `fig03_inference_ablation_cumulative_*.pdf`：看不同设置的累计收益路径。Full Controller 或 Controller+Outer 高于 Fixed HRL，说明主动切换和 outer 选择有效。
- `fig03b_inference_ablation_drawdown_*.pdf`：看不同设置的回撤路径。曲线越低，说明风险控制越好。
- `fig03c_inference_ablation_bar_*.pdf`：最适合主文使用，直接并列展示 total return、Sharpe、max drawdown 和 switch count。

读图时建议先看柱状图，再回到收益路径和回撤路径确认这些指标不是由单一时点偶然造成的。

## 主要结论

NAS 市场中，Fixed HRL 的总收益为 228.19%，最大回撤为 31.73%；Controller+Outer 将总收益提升到 238.28%，最大回撤降至 21.24%；Full Controller 进一步提升到 266.37%，最大回撤降至 18.62%。这说明 controller/outer 是 NAS 回撤控制的主要来源，inner 在 full setting 下进一步提高收益并继续降低回撤。

SH 市场中，Fixed HRL 的总收益为 155.37%，Sharpe 为 0.98；Controller+Outer 将总收益提升到 233.05%，Sharpe 提升到 1.21，是 SH 上收益和 Sharpe 最强的设置。Full Controller 的总收益为 200.73%，Sharpe 为 1.12，最大回撤略低于 Controller+Outer。SH 上应强调 controller/outer 本身已经带来主要收益提升，inner 的作用更偏路径调整和轻微回撤改善。

## 论文中怎么用

这组图建议作为主文图。它能直接支撑“controller 确实有效”和“inner 与 controller/outer 的贡献不同”这两个结论。
