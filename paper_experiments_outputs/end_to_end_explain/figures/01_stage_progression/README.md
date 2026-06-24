# Stage Progression 实验说明

## 这个实验是做什么的

该实验比较不同训练阶段 checkpoint 的测试期表现：Fixed HRL checkpoint 和 Controller-PG checkpoint。它用于回答一个整体问题：引入 controller 之后，模型是否从固定持仓节奏变成更有效的动态切换策略。

## 图怎么看

- `fig01_stage_progression_cumulative_*.pdf`：累计收益路径，纵轴是归一化组合价值。曲线越高，测试期收益越好。
- `fig02_stage_progression_drawdown_*.pdf`：回撤路径，纵轴是 drawdown。曲线越低，说明同一时期内风险暴露越小。
- `fig02b_stage_progression_bar_*.pdf`：阶段指标柱状图，集中比较 total return、Sharpe、max drawdown 和 switch count。

图中 Fixed HRL 表示固定切换节奏的层级策略；Controller-PG 表示启用 controller 后的策略。

## 主要结论

NAS 市场中，Fixed HRL 的总收益为 228.19%，最大回撤为 31.73%；启用 controller 后总收益升至 266.37%，最大回撤降至 18.62%。因此 NAS 上 controller-enabled 阶段同时改善收益和回撤。

SH 市场中，Fixed HRL 的总收益为 155.37%，Sharpe 为 0.98；启用 controller 后总收益升至 200.73%，Sharpe 升至 1.12。SH 上主要体现为收益和风险调整收益提升，最大回撤不作为该图的核心优势来强调。

## 论文中怎么用

这组图适合放在附录或实验总览部分，用来说明训练阶段带来的整体性能变化。主文若篇幅有限，优先使用推理消融和 case window 图。
