# Figures 目录说明

本目录按解释实验类型分组保存论文图。默认 `.pdf` 是字体安全版，适合直接插入论文；同名 `.png` 用于快速预览；`_editable.pdf` 保留可编辑矢量文字，仅建议后期改图时使用。

文件名中的 seed 只用于追踪复现实验，不建议写入论文图题或正文结论。

## 文件夹索引

- `01_stage_progression`：训练阶段对比，说明从 Fixed HRL 到 controller-enabled checkpoint 后整体收益/回撤路径如何变化。
- `02_inference_ablation`：推理消融，核心回答 inner、controller+outer、full controller 分别带来什么贡献。
- `03_inner_alpha`：inner module 的局部 alpha 解释，展示 inner 调整在路径层面的收益贡献。
- `04_switch_alignment`：controller exit probability 与 switch advantage 的对齐关系，适合作为辅助解释。
- `05_switch_events`：全体 switch 事件的事件研究，适合说明 switch 行为具有异质性。
- `06_random_switch`：等切换次数随机时机对照，检验 controller 的效果是否来自切换时机，而不是简单多切换。
- `07_case_windows`：精选持仓窗口案例，最适合主文展示 controller 在何时 switch 以及 switch 如何避免收益劣化或回撤。

## 推荐阅读顺序

主文优先使用 `02_inference_ablation` 和 `07_case_windows`。前者给出总体有效性，后者给出可解释案例。`06_random_switch` 可作为 controller 时机有效性的对照证据。`01_stage_progression`、`03_inner_alpha`、`04_switch_alignment`、`05_switch_events` 更适合放附录或作为补充分析。
