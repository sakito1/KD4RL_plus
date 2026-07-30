# Paper Experiments Final

本目录汇总 eval-only 论文实验图，覆盖 CSI-300、Nasdaq-100 两个市场。所有图同时保存为 PNG 和 PDF；图中文字使用 DejaVu Sans 并设置 TrueType PDF 字体嵌入，避免 PDF 字体乱码。

目录：
- `EXPERIMENT_REQUIREMENT_AUDIT.md`：逐条复核用户要求与当前输出文件的对应关系。
- `FIGURE_INTERPRETATION.md`：论文图怎么读、说明了什么、哪些结论能说以及哪些结论需要谨慎表述。
- `01_main_experiment/`：Ours vs matched baselines。
- `02_ablation/`：Outer、Inner、Controller 与固定窗口切仓消融。
- `03_controller_interpretability/`：controller switch case、固定持仓窗口对比、反事实收益分布、概率共振。
- `04_inner_actor_interpretability/`：inner actor 收益贡献、换仓/收益共振、switch 时候选权重分布。
- `tables/`：所有实验对应的 CSV 指标。
