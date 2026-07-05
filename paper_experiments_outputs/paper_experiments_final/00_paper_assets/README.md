# Paper Figure Assets

这个目录是论文图片素材的整理入口。原始实验输出仍保留在 `01_main_experiment/`、`02_ablation/`、`03_controller_interpretability/`、`04_inner_actor_interpretability/` 四个目录中；这里复制一份便于论文写作、PPT 重画和投稿打包使用的素材，不影响后续复现实验。

## 目录说明

- `used_in_paper/`
  当前 `paper/anonymous-submission-latex-2026.tex` 实际引用的图，按论文出现顺序重命名。写论文和最终投稿时优先看这里。

- `experiment_figures/`
  四个实验文件夹中的全部 PNG/PDF 图，按实验分组，并进一步分成 `png/` 和 `pdf/`。这些图包含论文最终选用图以及备选/诊断图。

- `experiment_tables/`
  四个实验文件夹中的 CSV/MD 表格和说明文字。用于核对图中数值、写实验结论、追踪 case 选择。

- `editable/`
  Figure 1 和 Figure 2 的可编辑 SVG，以及图例说明。适合导入 PPT、Illustrator、Figma 或 Inkscape 后重画。

- `ppt_assets/`
  PPT 版本使用的 PNG 素材，来自 `paper/cmtflow_codex_ppt/assets/figures/`。

## 当前论文实际使用的图

| Paper figure | File in this folder | Original LaTeX source |
|---|---|---|
| Framework modules | `used_in_paper/fig01_cmtflow_three_modules.pdf` | `paper/figures/cmtflow_architecture_vector.pdf` |
| Daily decision flow | `used_in_paper/fig02_cmtflow_decision_flow.pdf` | `paper/figures/cmtflow_decision_flow_imagegen.pdf` |
| Nasdaq wealth curve | `used_in_paper/fig03_main_equity_nas.pdf` | `paper/figures/main_equity_nas.pdf` |
| CSI-300 wealth curve | `used_in_paper/fig04_main_equity_sh.pdf` | `paper/figures/main_equity_sh.pdf` |
| Main metric bars | `used_in_paper/fig05_main_metric_bars.pdf` | `paper/figures/main_metric_bars.pdf` |
| Nasdaq controller case | `used_in_paper/fig06_controller_switch_case_nas.png` | `paper/figures/explainability/controller_switch_case_nas.png` |
| CSI-300 controller case | `used_in_paper/fig07_controller_switch_case_sh.png` | `paper/figures/explainability/controller_switch_case_sh.png` |
| Nasdaq learned vs fixed windows | `used_in_paper/fig08_fixed_window_nas.png` | `paper/figures/explainability/fixed_window_comparison_nas.png` |
| CSI-300 learned vs fixed windows | `used_in_paper/fig09_fixed_window_sh.png` | `paper/figures/explainability/fixed_window_comparison_sh.png` |
| Nasdaq inner actor case | `used_in_paper/fig10_inner_actor_nas.png` | `paper/figures/explainability/inner_actor_nas.png` |
| CSI-300 inner actor case | `used_in_paper/fig11_inner_actor_sh.png` | `paper/figures/explainability/inner_actor_sh.png` |

## 四个实验文件夹如何读

- `experiment_figures/01_main_experiment/`
  主实验图：ours vs matched baselines 的收益曲线和指标柱状图。

- `experiment_figures/02_ablation/`
  消融实验图：Outer-only、Outer + Inner、Outer + Controller、fixed-window controllers 和 full model。

- `experiment_figures/03_controller_interpretability/`
  Controller 可解释图：switch case、fixed-window comparison、switch/hold 反事实分布、exit probability resonance。

- `experiment_figures/04_inner_actor_interpretability/`
  Inner actor 可解释图：inner alpha、权重堆叠、base adjustment 与未来相对收益共振、weight-price resonance。

## 使用建议

- 论文 LaTeX 编译仍然使用 `paper/figures/`，不要直接改成引用本目录，避免破坏现有排版。
- 投稿打包或给 PPT 重画时，优先从 `used_in_paper/` 和 `editable/` 拿素材。
- 如果重跑实验，先更新四个原始实验目录，再重新同步本目录。
