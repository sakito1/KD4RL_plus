# AAAI-27 论文导向可复现代码清单

审计日期：2026-07-29  
论文：*Controller-Manager-Trader: Role-Decoupled Hierarchical Reinforcement Learning for Portfolio Management*  
论文文件：`Controller-Manager-Trader_AAAI.pdf`  
论文 SHA256：`a8e3d7817d34ca5a41c6ac5b45c5352d514182c0ecf35d275ed0a36549aea289`

## 0. 本清单的判断原则

本清单只服务于上述 PDF。判断某份代码是否需要提供时，按以下顺序：

1. 能否复现论文方法中的 Controller、Manager、Trader 和五阶段训练；
2. 能否复现论文 Table 1、Table 2、Figure 3、Figure 4；
3. 能否核对最终 checkpoint、seed、数据切分、手续费和指标；
4. 是否满足 AAAI Code and Data Supplement 的基本使用要求。

代码可以存在不影响论文结果的小瑕疵，例如命名不统一、少量重复代码、注释不完整、
警告信息、非核心单元测试不足或目录不够整洁。无需为了投稿全面重构仓库。

以下问题不能视为“小瑕疵”：

- 使用了错误模型或错误 seed；
- 不能得到论文表中的数值；
- 训练集、验证集或测试集划分与论文不一致；
- 手续费口径与论文不一致且未说明；
- 缺少生成论文表格或图片的代码；
- 代码依赖作者机器上的绝对路径，换一台机器无法运行；
- 使用测试期未来信息进行训练或模型选择。

完整仓库审计和非核心工程建议另见
`AAAI27_REPRODUCIBILITY_REPO_AUDIT.md`，不应把其中所有建议都当作投稿前阻塞项。

## 1. 论文需要复现的最小证据闭环

| 论文内容 | 复现目标 | 必须提供的材料 |
|---|---|---|
| 方法与五阶段训练 | 能从数据构建 CMTFlow 并解释每阶段参数更新范围 | 模型、环境、训练入口、阶段配置、运行命令 |
| Table 1 | 重算 CMTFlow 与 7 个论文 baseline 的 TR/SR/MDD/CR | 完整测试 trace、指标代码、baseline 来源 |
| Table 2 | 重算角色消融和 Fix-5/10/20/60d | 消融配置、固定周期评估代码、结果 trace |
| Figure 3 | 重画两个 Controller reconstruction case | exact case manifest、两条反事实曲线、绘图代码 |
| Figure 4 | 重画两个 Trader refinement case | exact window manifest、权重调整、未来相对收益、绘图代码 |
| 论文附录中的实验 | 只复现最终实际提交的附录内容 | 对应脚本、输入和输出；未提交的探索实验不必提供 |

## 2. 最终模型和论文结果：已经冻结

### 2.1 模型身份

| 市场 | Seed | 最终 checkpoint SHA256 | 说明 |
|---|---:|---|---|
| Nasdaq-100 | 49 | `e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d` | joint-finetune 版本 |
| CSI-300 | 90 | `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94` | 用户确认的“CSI 240 模型” |

“CSI 240”用于识别 checkpoint。上述 CSI 模型在原始 0.005% 口径下的 TR
为 240.13%，但指定 PDF 使用 0.01% 手续费，因此论文 Table 1 和 Table 2
报告 237.01%。

### 2.2 PDF 中的最终 CMTFlow 数值

| 市场 | 手续费 | TR | SR | MDD | CR |
|---|---:|---:|---:|---:|---:|
| Nasdaq-100 | 0.01% | 262.49% | 1.14 | 18.66% | 1.41 |
| CSI-300 | 0.01% | 237.01% | 1.24 | 22.91% | 1.18 |

这些数值是代码包的最终验收目标。旧的 265.53/204.99 论文线不再作为本次
提交依据。

### 2.3 必须随 checkpoint 一起提供

- [x] 两个最终 `best_model.pth` 的真实文件，不能使用指向作者目录的软链接。
- [x] 两个 `seed_<n>_command.json`。
- [x] CSI-300 的 `hrl_fixed_best.pth` 和 `controller_best.pth`。
- [x] Nasdaq-100 最初加载的 frozen HRL：
  SHA256 `c336325d91e0cd66491bdfc9bfa9dd2262fde705096a2154abcc592b25a9d03b`。
- [x] Nasdaq-100 的 `controller_best.pth` 和最终运行中的阶段 checkpoint。
- [x] `MODEL_MANIFEST.json`，记录文件名、SHA256、市场、seed、阶段和上游
  checkpoint。
- [ ] 最终测试 action trace、portfolio trace 和 metrics CSV。

不要求把每个调试 checkpoint 都放入 ZIP。

## 3. 手续费口径：训练保持原流程，回测以论文为准

已确认的口径为：

| 阶段 | 手续费 | 代码值 | 用途 |
|---|---:|---:|---|
| 原训练流程 | 0.005% | `0.00005` | 保持已训练模型的原始流程，不重新训练 |
| 论文回测与表格 | 0.01% | `0.0001` | 复现 PDF 中的 262.49% 和 237.01% |

代码包需要：

- [x] 保留原训练配置，不把 `utils/config*.py` 全局改成 `0.0001`。
- [x] 提供独立的 eval/replay 手续费参数。
- [x] 提供生成 0.01% 论文数值的实际脚本：
  `paper_experiments/analyze_transaction_cost_sensitivity.py`；新验证结果位于
  `reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/`。
- [x] 在 README 说明 0.01% 是论文回测口径，训练 checkpoint 沿用原流程。
- [ ] Table 1 和 Table 2 的所有可比方法使用一致的 0.01% 回测定义。

这里优先保证论文数值可复算。无需因为训练和最终回测使用不同手续费而重新跑训练，
但必须在 README 中明确区分两种口径。

## 4. CMTFlow 方法和训练代码：必须提供

### 4.1 核心源码

- [x] `run_hrl_training.py`
- [x] `Components/PPO_model.py`
- [x] `Train/PPO_train.py`
- [x] `Train/controller_pg.py`
- [x] `agent/PPO_agent.py`
- [x] `agent/__init__.py`
- [x] `env/PPO_env.py`
- [x] `env/__init__.py`
- [x] `utils/PriceMatrix.py`
- [x] `utils/Log.py`
- [x] `utils/config.py`
- [x] `utils/config_Nas.py`
- [x] `utils/config_SH.py`

如果上述文件还导入同目录中的其他本地模块，应连同依赖一起提供。可以直接保留
相关源码目录，不要求为精简 ZIP 而冒险删除运行依赖。

### 4.2 五阶段训练与运行记录

论文明确描述：

1. Manager Warm-up；
2. Trader Warm-up；
3. Manager–Trader Portfolio Generator Stabilization；
4. Controller Training；
5. End-to-end Alignment。

代码包必须让 README、shell 命令和实际实现能够映射到这五个阶段：

- [x] `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`
- [x] Nasdaq-100 frozen-HRL 上游训练命令及其 command JSON。
- [x] CSI-300 seed 90 最终 command JSON。
- [x] 每阶段训练哪些模块、冻结哪些模块、epoch/episode 数和学习率；见复现包
  `MODEL_PROVENANCE.md` 及三份 command JSON。
- [ ] checkpoint 选择依据，以及验证集与测试集的边界。

允许 Nasdaq-100 的五阶段由“上游 HRL 运行 + 后续 Controller/joint 运行”组成，
不要求强行改写成一个单命令；只要 README 给出正确的执行顺序即可。

## 5. 数据与预处理：必须提供

论文数据口径：

| 市场 | 股票数 | 论文总体范围 | 代码中的 train/valid/test |
|---|---:|---|---|
| Nasdaq-100 | 39 | 2000-04 至 2025-10 | train 至 2017-12-29；valid 2018-01-02 至 2020-04-22；test 2020-04-23 至 2025-10-03 |
| CSI-300 | 53 | 2000-04 至 2025-02 | train 至 2017-12-28；valid 2018-01-02 至 2019-12-31；test 2020-01-02 至 2025-02-28 |

需要提供：

- [x] 两个最终股票池列表，并保证顺序固定。
- [ ] 价格数据字段说明、复权方式和缺失值处理。
- [ ] 从原始 CSV 生成模型输入的预处理代码。
- [ ] 模型使用的 SSM state 文件，或生成这些 state 的完整代码与 checkpoint。
- [ ] 数据切分代码和因果归一化实现。
- [ ] `data_manifest.json`，记录每个数据文件的 SHA256、日期范围和股票数。
- [ ] 若数据许可证不允许放入 ZIP，提供合法的小样本和本地数据放置说明；同时明确
  小样本只能做 smoke test，不能复现完整论文数值。

## 6. Table 1：主结果代码

论文比较项只有：

- Buy&Hold
- Markowitz
- OLMAR
- UCRP
- AlphaStock
- DeepAries
- DeepTrader
- CMTFlow

因此不需要为了本论文打包 Anticor、WMAMR、SSM 等未进入 Table 1 的实验代码。

必须提供：

- [x] `paper_experiments/metrics.py`
- [x] `paper_experiments/trace_utils.py`
- [x] `paper_experiments/generate_baseline_matched.py`
- [x] Table 1 指标/生成代码。
- [ ] 每个 baseline 的最终 daily portfolio/value trace，或能产生该 trace 的
  checkpoint、代码和命令。
- [ ] `baseline_manifest.csv`：方法、来源、seed、数据切分、手续费、checkpoint
  和 trace SHA256。
- [x] `expected/table1.csv`，保存 PDF 中全部数值，供一键核对。

第三方 baseline 源码不必全部重构。只要能说明版本和修改，并能从提供的 trace
稳定重算论文指标即可。

## 7. Table 2：消融和固定周期代码

需要复现以下行：

- `w/o C+T`
- `w/o C`
- `w/o T`
- `Fix-5d`
- `Fix-10d`
- `Fix-20d`
- `Fix-60d`
- `CMTFlow`

必须提供：

- [x] `paper_experiments/eval_end_to_end_explain.py`
- [x] `paper_experiments/table_end_to_end_explain.py`
- [x] `paper_experiments/run_paper_experiments_final.py`
- [x] 根目录 `run_paper_experiments_final.py`
- [ ] 每个 variant 的定义、运行参数和结果 trace。
- [x] 固定周期 5/10/20/60 天的评估实现。
- [x] `expected/table2.csv`，保存 PDF Table 2 的全部数值。

无需为论文未报告的所有内部 ablation 提供代码。

## 8. Figure 3：Controller case study

PDF 使用的两个 case 为：

- CSI-300：决策日 2021-07-07，图示区间 2021-07-08 至 2021-08-18；
- Nasdaq-100：决策日 2021-04-19，图示区间 2021-04-20 至 2021-06-01。

必须提供：

- [x] `paper_experiments/run_paper_experiments_final.py` 中 Controller case
  生成与绘图部分。
- [x] `selected_controller_cases_sh.csv`
- [x] `selected_controller_cases_nas.csv`
- [x] 两个 case 的 retain/reconstruct return 和 drawdown 曲线。
- [x] 固定 case manifest 记录 seed、决策日、图示区间和最终 case ID。
- [x] 一条命令可重新生成 Figure 3；见
  `paper_experiments/render_aaai27_figure3.py`。

现有 `controller_case_combined_sh01_nas01` 与 PDF 的 Figure 3 对应，应将输入
CSV 和图片输出一起纳入复现包。

## 9. Figure 4：Trader refinement case study

PDF 显示：

- CSI-300：2024-01-23 至 2024-03-12；
- Nasdaq-100：2024-05-13 至 2024-06-25。

必须提供：

- [x] `paper_experiments/plot_inner_actor_base_adjustment.py`
- [x] 两个窗口的 base weight、executed weight、refinement tilt。
- [x] 相同窗口内的未来 5 日相对收益。
- [x] 固定 case manifest，记录日期、股票顺序和选择 seed。
- [x] 一条命令可重新生成 Figure 4。

当前 `inner_daily_stats_paper_selected` 自动输出的 Nasdaq 窗口是
2024-06-04 至 2024-07-17，与 PDF 不同。最终代码包必须锁定 PDF 使用的
2024-05-13 至 2024-06-25 窗口；这属于论文复现输入不一致，不能仅作为画图
样式小瑕疵忽略。

## 10. 最小运行环境与说明文件

必须提供：

- [ ] 根目录 `README.md`：安装、数据放置、训练、评估、Table 1、Table 2、
  Figure 3、Figure 4 的命令。
- [x] 锁定版本的 `requirements.txt` 或 `environment.yml`。
- [x] `EXPECTED_RESULTS.md`：允许的数值误差。
- [x] `MANIFEST.json`：代码、checkpoint 和关键输出的 SHA256。
- [ ] 主代码许可证及第三方 baseline 许可证说明。
- [x] 至少一个无需完整训练的 eval-only 命令。

推荐但不阻塞：

- `Dockerfile`
- 完整单元测试覆盖
- 5–10 分钟 CPU smoke test
- 全仓库类型检查、格式化和重构

## 11. 单 seed 与统计实验的处理

论文最终使用 Nasdaq-100 seed 49 和 CSI-300 seed 90。代码包应如实提供：

- 实际独立运行次数；
- seed 候选和选择规则；
- 最终报告的两个 seed；
- 已经出现在最终补充材料中的统计分析代码。

不要求为了整理代码包临时新增五个 seed 或重新完成大规模训练。缺少多 seed
结果是实验报告限制，应在 AAAI 可复现性清单中如实回答，不能用不存在的结果填充。

## 12. 可以接受的代码问题

只要一键复现链路可运行，以下问题不阻塞投稿：

- 文件名或变量名沿用旧术语；
- 某些模块较长或存在重复逻辑；
- 非核心 exploratory scripts 不够整洁；
- 日志中存在不影响结果的 warning；
- 只有关键路径测试，没有全面测试覆盖；
- 未提供 Docker，但环境版本和安装步骤完整；
- 注释没有覆盖每一行。

建议修复但可以降级处理：

- 少量绝对路径：在打包版入口中改为相对路径或命令行参数即可；
- 多个历史训练脚本：README 只标明最终使用的脚本；
- 无关输出目录较多：不放入 ZIP 即可，无需清理原仓库。

## 13. 论文导向的最终 ZIP 结构

```text
CMTFlow_AAAI27/
├── README.md
├── EXPECTED_RESULTS.md
├── MANIFEST.json
├── requirements.txt
├── src/
│   ├── run_hrl_training.py
│   ├── Components/
│   ├── Train/
│   ├── agent/
│   ├── env/
│   └── utils/
├── scripts/
│   ├── train/
│   ├── verify_package.py
│   └── render_expected_tables.py
├── configs/
├── data/
│   ├── pool_lists/
│   ├── manifests/
│   └── representative_smoke_subset/
├── checkpoints/
│   ├── nas_seed49/
│   └── sh_seed90/
├── traces/
│   ├── cmtflow/
│   ├── ablations/
│   └── baselines/
├── expected/
│   ├── table1.csv
│   ├── table2.csv
│   ├── figure3/
│   └── figure4/
└── third_party/
    └── licenses/
```

## 14. 投稿前优先级

### P0：直接影响论文复现

- [x] 锁定指定 PDF 及其 SHA256。
- [x] 锁定 joint-finetune 模型；CSI-300 使用“240 checkpoint”。
- [x] 锁定论文回测费率 0.01% 和最终 TR 262.49/237.01。
- [x] 补齐生成论文 0.01% 数值的实际 `.py` 脚本。
- [x] 将所需 checkpoint 复制为真实文件并生成 SHA256 manifest。
- [x] 生成 `expected/table1.csv` 和 `expected/table2.csv`。
- [x] 锁定 Figure 3 exact case manifest。
- [x] 修正 Figure 4 Nasdaq case manifest，使日期与 PDF 一致。
- [ ] 提供从相对路径运行的四条命令：Table 1、Table 2、Figure 3、Figure 4。

### P1：AAAI 提交包可使用性

- [x] README 和环境锁定文件。
- [ ] 数据/预处理说明及许可处理。
- [ ] baseline 版本与 trace manifest。
- [ ] 主代码和第三方许可证说明。
- [x] 匿名化绝对路径、用户名和元数据。

### 非阻塞工程项

- [ ] Docker
- [ ] 全面单元测试
- [ ] 全仓库重构和统一命名
- [ ] 未进入论文的 exploratory experiments 整理

当前最重要的工作不是修复所有代码瑕疵，而是确保指定 PDF、两个最终 checkpoint、
0.01% 论文回测、两张表和两张解释性图形成同一条可执行复现链。
