# AAAI-27 仓库完整审计记录（内部参考）

> 本文件保留完整仓库审计和工程改进建议，不作为最终提交代码清单。
> 最终范围以 `AAAI27_REPRODUCIBILITY_CODE_CHECKLIST.md` 和指定论文 PDF 为准。

审计日期：2026-07-29  
项目：CMTFlow / KD4RL_plus  
审计对象：当前论文、训练记录、模型 checkpoint、评估与论文作图代码

## 已确认的最终复现口径

用户于 2026-07-29 确认：

- 最终模型采用 **joint-finetune 结果线**；
- CSI-300 采用通常称为“240 版本”的 seed 90 模型，即 checkpoint SHA256 为
  `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94` 的版本；
- 训练保持原流程和原训练参数，不按新的回测费率重新训练；
- 回测/稳健性评估允许使用 **0.01%** 手续费，即代码参数 `0.0001`。

这里必须区分“模型身份”和“回测口径”：

- “CSI 240 版本”用于确定模型 checkpoint；其原始 240.13% 结果是在 0.005% 手续费下得到的。
- 对同一已记录交易路径按 0.01% 手续费重放，当前结果是 237.01%，不是 240.13%。
- 若最终论文把 0.01% 作为主回测口径，应重新运行完整 inference/backtest，让环境、Controller
  决策和交易成本特征都使用 `0.0001`；现有 237.01% 只能称为 fixed-path
  transaction-cost sensitivity，不能冒充完整策略重推理结果。

## 1. AAAI-27 的直接要求

根据 AAAI-27 官方投稿说明：

- 可复现性清单需要与主论文分开上传。
- 代码、数据和复现所需材料应在投稿时提供；“录用后再公开”不能作为当前可复现性的证据。
- Code and Data Supplement 应打包成 ZIP，包含源码、脚本、数据和说明。
- 双盲阶段不得在论文或补充材料中放匿名 GitHub、Hugging Face 等外部补充材料链接。
- 如果完整数据超过上传限制，应在 ZIP 中放代表性子集；但要明确说明该子集只能用于 smoke test，不能复现完整论文数值。
- AAAI-27 主赛道补充材料和代码截止时间是 2026-07-31 23:59 UTC-12。

官方页面：

- <https://aaai.org/conference/aaai/aaai-27/submission-instructions/>
- <https://aaai.org/conference/aaai/aaai-27/supplementary-material/>

## 2. 模型版本冻结：已选择 joint-finetune 结果线

### 2.1 当前论文正文实际对应的旧归档版本

当前论文 `paper/anonymous-submission-latex-2026-aaai-main.tex` 报告：

| 市场 | Seed | TR | Sharpe | MDD | 对应 checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| Nasdaq-100 | 49 | 265.53% | 1.15 | 18.62% | `7152fe3588ac3528e7ae54fafe440aeee516293b6b486c9d1420dbf253f4e55e` |
| CSI-300 | 90 | 204.99% | 1.14 | 22.78% | `8022c8cae48be9232fee9dd00337230b2cd88071587a996b0de73d3fed0e6a42` |

这两个旧模型的实际训练命令均为：

- 从已有 `hrl_fixed_best.pth` 加载 outer/inner；
- `warmup_outer_epochs=0`；
- `warmup_inner_epochs=0`；
- `joint_epochs=0`；
- 使用 `--controller_only_finetune`；
- 只训练 controller，没有执行论文所写的 controller-active outer/inner/controller joint finetune。

因此，这条结果线与论文当前的“五阶段训练 + 最终 joint finetune”表述不一致。

旧版本文件定位：

- NAS command JSON SHA256：`148ee8d31216682557341563ee4c3ce0390af81cb5814d2d2da7a54edcf42ffc`
- SH command JSON SHA256：`b72c990097bafcfc10c749e341cf702f91326c034f49da36e305444ab7d95b72`
- SH 旧模型现位于 `/home/tongwenxuan/KD4RL_plus/results/end/_backups/sh_seed90_before_240_13_20260727/`。

### 2.2 最终选定的 `paper_selected` joint-finetune 版本

`reproduced_inputs/paper_selected/`、inner/outer 统计验证和交易成本敏感性分析使用的是：

| 市场 | Seed | 最终 checkpoint SHA256 | 训练方式 | 原始主评估结果（0.005% 手续费） |
|---|---:|---|---|---|
| Nasdaq-100 | 49 | `e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d` | frozen HRL 初始化 → controller PG → 1 epoch controller-active joint finetune | TR 265.53% |
| CSI-300 | 90 | `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94` | outer warmup → inner warmup → fixed-HRL joint → controller PG → 1 epoch controller-active joint finetune | TR 240.13%，Sharpe 1.246，MDD 22.70% |

精确来源：

#### Nasdaq-100 joint 版本

- Final checkpoint：
  `/home/tongwenxuan/KD4RL_plus/results/controller_first_joint_lowlr_retry_20260622_02/lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/nas/ppo/seed_49/checkpoints/best_model.pth`
- Final command JSON SHA256：
  `42eb55217877c4cdf9a2f351b7a583a44e67efb7b518e3042d3ce9a0e63eb2d3`
- 必须额外提供最初的 frozen HRL checkpoint：
  `/home/tongwenxuan/KD4RL_plus/results/hrl_lookback60_hold30_inner_noaux_retrain/lookback60_hold30_inner_noaux_retrain/nas/ppo/seed_49/checkpoints/hrl_fixed_best.pth`
- Frozen HRL checkpoint SHA256：
  `c336325d91e0cd66491bdfc9bfa9dd2262fde705096a2154abcc592b25a9d03b`
- Frozen HRL command JSON SHA256：
  `9c7b4ecdf1713c307460d065ef70118b384e6db79750fced1ecf8b995a6fd055`

注意：final 目录内再次保存的 `hrl_fixed_best.pth` 不是最初加载的原始文件，不能替代上述 `c336...` 的初始 checkpoint。

#### CSI-300 joint 版本

- Final checkpoint：
  `/home/tongwenxuan/KD4RL_plus/results/e2e_standard_joint_lowlr_20260622_01/lookback60_hold30_standard_joint_lowlr_nas49_sh90/sh/ppo/seed_90/checkpoints/best_model.pth`
- Final command JSON SHA256：
  `08260d953d2a7f29c70e05c4d937887753d5633b84ad899214da1d7a827d7725`
- `hrl_fixed_best.pth` SHA256：
  `6206e0235adf8933c14131d32787412cca7dbfb85f2d260abd20cca94e11e239`
- `controller_best.pth` SHA256：
  `c81a3e9bb8f1103dcad992ed758e936352c227a16ef8399b0072870f0ec55f4f`
- `best_model.pth` SHA256：
  `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94`

### 2.3 已确认的最终版本

最终代码包以 **2.2 的 joint-finetune 版本** 为准。旧的 controller-only
结果线只作为内部审计历史，不应混入最终 checkpoint、主结果 trace 或复现命令。

选择该版本后，必须同步修改论文：

- CSI-300 的 Ours 从 204.99% 更新为 240.13%，并同步 Sharpe、MDD、CR。
- 主表、消融表、正文、图和补充材料必须统一重算。
- Nasdaq-100 要明确说明 fixed-HRL warmup 是先在独立运行中完成，再作为 frozen HRL 初始化进入 controller PG 和 joint finetune；不能写成单个命令从零完成全部阶段。
- 删除或更新正文中 CSI-300 204.99% 的所有旧引用。

不能将旧论文数值、旧 controller-only checkpoint、joint 模型 checkpoint
和新分析结果混在同一个补充包中。

### 2.4 训练与 0.01% 回测的参数边界

原训练配置为：

- `utils/config.py`：`TRANSACTION_COST_RATE=5e-5`
- `utils/config_Nas.py`：`TRANSACTION_COST_RATE=5e-5`
- `utils/config_SH.py`：`TRANSACTION_COST_RATE=5e-5`
- `env/PPO_env.py` 默认从上述配置读取交易成本，并将其用于净值、reward、
  Controller 候选切换优势和 `cost_feat`。

因此复现包应采用以下口径：

| 阶段 | 手续费 | 代码值 | 是否改变策略动作 | 对应 CSI-300 数值 |
|---|---:|---:|---|---:|
| 原训练及原模型选择 | 0.005% | `0.00005` | 是，属于原训练环境 | 不重新训练 |
| 原始主评估 | 0.005% | `0.00005` | 按原环境完整推理 | 240.13% |
| 0.01% 完整回测 | 0.01% | `0.0001` | 应重新执行完整推理 | 待完整回测确认 |
| 0.01% fixed-path 敏感性 | 0.01% | `0.0001` | 否，只重算同一路径成本 | 237.01% |

实现和打包要求：

- 不要把三个训练配置中的全局 `TRANSACTION_COST_RATE` 直接改为 `0.0001`；
  这会改变原训练复现口径。
- 在 eval-only 入口增加独立参数，例如
  `--transaction_cost_pct 0.0001`，并在构造 `PPO_Env` 时显式传入。
- 每个回测输出的 manifest 必须记录 `transaction_cost_pct`、是否重新推理、
  checkpoint SHA256、command JSON SHA256 和 trace SHA256。
- 主表若继续报告 240.13%，表注必须写明主评估费率为 0.005%，并把 0.01%
  结果作为稳健性分析。
- 主表若改用 0.01%，必须用完整回测的新结果替换 240.13%；不能直接用
  fixed-path 的 237.01% 代替完整回测。

## 3. 模型训练源码：必须提供

下面是 CMTFlow 主模型训练和评估所需的最小源码集合。

### 3.1 核心模型与环境

- `run_hrl_training.py`：统一训练、测试和命令记录入口。
- `Train/PPO_train.py`：五阶段训练、controller PG、joint finetune、测试与 checkpoint 选择。
- `Components/PPO_model.py`：outer actor、inner actor、controller 和完整网络。
- `agent/PPO_agent.py`
- `agent/__init__.py`
- `env/PPO_env.py`
- `env/__init__.py`
- `utils/PriceMatrix.py`
- `utils/Log.py`
- `utils/config.py`
- `utils/config_Nas.py`
- `utils/config_SH.py`
- `utils/NAS100_pool.txt`
- `utils/SH_pool.txt`

### 3.2 精确训练入口

- `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`
- `train_sh/run_controller_daily_aux_pg_from_noaux_retrain.sh`
- `train_sh/run_hrl_fixed60_inner_noaux_retrain.sh`
- 如继续保留兼容入口，可一并提供：
  `scripts/run_reproduce_hrl_controller_nas49_sh90.sh`

建议新建两个不含绝对路径的冻结命令：

- `scripts/reproduce_train_nas_seed49.sh`
- `scripts/reproduce_train_sh_seed90.sh`

这两个脚本应逐项复刻最终 command JSON，不再依赖多层环境变量默认值。

### 3.3 训练代码版本

两组 joint 模型训练时，核心训练文件最后一次已提交的共同版本可追溯到：

`0c14a407143282e22168920dae3240cd84272cc7`

但原始 command JSON 没有记录 Git commit，运行时也可能存在未提交改动，所以这只能视为“基于时间和 Git 历史重建的最可能版本”，不是严格证明。

提交前应：

- 建一个匿名 release snapshot/tag；
- 在 `MANIFEST.json` 写入 release commit；
- 对每个核心源码、checkpoint、command JSON、数据文件和结果文件记录 SHA256；
- 保存当前工作树是否 clean；
- 不再用会移动目标的软链接表示“最终模型”。

## 4. 数据与预处理代码：必须提供或明确解释

### 4.1 完整实验输入

主模型实际读取：

- `Dataset/Nas100数据/feature_ssm/`
- `Dataset/沪深数据/feature_ssm/`
- `utils/NAS100_pool.txt`
- `utils/SH_pool.txt`

其中每只股票至少需要：

- `<ticker>.csv`
- `<ticker>_ssm3_states.pt`

按当前股票池，约需：

- Nasdaq-100：39 只股票，未压缩约 446 MiB；
- CSI-300：53 只股票，未压缩约 120 MiB。

不要把全部 1.4 GiB `Dataset/` 原样打包；应只抽取股票池实际使用的文件，并生成 `data_manifest.csv`，记录相对路径、大小、行数、日期范围和 SHA256。

### 4.2 必须随数据提供

- 数据字段说明：价格列、技术特征、`ssm3_p`、`ssm3_q_bear`、`ssm3_q_bull`、hidden state 的 `h/z` 形状和含义。
- 股票筛选规则以及为何最终为 39/53 只。
- train/validation/test 的精确日期边界。
- 因果标准化和缺失值处理规则。
- 原始数据来源、访问日期、许可条件和是否允许再分发。
- 从原始数据到 `feature_ssm` 和 `*_ssm3_states.pt` 的完整预处理脚本。

当前 `archive/data_prep_legacy/` 只能视为历史代码，尚不能证明它就是生成论文输入的精确流水线。提交前需要把真正的数据生成入口单独冻结为：

- `scripts/prepare_data.py`
- `scripts/build_features.py`
- `scripts/export_ssm_states.py`

如果 Wind、Refinitiv 或其他商业数据许可不允许再分发，不能直接上传衍生数据而不检查许可。此时至少要提供：

- 数据源和字段映射；
- 下载/导入接口；
- 完整预处理代码；
- 一个许可允许的代表性小样本；
- 对“为什么公开替代数据在科学上不能等价替代”的说明。

## 5. Checkpoint 与运行记录：必须提供

对每个最终市场/seed，至少提供：

- `best_model.pth`
- `controller_best.pth`
- `hrl_fixed_best.pth`
- 原始初始化 checkpoint；Nasdaq-100 即 SHA256 为 `c336...` 的 frozen HRL。
- `seed_<n>_command.json`
- 完整训练日志和最终测试日志。
- 最终 `test_s3_AllModules.csv`。
- final portfolio trace 和 action trace。
- `checkpoint_manifest.json`，包括：
  - 模型 SHA256；
  - command JSON SHA256；
  - 训练代码 commit；
  - 数据 manifest SHA256；
  - seed；
  - checkpoint 选择指标；
  - 验证集最佳 epoch；
  - 测试期指标。

只提供 `best_model.pth` 不足以复现论文的阶段对比、消融和训练链。

## 6. 评估、表格、作图与统计代码：必须提供

### 6.1 主结果和消融

- `paper_experiments/eval_end_to_end_explain.py`
- `paper_experiments/trace_utils.py`
- `paper_experiments/metrics.py`
- `paper_experiments/run_paper_experiments_final.py`
- 根目录 `run_paper_experiments_final.py`
- `paper_experiments/plot_end_to_end_explain.py`
- `paper_experiments/table_end_to_end_explain.py`

### 6.2 Baseline 对齐

- `paper_experiments/generate_baseline_matched.py`
- `paper_experiments_outputs/baseline_matched/manifest/baseline_sources.csv`
- 每个 baseline 的精确 checkpoint、seed、命令、日志或可复算 daily trace。

当前 baseline 还有版本冲突：

- 论文中的 DeepTrader 数值与当前 replay 数值不一致；
- CSI-300 AlphaStock 只有日志指标，缺少对应 seed-72 action trajectory；
- 当前 `main_experiment_metrics.csv` 已使用新的 replay 值，但论文仍使用旧表值。

在解决这些冲突前，不能把当前 baseline 目录标记为“全部可复现”。

### 6.3 解释性和统计分析

根据最终论文实际保留的图和结论，提供：

- `paper_experiments/plot_inner_actor_base_adjustment.py`
- `paper_experiments/plot_inner_actor_resonance.py`
- `paper_experiments/analyze_inner_outer_statistical_validation.py`
- controller case、counterfactual、fixed-window 与交易成本敏感性对应的脚本。
- `scripts/plot_controller_base_filter_cases.py` 等只有在最终论文/补充材料使用其结果时才放入。

所有分析输出应保存输入 checkpoint hash、输入 trace hash、随机 seed 和代码 commit。现有 `reproduced_outputs/*/metadata/run_manifest.json` 可以作为模板。

交易成本部分当前还要补齐：

- `paper_experiments/eval_end_to_end_explain.py` 当前没有
  `--transaction_cost_pct` 参数，需增加 eval-only 覆盖入口；
- 当前仓库已有
  `reproduced_outputs/fixed_path_transaction_cost_sensitivity/` 的结果和
  manifest，但未发现生成该结果的对应 `.py` 源码；最终 ZIP 必须补入该脚本；
- 完整 0.01% 回测与 fixed-path 0.01% 敏感性必须使用不同的 scenario 名称和
  输出目录，防止误读；
- 对所有方法使用相同手续费口径；不能只对 CMTFlow 改费率而保留 baseline
  的旧费率结果。

## 7. Baseline 源码：按论文实际比较项提供

### 传统/在线策略

- `Baseline/BH/`
- `Baseline/markowitz/`
- `Baseline/ucrp/`
- `Baseline/anticor/`
- `Baseline/olmar/`
- `Baseline/wmamr/`
- `Train/baseline.py`

### 深度学习 baseline

- `AlphaStock/`
- `DeepTrader/src/` 或实际运行使用的 `DeepTrader/DeepTrader/src/`，二者只能保留一个确定版本。
- `DeepAries/`
- `run_deeparies_baseline.py`
- `create_deeptrader_data.py`
- `create_DeepAries_data.py`
- `recompute_deeparies_metrics.py`

每个 baseline 都要说明：

- 原仓库/论文来源和许可证；
- 本项目是否修改过；
- 具体修改 diff；
- seed、超参数、checkpoint 选择规则；
- 是否与 CMTFlow 使用完全相同的数据切分、交易成本和指标公式。

## 8. 运行环境：当前缺失根级冻结文件

主模型当前可检测到的环境为：

- Python 3.10.16
- PyTorch 2.4.0+cu124
- CUDA 12.4
- cuDNN 9.1
- NumPy 2.2.5
- pandas 2.2.3
- matplotlib 3.10.0
- Gym 0.26.2
- SciPy 1.15.3
- scikit-learn 1.6.1
- Ubuntu 24.04.4 LTS
- NVIDIA GeForce RTX 3090 24 GiB
- Intel Xeon Gold 5317，当前分配 6 vCPU
- 当前可见内存 15 GiB

这些是当前机器信息，不等于已经证明是 2026-06-21/22 训练时的精确环境。提交包应新增：

- `environment.yml` 或锁定版本的 `requirements.txt`
- `SYSTEM.md`
- 可选 `Dockerfile`
- `scripts/collect_system_info.sh`

当前只有 `DeepAries/requirements.txt`，不能覆盖 CMTFlow 主模型。

## 9. 随机性、多次运行和统计检验

代码中已经设置 Python、NumPy、PyTorch 和 CUDA seed，并关闭 cuDNN benchmark、启用 deterministic 模式。但当前论文主结果每个市场只报告一个挑选后的 seed：

- Nasdaq-100：49
- CSI-300：90

这不能满足 AAAI 清单中以下三项：

- 每个结果用了多少次独立运行；
- 是否报告方差、置信区间或其他分布信息；
- 改进是否通过适当统计检验判断。

建议补充：

- 至少 5 个预先固定的 seeds；
- 每个模型/市场报告 mean ± std 或 bootstrap CI；
- paired test，例如基于同一测试日收益或同一 seed 配对的 Wilcoxon signed-rank；
- `scripts/run_multiseed.sh`
- `scripts/summarize_multiseed.py`
- `scripts/statistical_tests.py`
- `results/multiseed_manifest.csv`

不能只把 seed 49/90 的选择过程写成“最终 seed”；需要公开 seed 搜索范围、候选数量和选择标准，否则也无法完整回答 AAAI 的超参数/模型选择问题。

## 10. README、许可证、匿名化和测试

提交包必须新增：

- 根目录 `README.md`：安装、数据准备、训练、评估、作图、预期输出。
- `LICENSE`：主代码许可证；第三方 baseline 保留各自许可证。
- `THIRD_PARTY_LICENSES.md`
- `MODEL_CARD.md`
- `DATA_CARD.md`
- `MANIFEST.json`
- `EXPECTED_RESULTS.md`
- `tests/test_smoke_reproduction.py`

双盲检查：

- 删除作者姓名、邮箱、实验室、机构和 Git remote。
- 删除所有 `/home/tongwenxuan/...` 绝对路径。
- 用真实相对文件替代 `results/end` 和 `reproduced_inputs` 中的绝对软链接。
- 检查 checkpoint、日志、Office/PDF 元数据。

## 11. 不要放进 AAAI Code and Data ZIP

- `.git/`
- `.pytest_cache/`
- `__pycache__/`、`*.pyc`
- `.vscode/`
- `.superpowers/`
- `paper_backup_*`
- `paper/backups/`
- PPT、演讲稿和无关 PDF
- 历史失败实验和整个 `results/`
- `archive/` 中未被最终流水线调用的旧实现
- DeepTrader 的全部历史 output/checkpoint 目录
- 已生成的重复 PNG/PDF 缓存
- 指向本机绝对路径的软链接

只保留最终论文涉及的源码、配置、数据、checkpoint、trace、指标 CSV、必要图和日志。

## 12. 建议的最终 ZIP 结构

```text
CMTFlow_AAAI27/
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── environment.yml
├── SYSTEM.md
├── MANIFEST.json
├── EXPECTED_RESULTS.md
├── configs/
│   ├── nas_seed49.yaml
│   └── sh_seed90.yaml
├── cmtflow/
│   ├── models/
│   ├── agents/
│   ├── envs/
│   └── data/
├── scripts/
│   ├── prepare_data.py
│   ├── train_nas_seed49.sh
│   ├── train_sh_seed90.sh
│   ├── evaluate.sh
│   ├── reproduce_tables.sh
│   ├── reproduce_figures.sh
│   ├── run_multiseed.sh
│   └── statistical_tests.py
├── data/
│   ├── README.md
│   ├── data_manifest.csv
│   ├── nas/
│   ├── sh/
│   └── representative_smoke_subset/
├── checkpoints/
│   ├── nas_seed49/
│   └── sh_seed90/
├── baselines/
├── outputs/
│   ├── traces/
│   ├── metrics/
│   ├── tables/
│   └── figures/
└── tests/
    └── test_smoke_reproduction.py
```

## 13. 按优先级列出的待补材料

### P0：投稿前必须解决

- [x] 已确定采用 joint-finetune 版本；CSI-300 为 checkpoint
  `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94`
  的“240 版本”。
- [ ] 让论文数值、checkpoint、command JSON、trace、表格和图全部指向同一版本。
- [ ] 冻结手续费报告口径：主表保留 0.005% 的 240.13%，或使用完整
  0.01% 回测的新结果；不得把 fixed-path 237.01% 写成完整回测。
- [ ] 给评估入口增加 eval-only `--transaction_cost_pct 0.0001`，并补齐
  fixed-path 敏感性生成脚本。
- [ ] 冻结训练代码 commit，并生成文件级 SHA256 manifest。
- [ ] 提供根级环境锁定文件。
- [ ] 提供主代码 LICENSE 和第三方许可证。
- [ ] 去除所有绝对软链接和个人路径。
- [ ] 冻结完整数据预处理流水线并确认数据再分发许可。
- [ ] 修复 DeepTrader、AlphaStock baseline 的结果/trajectory 对齐问题。

### P1：AAAI 可复现性清单的主要缺口

- [ ] 报告超参数搜索范围、候选数量和最终选择规则。
- [ ] 提供多 seed 运行、方差/置信区间和统计检验。
- [ ] 给新方法源码增加与论文公式/算法步骤对应的注释。
- [ ] 提供一键评估、表格和作图命令。
- [ ] 提供 smoke test 和预期输出。

### P2：提高审稿人使用体验

- [ ] 提供 5–10 分钟 CPU smoke subset。
- [ ] 提供预训练模型的 eval-only 命令。
- [ ] 提供完整训练预计时间、显存和磁盘占用。
- [ ] 将训练、评估、统计和作图输出统一写入带 manifest 的目录。

## 14. 如果以当前状态直接填写 AAAI 清单

| AAAI 计算实验条目 | 当前判断 |
|---|---|
| 数据预处理代码已包含 | `partial/no` |
| 训练和分析全部源码已包含 | `partial` |
| 源码将以允许研究使用的许可证公开 | `no`，当前无根 LICENSE |
| 新方法源码有论文步骤注释 | `partial` |
| 随机 seed 足以复现 | `partial`，有 seed 设置但有模型挑选与 rollout 随机性问题 |
| 计算基础设施完整说明 | `partial/no` |
| 指标定义和选择理由 | `yes/partial` |
| 每项结果的独立运行次数 | `no` |
| 报告方差、置信区间或分布信息 | `no` |
| 使用适当统计检验 | `no/partial` |
| 列出所有最终超参数 | `partial` |
| 给出超参数搜索范围和选择标准 | `partial/no` |

当前最大风险不是“代码文件数量不够”，而是**论文、checkpoint 和分析输出没有冻结在同一版本线上**。应先解决版本一致性，再做 ZIP 精简和匿名化。
