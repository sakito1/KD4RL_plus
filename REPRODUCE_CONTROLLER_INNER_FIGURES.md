# Controller 与 Inner-Actor 图复现目录记录

## 1. 复现范围

本文档记录 CSI-300 和 Nasdaq-100 两个市场解释性图的代码、模型、数据和输出位置：

1. CSI-300 Controller 在 `2021-07-07` 的 Hold/Switch 30 日冻结反事实图。
2. Nasdaq-100 Controller 在 `2021-04-19` 的 Hold/Switch 30 日冻结反事实图。
3. 两个市场的 Inner-Actor future 5-day relative return、inner tilt、executed weights 和 tilt-return alignment 图。

这里复现的是论文实验所用的计算方法和案例图。截图中的紧凑版排版曾在后续版本中调整；当前 master 快照保留了核心计算代码，但没有完整提交最终紧凑版绘图实现和当时的 CSV 缓存。

## 2. 固定版本

### 实验代码

```text
/home/tongwenxuan/KD_abk/KD4RL_plus
```

- 分支：`master`
- commit：`c9c779e39f8d3c28f784938841c453fe956adede`
- 用途：保存本次论文实验实际使用的代码。

复现时不要使用 `/home/tongwenxuan/KD4RL_plus` 当前 `v1` 分支中的训练代码，因为该目录已经用于后续开发。

### Python 环境

```text
/home/tongwenxuan/conda/envs/xuangu/bin/python
```

已核对的主要环境：

```text
torch       2.4.0+cu124
pandas      2.2.3
numpy       2.2.5
matplotlib  3.10.0
CUDA        available
```

## 3. 当前文件实际存放位置

### 3.1 代码

```text
/home/tongwenxuan/KD_abk/KD4RL_plus/
├── run_hrl_training.py
├── agent.py
├── env.py
├── Train/
├── utils/
└── paper_experiments/
    ├── eval_end_to_end_explain.py
    ├── run_paper_experiments_final.py
    ├── plot_inner_actor_base_adjustment.py
    └── trace_utils.py
```

关键代码：

- `paper_experiments/eval_end_to_end_explain.py`
  - 加载模型并执行 eval-only 回放。
  - 构造 Hold/Switch 冻结反事实曲线。
  - 记录 base weight、executed weight 和 inner tilt。
- `paper_experiments/run_paper_experiments_final.py`
  - 自动选择 Controller 代表性案例。
  - 计算30日收益、回撤和 MDD。
  - 绘制 Controller case。
- `paper_experiments/plot_inner_actor_base_adjustment.py`
  - 计算未来5日横截面相对收益。
  - 计算 `inner tilt = executed weight - base weight`。
  - 选择代表性窗口并绘制 Inner-Actor 图。

### 3.2 模型与训练记录

论文表格和截图最终选中的两份模型保存在两个不同的实验运行目录，不是当前 `results/end` 中的旧归档：

```text
CSI-300:
/home/tongwenxuan/KD4RL_plus/results/
  e2e_standard_joint_lowlr_20260622_01/
  lookback60_hold30_standard_joint_lowlr_nas49_sh90/
  sh/
  ├── seed_90_command.json
  ├── seed_90.log
  └── ppo/seed_90/
      ├── checkpoints/best_model.pth
      └── test_s3_AllModules.csv

Nasdaq-100:
/home/tongwenxuan/KD4RL_plus/results/
  controller_first_joint_lowlr_retry_20260622_02/
  lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/
  nas/
  ├── seed_49_command.json
  ├── seed_49.log
  └── ppo/seed_49/
      ├── checkpoints/best_model.pth
      └── test_s3_AllModules.csv
```

论文选择依据：

```text
CSI-300:   TR 240.13%, Sharpe 1.2453, MDD 22.70%
Nasdaq-100: TR 265.53%, Sharpe 1.1500, MDD 18.62%
```

文件校验值：

```text
SHA256(SH selected best_model.pth)
9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94

SHA256(SH selected seed_90_command.json)
08260d953d2a7f29c70e05c4d937887753d5633b84ad899214da1d7a827d7725

SHA256(NAS selected best_model.pth)
e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d

SHA256(NAS selected seed_49_command.json)
42eb55217877c4cdf9a2f351b7a583a44e67efb7b518e3042d3ce9a0e63eb2d3
```

当前 `results/end` 中的 checkpoint 是旧归档或同指标的另一随机运行：

```text
results/end/sh_seed90:  TR 204.99%，不是论文的 240.13% 模型
results/end/nas_seed49: TR 265.53%，但 checkpoint 哈希与论文最终选择不同
```

因此，复现论文截图和统计时必须使用上面的 selected 模型。

### 3.3 CSI-300 与 Nasdaq-100 数据

master 实验代码内的数据：

```text
/home/tongwenxuan/KD_abk/KD4RL_plus/DeepAries/data/sh/sh_data.csv
/home/tongwenxuan/KD_abk/KD4RL_plus/DeepAries/data/nas/nas_data.csv
```

v1 目录中的副本：

```text
/home/tongwenxuan/KD4RL_plus/DeepAries/data/sh/sh_data.csv
/home/tongwenxuan/KD4RL_plus/DeepAries/data/nas/nas_data.csv
```

两个市场在 master 与 v1 目录中的对应数据文件 SHA256 均相同：

```text
CSI-300:
6917725757c1087d6c6a902ed2370423f63060a20afeb1513195d9d81f06ad47

Nasdaq-100:
114239fe8ab11443416b59b3e896007e80433d6829fa8afa3de77665f605b3b4
```

因此复现时直接使用 `KD_abk` 内的数据即可。

## 4. 推荐的可复现目录结构

评估代码会根据 `results_root` 的真实路径决定导入哪一份训练代码。当前已经建立了一个不会修改源模型的只读映射：

```text
/home/tongwenxuan/KD_abk/KD4RL_plus/
  reproduced_inputs/paper_selected/results_root/
  ├── sh_seed90/
  │   ├── seed_90_command.json -> SH selected command
  │   └── checkpoints/
  │       ├── hrl_fixed_best.pth -> SH selected checkpoint
  │       ├── controller_best.pth -> SH selected checkpoint
  │       └── best_model.pth -> SH selected checkpoint
  └── nas_seed49/
      ├── seed_49_command.json -> NAS selected command
      └── checkpoints/
          ├── hrl_fixed_best.pth -> NAS selected checkpoint
          ├── controller_best.pth -> NAS selected checkpoint
          └── best_model.pth -> NAS selected checkpoint
```

后续命令统一将这个目录作为 `RESULTS_ROOT`。由于它不是指向 v1 项目根目录的整体软链接，解释脚本仍然导入 `KD_abk` 的 master 代码。

## 5. 输出目录

建议所有新结果写入独立目录，不覆盖旧结果：

```text
/home/tongwenxuan/KD_abk/KD4RL_plus/reproduced_outputs/
├── end_to_end_explain/
│   ├── traces/
│   ├── metrics/
│   ├── figures/
│   ├── tables/
│   └── logs/
└── paper_figures/
    ├── 03_controller_interpretability/
    ├── 04_inner_actor_interpretability/
    ├── tables/
    └── _cache/
```

预期的目标文件：

```text
reproduced_outputs/paper_figures/
├── 03_controller_interpretability/
│   ├── controller_case_sh_01.png
│   ├── controller_case_sh_01.pdf
│   ├── controller_case_nas_01.png
│   ├── controller_case_nas_01.pdf
│   ├── selected_controller_cases_sh.csv
│   └── selected_controller_cases_nas.csv
└── 04_inner_actor_interpretability/
    ├── inner_actor_base_adjustment_future_return_sh.png
    ├── inner_actor_base_adjustment_future_return_sh.pdf
    ├── inner_actor_base_adjustment_future_return_nas.png
    ├── inner_actor_base_adjustment_future_return_nas.pdf
    └── inner_actor_base_adjustment_future_return_summary.csv
```

## 6. 复现前检查

```bash
cd /home/tongwenxuan/KD_abk/KD4RL_plus

PY=/home/tongwenxuan/conda/envs/xuangu/bin/python
RESULTS_ROOT="$PWD/reproduced_inputs/paper_selected/results_root"

"$PY" -m paper_experiments.eval_end_to_end_explain \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$PWD/reproduced_outputs/dry_run" \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --dry_run
```

输出清单中以下三个 checkpoint 应显示为存在：

```text
hrl_fixed_best.pth
controller_best.pth
best_model.pth
```

也可以重新检查版本和文件哈希：

```bash
git rev-parse HEAD

sha256sum \
  "$RESULTS_ROOT/sh_seed90/checkpoints/best_model.pth" \
  "$RESULTS_ROOT/sh_seed90/seed_90_command.json" \
  "$RESULTS_ROOT/nas_seed49/checkpoints/best_model.pth" \
  "$RESULTS_ROOT/nas_seed49/seed_49_command.json" \
  "$PWD/DeepAries/data/sh/sh_data.csv" \
  "$PWD/DeepAries/data/nas/nas_data.csv"
```

## 7. Inner-Actor 图复现命令

该脚本会从 `best_model.pth` 重新生成 action trace，然后绘图：

```bash
cd /home/tongwenxuan/KD_abk/KD4RL_plus

PY=/home/tongwenxuan/conda/envs/xuangu/bin/python
RESULTS_ROOT="$PWD/reproduced_inputs/paper_selected/results_root"
OUTPUT_ROOT="$PWD/reproduced_outputs/paper_figures"

"$PY" -m paper_experiments.plot_inner_actor_base_adjustment \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$OUTPUT_ROOT" \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --future_horizon 5 \
  --force_eval
```

生成文件：

```text
reproduced_outputs/paper_figures/04_inner_actor_interpretability/
```

其 eval 缓存位于：

```text
reproduced_outputs/paper_figures/_cache/inner_base_adjustment/
```

保留这个缓存可以避免以后每次都重新运行完整测试集。

## 8. Controller 30日案例图复现

Controller 图需要重新生成 `counterfactual_horizon=30` 的 trace。以下命令只调用 Controller 案例所需的函数，不依赖主结果表和 baseline manifest：

```bash
cd /home/tongwenxuan/KD_abk/KD4RL_plus

PY=/home/tongwenxuan/conda/envs/xuangu/bin/python

"$PY" - <<'PY'
from argparse import Namespace
from pathlib import Path

from paper_experiments.run_paper_experiments_final import (
    controller_experiment,
    ensure_counterfactual_horizon_eval,
    ensure_dirs,
)

root = Path("/home/tongwenxuan/KD_abk/KD4RL_plus")
results_root = root / "reproduced_inputs" / "paper_selected" / "results_root"
end2end_dir = root / "reproduced_outputs" / "end_to_end_explain"
output_root = root / "reproduced_outputs" / "paper_figures"

args = Namespace(
    results_root=str(results_root),
    output_dir=str(output_root),
    device="cuda",
    test_max_days=None,
    counterfactual_horizon=30,
    force_counterfactual_eval=True,
)

dirs = ensure_dirs(output_root)
markets = ["sh", "nas"]
seeds = {"sh": 90, "nas": 49}

bundles = ensure_counterfactual_horizon_eval(
    args,
    markets,
    seeds,
    dirs,
)

# controller_experiment 还会读取标准 full-controller trace。
# 如果该 trace 尚未生成，请先执行第9节的基础 eval。
controller_experiment(
    end2end_dir,
    markets,
    seeds,
    dirs,
    case_count=2,
    counterfactual_bundles=bundles,
    counterfactual_horizon=30,
)
PY
```

生成文件：

```text
reproduced_outputs/paper_figures/03_controller_interpretability/
```

30日反事实缓存位于：

```text
reproduced_outputs/paper_figures/_cache/counterfactual_horizon30/
```

## 9. 首次复现所需的基础 eval

第8节的 `controller_experiment` 还需要标准 full-controller portfolio/action trace。首次复现时先运行：

```bash
cd /home/tongwenxuan/KD_abk/KD4RL_plus

PY=/home/tongwenxuan/conda/envs/xuangu/bin/python
RESULTS_ROOT="$PWD/reproduced_inputs/paper_selected/results_root"
END2END_OUT="$PWD/reproduced_outputs/end_to_end_explain"

"$PY" -m paper_experiments.eval_end_to_end_explain \
  --results_root "$RESULTS_ROOT" \
  --output_dir "$END2END_OUT" \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --random_runs 0 \
  --force
```

需要存在的基础文件：

```text
reproduced_outputs/end_to_end_explain/traces/
├── sh_seed90_full_controller_portfolio.csv
├── sh_seed90_full_controller_actions.csv
├── sh_seed90_full_controller_switch_events.csv
├── nas_seed49_full_controller_portfolio.csv
├── nas_seed49_full_controller_actions.csv
└── nas_seed49_full_controller_switch_events.csv
```

执行顺序：

```text
第4节：建立模型目录映射
  ↓
第6节：dry-run 检查
  ↓
第9节：生成标准 full-controller trace
  ↓
第8节：生成30日 Controller 案例图
  ↓
第7节：生成 Inner-Actor 图
```

## 10. 截图版本差异

当前 master 绘图代码会生成方法一致的原始三面板 Controller 图和较完整标题的 Inner-Actor 图。用户截图中的两面板紧凑布局记录在：

```text
/home/tongwenxuan/KD4RL_plus/docs/superpowers/
├── specs/2026-07-13-paper-figure-readability-design.md
├── plans/2026-07-13-paper-figure-readability-implementation.md
└── ../../tests/test_paper_figure_readability.py
```

当前已发现的数值差异：

```text
截图 Nasdaq Controller：
Hold -1.91%，Switch +0.52%，gap +2.43 pp，MDD reduction +3.31 pp

当前论文记录：
Hold -1.91%，Switch +0.52%，gap +2.43 pp，MDD reduction +3.31 pp

截图 Nasdaq Inner-Actor：
Mean r 0.46，Positive days 73%

selected NAS 模型重新生成：
Mean r 0.45697，Positive days 73.33%

截图 CSI-300 Controller：Hold -10.27%，Switch +10.29%，gap +20.56 pp
论文代码记录：Hold -10.29%，Switch +10.25%，gap +20.54 pp

截图 CSI-300 Inner-Actor：Mean r 0.44，Positive days 73%
selected SH 模型重新生成：Mean r 0.43985，Positive days 73.33%
```

因此：

- 使用 selected 模型、本文目录和命令，可以从保存的模型、数据和 master 代码重新生成与两张 Inner-Actor 截图一致的案例统计。
- 若要求 Controller 数值和所有图形元素逐像素一致，还需要恢复截图生成时使用的最终绘图布局。
- 新生成的 CSV、缓存、PNG 和 PDF 应与本文件一起保存在 `reproduced_outputs/`，不要只保留最终截图。
