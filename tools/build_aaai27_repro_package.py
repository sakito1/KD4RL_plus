#!/usr/bin/env python3
"""Build the paper-first AAAI-27 CMTFlow reproduction directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


PAPER_SHA256 = "a8e3d7817d34ca5a41c6ac5b45c5352d514182c0ecf35d275ed0a36549aea289"

SOURCE_FILES = (
    "run_hrl_training.py",
    "run_paper_experiments_final.py",
    "Components/__init__.py",
    "Components/PPO_model.py",
    "Train/PPO_train.py",
    "Train/controller_pg.py",
    "agent/__init__.py",
    "agent/PPO_agent.py",
    "env/__init__.py",
    "env/PPO_env.py",
    "utils/__init__.py",
    "utils/PriceMatrix.py",
    "utils/Log.py",
    "utils/config.py",
    "utils/config_Nas.py",
    "utils/config_SH.py",
    "paper_experiments/__init__.py",
    "paper_experiments/metrics.py",
    "paper_experiments/trace_utils.py",
    "paper_experiments/eval_end_to_end_explain.py",
    "paper_experiments/analyze_transaction_cost_sensitivity.py",
    "paper_experiments/generate_baseline_matched.py",
    "paper_experiments/plot_end_to_end_explain.py",
    "paper_experiments/plot_inner_actor_base_adjustment.py",
    "paper_experiments/render_aaai27_figure3.py",
    "paper_experiments/run_paper_experiments_final.py",
    "paper_experiments/table_end_to_end_explain.py",
    "train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh",
    "configs/aaai27_figure3_cases.json",
    "configs/aaai27_figure4_cases.json",
    "tools/build_aaai27_repro_package.py",
)

TRACE_FILE_MAP = {
    "paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_experiment_metrics.csv":
        "traces/table1/source_metrics.csv",
    "paper_experiments_outputs/paper_experiments_final/02_ablation/ablation_metrics.csv":
        "traces/table2/source_metrics.csv",
    "paper_experiments_outputs/baseline_matched/manifest/baseline_sources.csv":
        "traces/table1/baseline_sources.csv",
    "paper_experiments_outputs/baseline_matched/log_snippets/alphastock_nas_seed46.txt":
        "traces/table1/logs/alphastock_nas_seed46.txt",
    "paper_experiments_outputs/baseline_matched/log_snippets/alphastock_sh_seed72.txt":
        "traces/table1/logs/alphastock_sh_seed72.txt",
    "paper_experiments_outputs/baseline_matched/log_snippets/deeptrader_nas_epoch6.txt":
        "traces/table1/logs/deeptrader_nas_epoch6.txt",
    "paper_experiments_outputs/baseline_matched/log_snippets/deeptrader_sh_epoch13.txt":
        "traces/table1/logs/deeptrader_sh_epoch13.txt",
    "paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/selected_controller_cases_nas.csv":
        "traces/figure3/selected_controller_cases_nas.csv",
    "paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/selected_controller_cases_sh.csv":
        "traces/figure3/selected_controller_cases_sh.csv",
    "paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_summary.csv":
        "traces/figure3/controller_case_summary.csv",
    "paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_combined_sh01_nas01.png":
        "expected/figure3/controller_case_combined_sh01_nas01.png",
    "paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_combined_sh01_nas01.pdf":
        "expected/figure3/controller_case_combined_sh01_nas01.pdf",
    "reproduced_outputs/inner_daily_stats_paper_selected/_cache/inner_base_adjustment/nas_seed49_full_controller_inner_base_actions.csv":
        "traces/figure4/_cache/inner_base_adjustment/nas_seed49_full_controller_inner_base_actions.csv",
    "reproduced_outputs/inner_daily_stats_paper_selected/_cache/inner_base_adjustment/sh_seed90_full_controller_inner_base_actions.csv":
        "traces/figure4/_cache/inner_base_adjustment/sh_seed90_full_controller_inner_base_actions.csv",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_nas.pdf":
        "expected/figure4/inner_actor_base_adjustment_future_return_nas.pdf",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_nas.png":
        "expected/figure4/inner_actor_base_adjustment_future_return_nas.png",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_sh.pdf":
        "expected/figure4/inner_actor_base_adjustment_future_return_sh.pdf",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_sh.png":
        "expected/figure4/inner_actor_base_adjustment_future_return_sh.png",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/trader_refinement_two_markets.pdf":
        "expected/figure4/trader_refinement_two_markets.pdf",
    "reproduced_outputs/aaai27_figure4_fixed/04_inner_actor_interpretability/trader_refinement_two_markets.png":
        "expected/figure4/trader_refinement_two_markets.png",
    "reproduced_outputs/aaai27_figure4_fixed/tables/inner_actor_base_adjustment_future_return_summary.csv":
        "expected/figure4/inner_actor_base_adjustment_future_return_summary.csv",
    "reproduced_outputs/aaai27_figure4_fixed/tables/inner_actor_base_adjustment_future_return_summary_display.csv":
        "expected/figure4/inner_actor_base_adjustment_future_return_summary_display.csv",
    "reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/TRANSACTION_COST_SENSITIVITY.md":
        "traces/transaction_cost/TRANSACTION_COST_SENSITIVITY.md",
    "reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/metadata/run_manifest.json":
        "traces/transaction_cost/metadata/run_manifest.json",
    "reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/tables/transaction_cost_sensitivity.csv":
        "traces/transaction_cost/tables/transaction_cost_sensitivity.csv",
    "reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/tables/nas_daily_replay.csv":
        "traces/transaction_cost/tables/nas_daily_replay.csv",
    "reproduced_outputs/fixed_path_transaction_cost_sensitivity_aaai27/tables/sh_daily_replay.csv":
        "traces/transaction_cost/tables/sh_daily_replay.csv",
}
TRACE_FILES = tuple(TRACE_FILE_MAP)

BASELINE_METHODS = (
    "buy_hold",
    "markowitz",
    "olmar",
    "ucrp",
    "alphastock",
    "deeparies",
    "deeptrader",
)

DEFAULT_ARTIFACT_ROOTS = {
    "nas_run": Path(
        "/home/tongwenxuan/KD4RL_plus/results/"
        "controller_first_joint_lowlr_retry_20260622_02/"
        "lookback60_hold30_controller_first_joint_lowlr_nas49_retry2/"
        "nas/ppo/seed_49"
    ),
    "nas_frozen_run": Path(
        "/home/tongwenxuan/KD4RL_plus/results/"
        "hrl_lookback60_hold30_inner_noaux_retrain/"
        "lookback60_hold30_inner_noaux_retrain/nas/ppo/seed_49"
    ),
    "sh_run": Path(
        "/home/tongwenxuan/KD4RL_plus/results/"
        "e2e_standard_joint_lowlr_20260622_01/"
        "lookback60_hold30_standard_joint_lowlr_nas49_sh90/"
        "sh/ppo/seed_90"
    ),
}

TABLE1_ROWS = (
    ("Nasdaq-100", "Buy&Hold", 160.64, 0.95, 29.79, 0.66),
    ("Nasdaq-100", "Markowitz", 57.76, 0.57, 24.73, 0.40),
    ("Nasdaq-100", "OLMAR", 157.77, 0.63, 45.48, 0.56),
    ("Nasdaq-100", "UCRP", 170.39, 1.02, 26.63, 0.76),
    ("Nasdaq-100", "AlphaStock", 185.35, 1.02, 21.90, 0.98),
    ("Nasdaq-100", "DeepAries", 157.07, 1.10, 17.16, 1.13),
    ("Nasdaq-100", "DeepTrader", 196.27, 0.90, 24.54, 0.94),
    ("Nasdaq-100", "CMTFlow", 262.49, 1.14, 18.66, 1.41),
    ("CSI-300", "Buy&Hold", 54.37, 0.53, 29.86, 0.36),
    ("CSI-300", "Markowitz", 50.53, 0.45, 36.11, 0.32),
    ("CSI-300", "OLMAR", 180.46, 0.69, 45.34, 0.67),
    ("CSI-300", "UCRP", 72.58, 0.66, 23.56, 0.55),
    ("CSI-300", "AlphaStock", 128.39, 0.87, 36.37, 0.53),
    ("CSI-300", "DeepAries", 148.76, 0.90, 15.23, 1.38),
    ("CSI-300", "DeepTrader", 212.81, 1.03, 31.86, 0.83),
    ("CSI-300", "CMTFlow", 237.01, 1.24, 22.91, 1.18),
)

TABLE2_ROWS = (
    ("Nasdaq-100", "w/o C+T", 220.42, 1.09, 32.09, 0.74),
    ("Nasdaq-100", "w/o C", 227.43, 1.11, 31.73, 0.76),
    ("Nasdaq-100", "w/o T", 237.50, 1.09, 21.24, 1.18),
    ("Nasdaq-100", "Fix-5d", 219.84, 1.03, 25.56, 0.95),
    ("Nasdaq-100", "Fix-10d", 199.65, 0.98, 28.24, 0.81),
    ("Nasdaq-100", "Fix-20d", 196.03, 0.93, 25.72, 0.83),
    ("Nasdaq-100", "Fix-60d", 170.88, 0.95, 29.96, 0.69),
    ("Nasdaq-100", "CMTFlow", 262.49, 1.14, 18.66, 1.41),
    ("CSI-300", "w/o C+T", 147.05, 0.94, 20.99, 0.99),
    ("CSI-300", "w/o C", 158.97, 0.99, 20.85, 1.04),
    ("CSI-300", "w/o T", 226.16, 1.19, 22.94, 1.15),
    ("CSI-300", "Fix-5d", 102.16, 0.76, 20.47, 0.81),
    ("CSI-300", "Fix-10d", 105.81, 0.79, 21.69, 0.78),
    ("CSI-300", "Fix-20d", 99.16, 0.76, 19.94, 0.81),
    ("CSI-300", "Fix-60d", 129.95, 0.89, 21.22, 0.90),
    ("CSI-300", "CMTFlow", 237.01, 1.24, 22.91, 1.18),
)

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".txt", ".yml", ".yaml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_regular_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)


def _command_json_for(run_root: Path, seed: int) -> Path:
    filename = f"seed_{int(seed)}_command.json"
    candidates = (
        run_root / filename,
        run_root.parents[1] / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing {filename}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _sanitize_packaged_text(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    replacements = (
        ("/home/tongwenxuan/conda/envs/xuangu/bin/python", "python"),
        ("/home/tongwenxuan/KD_abk/KD4RL_plus/", ""),
        ("/home/tongwenxuan/KD4RL_plus/", ""),
        ("/home/tongwenxuan/KD_abk/KD4RL_plus", "."),
        ("/home/tongwenxuan/KD4RL_plus", "."),
        ("/home/tongwenxuan/KD4RL", "."),
        ("/home/tongwenxuan/", "./"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _source_destination(relative: str) -> Path:
    path = Path(relative)
    if path.parts[0] == "train_sh":
        return Path("scripts/train") / path.name
    if path.parts[0] == "configs":
        return Path("configs") / path.name
    if path.parts[0] == "tools":
        return Path("scripts") / path.name
    return Path("src") / path


def _write_expected_table(path: Path, rows: Sequence[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "market",
                "method",
                "total_return_pct",
                "sharpe",
                "max_drawdown_pct",
                "calmar",
            ]
        )
        writer.writerows(rows)


def _requirements_text() -> str:
    return """numpy==2.2.5
pandas==2.2.3
matplotlib==3.10.0
scipy==1.15.3
scikit-learn==1.6.1
gym==0.26.2
torch==2.4.0
pytest==9.1.1
"""


def _readme_text() -> str:
    return f"""# CMTFlow AAAI-27 Reproduction Package

This directory is organized around the anonymous paper
*Controller-Manager-Trader: Role-Decoupled Hierarchical Reinforcement Learning
for Portfolio Management* (paper SHA256 `{PAPER_SHA256}`).

## Reproduction target

- Nasdaq-100 seed 49 checkpoint: paper backtest TR 262.49% at 0.01% cost.
- CSI-300 seed 90 "240 model": paper backtest TR 237.01% at 0.01% cost.
- Original training is preserved at its recorded 0.005% environment cost.
- The 0.01% paper values are a fixed-path mechanical cost replay; policy actions
  are not recomputed at the alternative fee.

Read `PACKAGE_STATUS.md` first. It separates material already included from
author-supplied data, licensing, and provenance items still required for a
public AAAI supplement.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 2.4.0 was originally used with CUDA 12.4. Install the matching PyTorch
wheel for your platform when GPU training is required.

## Data

Read `data/README.md`. Full price and SSM-state data are not redistributed until
their license is confirmed. Expected local layout:

```text
data/full/nas/nas_data.csv
data/full/sh/sh_data.csv
```

## Verify package integrity

```bash
python scripts/verify_package.py
```

## Reproduce the paper cost table

```bash
python src/paper_experiments/analyze_transaction_cost_sensitivity.py \\
  --full_actions_root traces/figure4/_cache/inner_base_adjustment \\
  --prices_root data/full \\
  --results_root checkpoints \\
  --output_dir outputs/transaction_cost \\
  --markets nas sh \\
  --seeds nas:49 sh:90 \\
  --cost_rates 0.00005 0.00010 0.00015 0.00020 0.00050 \\
  --reference_rate 0.00005
```

## Reproduce Figure 3

```bash
python src/paper_experiments/render_aaai27_figure3.py \\
  --trace_root traces/figure3 \\
  --case_manifest configs/aaai27_figure3_cases.json \\
  --output_dir outputs/figure3
```

## Reproduce Figure 4

```bash
python src/paper_experiments/plot_inner_actor_base_adjustment.py \\
  --results_root checkpoints \\
  --prices_root data/full \\
  --actions_root traces/figure4/_cache/inner_base_adjustment \\
  --output_dir outputs/figure4 \\
  --case_manifest configs/aaai27_figure4_cases.json \\
  --markets sh nas --seeds sh:90 nas:49
```

The cache CSVs under `traces/figure4/` are the exact model traces, and the
paper-fixed rendered outputs are under `expected/figure4/`. Figure 3 inputs and
the paper-rendered output are under `traces/figure3/` and `expected/figure3/`.

## Tables

`expected/table1.csv` and `expected/table2.csv` are transcribed from the locked
PDF. The source metrics and baseline provenance available in the working
repository are retained under `traces/table1/` and `traces/table2/`. Run:

```bash
python scripts/render_expected_tables.py
```

to create Markdown copies under `outputs/`.

## Training provenance

The paper checkpoints are already supplied; reproducing the displayed tables
does not require retraining. Original training remains at 0.005% cost. Exact
argument arrays are stored beside each checkpoint, and the five-stage mapping
is documented in `MODEL_PROVENANCE.md`. The preserved shell driver is under
`scripts/train/`.

## Known boundaries

- AlphaStock CSI-300 has a logged metric but no matched daily action trajectory.
- Some historical baseline replay files differ from the values frozen in the
  submitted PDF; the expected CSVs are the paper authority.
- License selection and full market-data redistribution require author approval;
  see `third_party/licenses/LICENSE_STATUS.md`.
"""


def _expected_results_text() -> str:
    return """# Expected Results

The authoritative expected values are stored in `expected/table1.csv` and
`expected/table2.csv`.

For the fixed-path cost replay, rounding to two decimals must give:

| Market | Cost | TR | SR | MDD | CR |
|---|---:|---:|---:|---:|---:|
| Nasdaq-100 | 0.01% | 262.49% | 1.14 | 18.66% | 1.41 |
| CSI-300 | 0.01% | 237.01% | 1.24 | 22.91% | 1.18 |

CSV-level floating-point differences below `1e-8` are acceptable. Paper table
comparisons use the displayed two-decimal values.
"""


def _package_status_text() -> str:
    return """# Package Status and Author-Supplied Items

## Included and verified

- Core Controller/Manager/Trader training and evaluation source.
- Nasdaq-100 seed 49 and CSI-300 seed 90 final checkpoints as real files.
- Checkpoint command JSONs and SHA-256 model manifest.
- Paper-authoritative Table 1 and Table 2 CSVs.
- Trace-calibrated 0.01% fixed-path replay code and recorded action traces.
- Figure 3 and Figure 4 fixed case manifests, plot code, inputs, and outputs.
- Relative-path README commands, pinned Python requirements, and file manifest.

## Still required from the authors before public release

1. Choose the original-code license and confirm third-party notices.
2. Confirm redistribution rights for the full Nasdaq-100/CSI-300 price data and
   derived SSM states; otherwise provide download/preparation instructions and
   hashes without redistributing restricted files.
3. Supply a full data manifest (file hashes, date ranges, field definitions,
   adjustment method, and missing-value policy).
4. Supply matched daily traces or runnable upstream artifacts for every Table 1
   baseline. The current evidence does not include a matched AlphaStock
   CSI-300 daily trajectory.
5. Supply daily traces for every Table 2 ablation/fixed-window row if reviewers
   must recompute them from paths rather than compare frozen metrics.
6. Record hardware, approximate runtime, number of independent runs, candidate
   seeds, and the final seed-selection rule.

These are explicit release boundaries. Missing licensed data and third-party
traces are not replaced with synthetic or guessed material.
"""


def _model_provenance_text() -> str:
    return """# Model and Training Provenance

## Paper-selected models

| Market | Seed | Package checkpoint | SHA-256 | Identity |
|---|---:|---|---|---|
| Nasdaq-100 | 49 | `checkpoints/nas_seed49/checkpoints/best_model.pth` | `e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d` | joint-finetune paper model |
| CSI-300 | 90 | `checkpoints/sh_seed90/checkpoints/best_model.pth` | `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94` | author-confirmed “CSI 240” model |

“CSI 240” identifies the checkpoint via its original 0.005% result (240.13%).
The paper uses a 0.01% fixed-path fee replay and reports 237.01%.

## Five-stage mapping

1. Manager warm-up: `--warmup_outer_epochs`.
2. Trader warm-up: `--warmup_inner_epochs`.
3. Manager–Trader stabilization: fixed-HRL/joint phase and
   `hrl_fixed_best.pth`.
4. Controller training: monitor/controller epochs and controller checkpoint.
5. End-to-end alignment: `--joint_epochs`, controller-active joint options, and
   final `best_model.pth`.

The exact argument arrays are retained as:

- `checkpoints/nas_seed49/frozen_hrl_seed_49_command.json`
- `checkpoints/nas_seed49/seed_49_command.json`
- `checkpoints/sh_seed90/seed_90_command.json`

Nasdaq uses the supplied `original_hrl_fixed_best.pth` as the frozen
Manager–Trader upstream checkpoint before Controller/joint fine-tuning. CSI
uses the supplied `hrl_fixed_best.pth`, `controller_best.pth`, and final model
from its seed-90 run.

Original training configuration files retain `TRANSACTION_COST_RATE=5e-5`.
Only the separate paper replay script applies `0.0001`; no training config is
globally changed.
"""


def _data_readme_text() -> str:
    return """# Data Placement and Licensing

The paper uses 39 Nasdaq-100 stocks and 53 CSI-300 stocks. Exact ordered pool
lists are provided in `pool_lists/`.

Full price CSVs and precomputed SSM states are intentionally not copied into
this directory because the repository does not establish redistribution rights.
After confirming the data license, place the files under `data/full/` using the
layout documented in the package README.

Code split:

- Nasdaq: train through 2017-12-29, validation 2018-01-02 through 2020-04-22,
  test 2020-04-23 through 2025-10-03.
- CSI-300: train through 2017-12-28, validation 2018-01-02 through 2019-12-31,
  test 2020-01-02 through 2025-02-28.
"""


def _license_status_text() -> str:
    return """# License Status — Author Action Required

No license has been selected automatically.

Before external submission or public release, the authors must:

1. choose a license for the original CMTFlow source;
2. retain and document licenses/notices for AlphaStock, DeepTrader, DeepAries,
   and any other redistributed third-party material;
3. confirm whether the Nasdaq-100 and CSI-300 market data and derived SSM states
   may be redistributed.

This file is a legal-status reminder, not a software license.
"""


VERIFY_SCRIPT = '''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
failures = []
for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file() or path.is_symlink():
        failures.append(f"missing-or-symlink: {entry['path']}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        failures.append(f"hash-mismatch: {entry['path']}")
if failures:
    raise SystemExit("\\n".join(failures))
print(f"verified {len(manifest['files'])} packaged files")
'''

RENDER_SCRIPT = '''#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "outputs"
output.mkdir(exist_ok=True)
for table_name in ("table1", "table2"):
    source = root / "expected" / f"{table_name}.csv"
    rows = list(csv.reader(source.open(encoding="utf-8")))
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    (output / f"{table_name}.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(output / f"{table_name}.md")
'''


def _write_model_manifest(destination: Path) -> None:
    entries = []
    for path in sorted((destination / "checkpoints").rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination).as_posix()
            if "nas_seed49" in path.parts:
                market, seed = "Nasdaq-100", 49
            elif "sh_seed90" in path.parts:
                market, seed = "CSI-300", 90
            else:
                market, seed = "unknown", None
            if path.name == "best_model.pth":
                role = "paper_final_checkpoint"
            elif path.name == "controller_best.pth":
                role = "controller_checkpoint"
            elif path.name.endswith("hrl_fixed_best.pth"):
                role = "manager_trader_stabilization_checkpoint"
            elif path.name.endswith("_command.json"):
                role = "training_command_record"
            else:
                role = "supporting_artifact"
            entry = {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "market": market,
                "seed": seed,
                "role": role,
            }
            if role == "paper_final_checkpoint":
                upstream_name = (
                    "original_hrl_fixed_best.pth"
                    if market == "Nasdaq-100"
                    else "hrl_fixed_best.pth"
                )
                entry["upstream_checkpoint"] = (
                    f"checkpoints/{'nas_seed49' if seed == 49 else 'sh_seed90'}"
                    f"/checkpoints/{upstream_name}"
                )
            entries.append(entry)
    (destination / "checkpoints/MODEL_MANIFEST.json").write_text(
        json.dumps({"files": entries}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_file_manifest(destination: Path) -> None:
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"packaged symlink is not allowed: {path}")
        if path.is_file() and path.name != "MANIFEST.json":
            entries.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_sha256": PAPER_SHA256,
        "files": entries,
    }
    (destination / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def build_package(
    source_root: Path,
    destination: Path,
    *,
    artifact_roots: Mapping[str, Path] | None = None,
) -> Path:
    source_root = Path(source_root).resolve()
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    roots = {
        key: Path(value).resolve()
        for key, value in (artifact_roots or DEFAULT_ARTIFACT_ROOTS).items()
    }
    required_root_keys = {"nas_run", "nas_frozen_run", "sh_run"}
    if set(roots) != required_root_keys:
        raise ValueError(f"artifact_roots must contain {sorted(required_root_keys)}")

    for relative in SOURCE_FILES:
        target = destination / _source_destination(relative)
        _copy_regular_file(source_root / relative, target)

    _copy_regular_file(
        source_root / "utils/NAS100_pool.txt",
        destination / "data/pool_lists/NAS100_pool.txt",
    )
    _copy_regular_file(
        source_root / "utils/SH_pool.txt",
        destination / "data/pool_lists/SH_pool.txt",
    )
    for source_relative, target_relative in TRACE_FILE_MAP.items():
        _copy_regular_file(
            source_root / source_relative,
            destination / target_relative,
        )
    for market in ("nas", "sh"):
        for method in BASELINE_METHODS:
            source = (
                source_root
                / "paper_experiments_outputs/baseline_matched"
                / market
                / "curves"
                / f"{method}.csv"
            )
            if source.exists():
                _copy_regular_file(
                    source,
                    destination
                    / "traces/table1/curves"
                    / market
                    / f"{method}.csv",
                )

    checkpoint_map = {
        roots["nas_run"] / "checkpoints/best_model.pth":
            "checkpoints/nas_seed49/checkpoints/best_model.pth",
        roots["nas_run"] / "checkpoints/controller_best.pth":
            "checkpoints/nas_seed49/checkpoints/controller_best.pth",
        roots["nas_frozen_run"] / "checkpoints/hrl_fixed_best.pth":
            "checkpoints/nas_seed49/checkpoints/original_hrl_fixed_best.pth",
        _command_json_for(roots["nas_run"], 49):
            "checkpoints/nas_seed49/seed_49_command.json",
        _command_json_for(roots["nas_frozen_run"], 49):
            "checkpoints/nas_seed49/frozen_hrl_seed_49_command.json",
        roots["sh_run"] / "checkpoints/best_model.pth":
            "checkpoints/sh_seed90/checkpoints/best_model.pth",
        roots["sh_run"] / "checkpoints/controller_best.pth":
            "checkpoints/sh_seed90/checkpoints/controller_best.pth",
        roots["sh_run"] / "checkpoints/hrl_fixed_best.pth":
            "checkpoints/sh_seed90/checkpoints/hrl_fixed_best.pth",
        _command_json_for(roots["sh_run"], 90):
            "checkpoints/sh_seed90/seed_90_command.json",
    }
    for source, target in checkpoint_map.items():
        _copy_regular_file(source, destination / target)

    _write_expected_table(destination / "expected/table1.csv", TABLE1_ROWS)
    _write_expected_table(destination / "expected/table2.csv", TABLE2_ROWS)
    (destination / "README.md").write_text(_readme_text(), encoding="utf-8")
    (destination / "EXPECTED_RESULTS.md").write_text(
        _expected_results_text(),
        encoding="utf-8",
    )
    (destination / "PACKAGE_STATUS.md").write_text(
        _package_status_text(),
        encoding="utf-8",
    )
    (destination / "MODEL_PROVENANCE.md").write_text(
        _model_provenance_text(),
        encoding="utf-8",
    )
    (destination / "requirements.txt").write_text(
        _requirements_text(),
        encoding="utf-8",
    )
    (destination / "data/README.md").write_text(
        _data_readme_text(),
        encoding="utf-8",
    )
    license_path = destination / "third_party/licenses/LICENSE_STATUS.md"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text(_license_status_text(), encoding="utf-8")
    (destination / "PAPER_REFERENCE.json").write_text(
        json.dumps(
            {
                "title": (
                    "Controller-Manager-Trader: Role-Decoupled Hierarchical "
                    "Reinforcement Learning for Portfolio Management"
                ),
                "sha256": PAPER_SHA256,
                "authority": "Controller-Manager-Trader_AAAI.pdf",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "scripts/verify_package.py").write_text(
        VERIFY_SCRIPT,
        encoding="utf-8",
    )
    (destination / "scripts/render_expected_tables.py").write_text(
        RENDER_SCRIPT,
        encoding="utf-8",
    )

    for path in destination.rglob("*"):
        if path.is_file():
            _sanitize_packaged_text(path)
    _write_model_manifest(destination)
    _write_file_manifest(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, default=Path("CMTFlow_AAAI27"))
    parser.add_argument("--nas-run", type=Path, default=DEFAULT_ARTIFACT_ROOTS["nas_run"])
    parser.add_argument(
        "--nas-frozen-run",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOTS["nas_frozen_run"],
    )
    parser.add_argument("--sh-run", type=Path, default=DEFAULT_ARTIFACT_ROOTS["sh_run"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = build_package(
        args.source_root,
        args.destination,
        artifact_roots={
            "nas_run": args.nas_run,
            "nas_frozen_run": args.nas_frozen_run,
            "sh_run": args.sh_run,
        },
    )
    print(package.resolve())


if __name__ == "__main__":
    main()
