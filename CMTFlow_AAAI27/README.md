# CMTFlow AAAI-27 Reproduction Package

This directory is organized around the anonymous paper
*Controller-Manager-Trader: Role-Decoupled Hierarchical Reinforcement Learning
for Portfolio Management* (paper SHA256 `a8e3d7817d34ca5a41c6ac5b45c5352d514182c0ecf35d275ed0a36549aea289`).

## Reproduction target

- Nasdaq-100 seed 49 checkpoint: paper backtest TR 262.49% at 0.01% cost.
- CSI-300 seed 90 checkpoint: paper backtest TR 237.01% at 0.01% cost
  (240.13% at the 0.005% reference-cost replay).
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
python src/paper_experiments/analyze_transaction_cost_sensitivity.py \
  --full_actions_root traces/figure4/_cache/inner_base_adjustment \
  --prices_root data/full \
  --results_root checkpoints \
  --output_dir outputs/transaction_cost \
  --markets nas sh \
  --seeds nas:49 sh:90 \
  --cost_rates 0.00005 0.00010 0.00015 0.00020 0.00050 \
  --reference_rate 0.00005
```

## Reproduce Figure 3

```bash
python src/paper_experiments/render_aaai27_figure3.py \
  --trace_root traces/figure3 \
  --case_manifest configs/aaai27_figure3_cases.json \
  --output_dir outputs/figure3
```

## Reproduce Figure 4

```bash
python src/paper_experiments/plot_inner_actor_base_adjustment.py \
  --results_root checkpoints \
  --prices_root data/full \
  --actions_root traces/figure4/_cache/inner_base_adjustment \
  --output_dir outputs/figure4 \
  --case_manifest configs/aaai27_figure4_cases.json \
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

The lower training fee is intentional for both markets: it prevents transaction
costs from suppressing the Trader's weak daily incremental reward signal. Paper
evaluation is reported at 0.01%, and the appendix evaluates a wider fee sweep.

## Appendix

Additional architecture/training documentation, transaction-cost sensitivity,
dense 1–60 day fixed holding-window sensitivity, Controller case analysis, and
Controller/Trader statistical validation are organized under `appendix/`. Run
`python appendix/code/run_appendix.py` after installing the requirements to
reproduce its directly packaged tables and figures.

## Known boundaries

- AlphaStock CSI-300 has a logged metric but no matched daily action trajectory.
- Some historical baseline replay files differ from the values frozen in the
  submitted PDF; the expected CSVs are the paper authority.
- License selection and full market-data redistribution require author approval;
  see `third_party/licenses/LICENSE_STATUS.md`.
