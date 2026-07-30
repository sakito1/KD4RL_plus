# K Asset-Selection 4-GPU Seed Sweep Design

## Goal

Run the final end-to-end CMTFlow training recipe for `K=5` and `K=15`
on Nasdaq-100 and CSI-300, using four GPUs concurrently and fifteen seed
candidates per GPU.

## Experiment Matrix

| GPU | Market | K (`trade_num`) | Seeds |
|---:|---|---:|---|
| 0 | Nasdaq-100 (`nas`) | 5 | 42--56 |
| 1 | Nasdaq-100 (`nas`) | 15 | 42--56 |
| 2 | CSI-300 (`sh`) | 5 | 83--97 |
| 3 | CSI-300 (`sh`) | 15 | 83--97 |

Each GPU runs one worker. A worker runs its fifteen seeds sequentially, while
the four workers run concurrently.

## Implementation

Add one user-facing launcher:

`scripts/run_k5_k15_asset_selection_4gpu.sh`

The launcher starts four background invocations of the existing final
end-to-end script. It assigns one physical GPU to each invocation with
`CUDA_VISIBLE_DEVICES`, passes the market-specific seed list, and writes each
configuration to a distinct run directory.

Make the minimum compatibility changes to
`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`:

- accept `TRADE_NUM`, defaulting to the existing value `10`;
- pass `--trade_num "$TRADE_NUM"` to `run_hrl_training.py`;
- accept `MARKETS`, defaulting to the existing order `sh nas`;
- run only the requested market while retaining empty-seed skipping;
- derive the repository root from the script location instead of a fixed
  machine path.

Existing invocations remain unchanged because all new controls have
backward-compatible defaults.

## Operational Behavior

- `DRY_RUN=1` prints the four worker configurations without starting training.
- A stable default output root makes relaunching predictable.
- Each worker has its own top-level log.
- The launcher waits for all workers and returns nonzero if any worker fails.
- `PYTHON_BIN`, `OUTPUT_ROOT`, GPU IDs, and seed strings remain overrideable
  through environment variables.

## Result Selection

The launcher produces all 60 trained/tested runs. Existing per-seed logs and
`test_s3_AllModules.csv` files contain the final all-module test results.
Seed selection is intentionally outside the launcher so running and selecting
results are not coupled and a failed summary cannot interrupt training.

## Validation

Tests run the scripts in dry-run mode and verify:

- exactly four workers are emitted;
- the GPU, market, K, and seed mappings match the table above;
- the final training script forwards `--trade_num`;
- the market filter prevents the other market from being scheduled;
- no training process starts during dry-run.

