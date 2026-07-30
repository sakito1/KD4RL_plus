# Multi-Market, Multi-K, Multi-Seed Sweep Design

## Goal

Run the final end-to-end CMTFlow training recipe over configurable markets,
asset-selection values, and seed lists. Use every GPU listed by the operator
without scheduling two concurrent configurations on the same GPU.

## Default Experiment Matrix

| Market | K (`trade_num`) | Seeds |
|---|---:|---|
| Nasdaq-100 (`nas`) | 5 | 42--56 |
| Nasdaq-100 (`nas`) | 15 | 42--56 |
| CSI-300 (`sh`) | 5 | 83--97 |
| CSI-300 (`sh`) | 15 | 83--97 |

The launcher forms the Cartesian product `MARKETS × K_VALUES`. Configurations
are assigned round-robin to `GPU_IDS`. Each GPU gets one worker and runs all
of its assigned configurations sequentially; workers on different GPUs run
concurrently. The same launcher therefore supports one or many GPUs.

## Implementation

Add one user-facing launcher:

`scripts/run_multi_market_multi_k_seed_sweep.sh`

The launcher starts one background worker per configured GPU. Each worker
invokes the existing final end-to-end script for its assigned configurations,
sets `CUDA_VISIBLE_DEVICES`, passes the market-specific seed list, and writes
each configuration to a distinct run directory.

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

- `DRY_RUN=1` prints all configurations and GPU assignments without training.
- A stable default output root makes relaunching predictable.
- Each worker has its own top-level log.
- The launcher waits for every GPU worker and returns nonzero if any fails.
- `PYTHON_BIN`, `OUTPUT_ROOT`, `GPU_IDS`, `MARKETS`, `K_VALUES`, and both
  market seed strings are overrideable through environment variables.

## Result Selection

The launcher produces all 60 trained/tested runs. Existing per-seed logs and
`test_s3_AllModules.csv` files contain the final all-module test results.
Seed selection is intentionally outside the launcher so running and selecting
results are not coupled and a failed summary cannot interrupt training.

## Validation

Tests run the scripts in dry-run mode and verify:

- the complete market-by-K matrix is emitted;
- configurations are round-robin assigned over the supplied GPU list;
- one-GPU and multi-GPU dry runs report the correct total seed-run count;
- the final training script forwards `--trade_num`;
- the market filter prevents the other market from being scheduled;
- no training process starts during dry-run.
