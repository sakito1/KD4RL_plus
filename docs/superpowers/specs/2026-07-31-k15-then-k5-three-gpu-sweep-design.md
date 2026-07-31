# K=15 Then K=5 Three-GPU Sweep Design

## Goal

Run the abk repository's final end-to-end CMTFlow training for Nasdaq-100 and
CSI-300 with seeds 50 through 61. Complete all `K=15` runs before starting any
`K=5` run, while keeping three GPUs occupied in each phase.

## Entry Point

Every training invocation must call:

`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`

The launcher resolves this path relative to its own location, ensuring it uses
`/home/tongwenxuan/KD_abk/KD4RL_plus` when launched from this checkout.

## Work Allocation

The launcher uses GPU IDs `0 1 2` by default and assigns seeds by round-robin
position:

| GPU | Seed shard |
|---:|---|
| 0 | 50, 53, 56, 59 |
| 1 | 51, 54, 57, 60 |
| 2 | 52, 55, 58, 61 |

For each K phase, every GPU worker runs its seed shard on both markets. This is
eight runs per GPU per phase, 24 runs per K, and 48 runs overall.

## Phase Ordering

The launcher executes two strict phases:

1. Start three `K=15` GPU workers and wait for all of them.
2. Only if all `K=15` workers succeed, start three `K=5` GPU workers and wait
   for all of them.

If any worker fails, the launcher reports the failed GPU/phase and exits
nonzero. It does not start the next phase after a failure.

## Output Isolation

Each phase/GPU worker uses a distinct run name:

`<RUN_PREFIX>_k<K>_gpu<GPU>`

This prevents concurrent workers from overwriting per-run logs or seed
artifacts. Top-level launcher logs are written under
`<OUTPUT_ROOT>/launcher_logs/`.

## Operator Controls

The user-facing script is:

`scripts/run_k15_then_k5_3gpu_seed50_61.sh`

It supports environment overrides for `GPU_IDS`, `SEEDS`, `MARKETS`,
`PYTHON_BIN`, `OUTPUT_ROOT`, `RUN_PREFIX`, and `DRY_RUN`. Validation requires
exactly three unique non-negative GPU IDs and twelve unique non-negative seeds.

`DRY_RUN=1` prints both phases and all three seed shards without starting a
training process.

## Tests

Focused tests verify that:

- dry-run output shows K=15 before K=5;
- the three default seed shards match the table;
- both markets are passed to every worker;
- the resolved E2E entry belongs to the current abk checkout;
- invalid GPU or seed counts are rejected;
- shell syntax is valid.

