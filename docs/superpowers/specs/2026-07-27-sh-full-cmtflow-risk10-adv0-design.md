# SH Full CMTFlow Risk-10/Advantage-0 Training Design

## Goal

Provide dedicated scripts that train the Outer actor, Inner actor, and Controller
from scratch for SH seeds 44, 46, 49, and 54 using this Controller guidance rule:

```text
(Risk >= 0.10 and Advantage > 0.00) or Advantage >= 0.10
```

## Approach

Add a dedicated single-seed training script and a dedicated four-GPU scheduler.
Do not modify or reuse the semantics of the existing full-CMTFlow scripts, and do
not alter the current uncommitted 2% Controller experiment.

The single-seed script will follow the existing full-CMTFlow command structure
and use this schedule:

1. Outer actor warmup: 4 epochs.
2. Inner actor warmup: 2 epochs.
3. Outer/Inner joint training: 1 epoch.
4. Controller supervised pretraining: 3 epochs.
5. Controller policy-gradient training: 5 epochs.
6. Final test evaluation.

The Controller guidance thresholds will be passed explicitly:

- `--controller_guidance_risk_threshold 0.10`
- `--controller_guidance_risk_min_advantage_threshold 0.00`
- `--controller_guidance_advantage_threshold 0.10`

All other training parameters will match `train_sh/train_full_cmtflow_seed.sh`.

## Scripts

The single-seed script will:

- accept `SEED`, `GPU_ID`, `PYTHON_BIN`, `OUTPUT_ROOT`, and `RUN_NAME`;
- accept `DRY_RUN=1` to print the exact command without training;
- reject unsupported markets by training SH only;
- reject an existing run directory unless `ALLOW_EXISTING_OUTPUT=1`;
- write each run and log under a new experiment-specific output root.

The scheduler will:

- default to SH seeds `44 46 49 54`;
- map one seed to each of four GPUs by default;
- support GPU ID overrides and `JOBS_PER_GPU`;
- call the dedicated single-seed script;
- propagate failures after waiting for all queues;
- collect final test metrics into a summary file.

## Safety and Compatibility

The new scripts use unique result and run names containing
`risk10_adv0_or_adv10`, so they cannot silently overwrite the existing 5% or 2%
experiments. Existing scripts and current uncommitted changes remain untouched.

## Verification

An automated dry-run test will verify:

- exactly four SH jobs are emitted for seeds 44, 46, 49, and 54;
- Outer, Inner, joint, Controller pretrain, and Controller PG epoch counts match
  the approved schedule;
- the guidance thresholds are exactly 0.10, 0.00, and 0.10;
- Controller training is enabled;
- no frozen checkpoint or Controller-only mode is used;
- each job has a distinct output run name and GPU assignment.

Shell syntax checks and the focused automated test must pass before handoff.
