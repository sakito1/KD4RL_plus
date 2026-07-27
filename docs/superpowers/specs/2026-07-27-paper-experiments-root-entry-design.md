# Paper Experiments Root Entry Design

## Goal

Make `/home/tongwenxuan/KD_abk/KD4RL_plus/run_paper_experiments_final.py`
a reliable command-line entry point without maintaining a second copy of the
paper experiment implementation.

## Design

- Replace the root-level duplicated implementation with a small wrapper.
- Import `main` from
  `paper_experiments.run_paper_experiments_final`.
- Execute that `main` only under `if __name__ == "__main__"`.
- Keep all argument parsing, model loading, experiment logic, and figure
  generation in the package module.

This ensures the package module computes its `ROOT` from its actual location,
so default paths resolve under `/home/tongwenxuan/KD_abk/KD4RL_plus`.

## Error Handling

The wrapper does not catch exceptions. Errors from the canonical implementation
retain their original traceback and exit status.

## Verification

Run the root entry with the already validated SH90/NAS49 inputs and
`--skip_fixed_eval`. Verify exit code 0 and the expected summary:

- 20 main metric rows
- 8 ablation metric rows
- 4 controller cases
- 2 inner summary rows
