# End-to-End HRL/Controller Eval-Only Explanation Experiments

This package generates replay-only paper explanation experiments for the current
end-to-end HRL/controller joint training path:

`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`

It evaluates existing checkpoints under `results/end` and does not retrain,
update optimizers, or write new checkpoint parameters.

## Experiments

1. **Stage-wise Checkpoint Progression**
   Evaluates `hrl_fixed_best.pth`, `controller_best.pth`, and `best_model.pth`
   to show the contribution of fixed HRL, controller PG, and final controller-active
   end-to-end joint finetuning.

2. **Fixed HRL / No Inner / Full**
   Runs inference ablations from `best_model.pth`: fixed HRL without inner,
   fixed HRL with inner, and the full controller model. This supports claims
   about both inner actor and controller effectiveness.

3. **Cumulative Inner Alpha**
   Computes `exec_log_return - base_log_return` and its cumulative sum to show
   whether the inner actor provides persistent local alpha.

4. **Switch Advantage Alignment**
   Compares controller exit probabilities with future switch advantage on free
   decision days. This tests whether switch probability is economically meaningful.

5. **Switch Event Study**
   Summarizes learned free switch events and their continue-hold vs switch
   counterfactuals, supporting early-exit timing interpretation.

6. **Random Switch Matched Count**
   Replays random switch schedules with the same number of free switch decisions
   as the learned controller to test whether performance comes from timing rather
   than merely switching more often.

## Not Included

These are intentionally not included as core experiments because they are not
part of the final end-to-end joint training mechanism:

- threshold sensitivity
- stride curriculum
- local advantage ablation
- expected switch penalty ablation
- controller-only frozen-HRL main result

## Commands

Dry run:

```bash
python -m paper_experiments.eval_end_to_end_explain \
  --results_root /home/tongwenxuan/KD4RL_plus/results/end \
  --output_dir paper_experiments_outputs/end_to_end_explain \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --dry_run
```

Full eval-only run:

```bash
python -m paper_experiments.eval_end_to_end_explain \
  --results_root /home/tongwenxuan/KD4RL_plus/results/end \
  --output_dir paper_experiments_outputs/end_to_end_explain \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --random_runs 50
```

Smoke run:

```bash
python -m paper_experiments.eval_end_to_end_explain \
  --results_root /home/tongwenxuan/KD4RL_plus/results/end \
  --output_dir paper_experiments_outputs/end_to_end_explain_smoke \
  --markets sh \
  --seeds sh:90 \
  --device cuda \
  --test_max_days 120 \
  --random_runs 5
```

Regenerate figures:

```bash
python -m paper_experiments.plot_end_to_end_explain \
  --input_dir paper_experiments_outputs/end_to_end_explain \
  --output_dir paper_experiments_outputs/end_to_end_explain/figures
```

Regenerate tables:

```bash
python -m paper_experiments.table_end_to_end_explain \
  --input_dir paper_experiments_outputs/end_to_end_explain \
  --output_dir paper_experiments_outputs/end_to_end_explain/tables
```

Missing checkpoints or insufficient samples are reported as warnings and marked
as missing/empty outputs rather than being replaced by fabricated values.

