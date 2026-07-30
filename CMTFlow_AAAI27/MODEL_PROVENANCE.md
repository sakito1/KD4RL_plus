# Model and Training Provenance

## Paper-selected models

| Market | Seed | Package checkpoint | SHA-256 | Identity |
|---|---:|---|---|---|
| Nasdaq-100 | 49 | `checkpoints/nasdaq100/checkpoints/best_model.pth` | `e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d` | joint-finetune paper model |
| CSI-300 | 90 | `checkpoints/csi300/checkpoints/best_model.pth` | `9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94` | joint-finetune paper model |

Both rows use the market datasets named in the paper: Nasdaq-100 and CSI-300.
The informal “240 model” label refers only to the CSI-300 seed-90 checkpoint's
240.13% return at the 0.005% reference cost; it does not mean CSI-240. The paper
uses a 0.01% fixed-path fee replay and reports 237.01%.

## Five-stage mapping

1. Manager warm-up: `--warmup_outer_epochs`.
2. Trader warm-up: `--warmup_inner_epochs`.
3. Manager–Trader stabilization: fixed-HRL/joint phase and
   `hrl_fixed_best.pth`.
4. Controller training: monitor/controller epochs and controller checkpoint.
5. End-to-end alignment: `--joint_epochs`, controller-active joint options, and
   final `best_model.pth`.

The exact argument arrays are retained as:

- `checkpoints/nasdaq100/manager_trader_training_command.json`
- `checkpoints/nasdaq100/five_stage_training_command.json`
- `checkpoints/csi300/five_stage_training_command.json`

Only the final paper-selected checkpoint is distributed for each market.
Intermediate Manager–Trader and Controller stage checkpoints are intentionally
omitted: they are not needed for final evaluation, Appendix rendering, or
seed-locked five-stage retraining. The command records preserve the selected
seed and stage configuration without exposing it in public filenames.

Both markets follow all five stages above. Original training configuration files
retain `TRANSACTION_COST_RATE=5e-5`. The lower training cost is deliberate: it
prevents proportional fees from overwhelming the Trader's weak daily
incremental-return signal. Only the separate paper replay script applies
`0.0001`; no training config is globally changed.
