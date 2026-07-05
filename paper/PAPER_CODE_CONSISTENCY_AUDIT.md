# Paper-Code Consistency Audit

This note records the consistency check between `paper/anonymous-submission-latex-2026.tex` and the actual training path
`train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`.

## Checked Runtime Path

- Final seeds: Nasdaq-100 `49`, CSI-300 `90`.
- Runtime sequence: fixed-segment HRL warmup, fixed-HRL joint training, counterfactual controller PG, controller-active joint finetuning.
- The current path is end-to-end. It does not load an archived HRL checkpoint as the final training logic and does not use external switch or regime-label controller supervision.
- Outer lookback is overridden to `60`; inner lookback remains `10`; controller window is `30`; Top-K base size is `10`; inner refinement strength is `0.6`; transaction cost is `5e-5`.

## Corrected Paper Details

- Outer reward:
  The previous paper text described a segment Sharpe improvement over an equal-weight portfolio. The code instead accumulates daily cost-aware log portfolio returns between switch points and credits that segment return to the outer decision. The Method and Appendix reward formulas were updated accordingly.

- Inner reward:
  The code uses the relative daily log return between the executed portfolio and the active base portfolio, with transaction cost already included in the executed return. The paper now describes this as
  `log(mu_t y_t^T w_t) - log(y_t^T b_t)`, rather than a separate handcrafted turnover penalty term.

- Inner action:
  The actual inner actor uses a two-layer LSTM plus temporal attention. Its sampled Gaussian signal is used directly as masked softmax logits on the active base support, then convexly mixed with the base portfolio. The appendix no longer says the inner signal is `tanh`-squashed.

- Controller state:
  The controller explicitly conditions on the current drifted executed holding and the candidate outer portfolio. The final path supplies `asset_state=obs["outer_state"]`, so `MonitorAC` encodes the recent market tensor with a one-layer asset-wise LSTM and two temporal-attention blocks. The legacy API arguments `z/h/p/q_bear/q_bull` are not the active controller feature description in this path. The paper now emphasizes candidate turnover, candidate concentration, elementwise-min support overlap, and hold-versus-candidate representation differences.

- Controller evaluation constraint:
  The script passes `--controller_no_hold_constraints`, but `controller_eval_max_hold=-1` falls back to the global `max_hold=30`. Therefore final evaluation has no minimum-hold lock, daily checks, threshold `0.5`, and a 30-day maximum-hold cap. The paper and tables were corrected from "no min/max hold constraint" to this exact protocol.

- Controller objective:
  The controller is trained from counterfactual return uplift over a fixed-segment reference trajectory with overflow regularization on the controlled segment/switch count. Return and risk heads are auxiliary; only the switch-advantage estimate modulates the switch logit.

- Figure assets:
  The active framework figure now uses `paper/figures/cmtflow_architecture_vector.pdf`, whose labels match the current implementation: recent market tensor, current holding, candidate base, state/action comparison features, outer LSTM-HA+CAAN, inner LSTM-attention, and controller switch-advantage modulation. Deprecated imagegen figures with older labels were moved to `paper/backups/deprecated_old_figures/`.

## Verified Build

- Recompiled `paper/anonymous-submission-latex-2026.pdf` successfully after the latest code-path audit.
- Output remains 15 pages.
- The latest LaTeX log has no unresolved references, rerun warnings, overfull boxes, or compile errors; remaining messages are underfull box warnings only.

## Backup

Before editing, the previous TeX/PDF were copied to:

- `paper/backups/anonymous-submission-latex-2026.before_code_consistency_audit.tex`
- `paper/backups/anonymous-submission-latex-2026.before_code_consistency_audit.pdf`
