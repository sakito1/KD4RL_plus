# Package Status and Author-Supplied Items

## Included and verified

- Core Controller/Manager/Trader training and evaluation source.
- Nasdaq-100 seed 49 and CSI-300 seed 90 final checkpoints as real files.
- Checkpoint command JSONs and SHA-256 model manifest.
- Paper-authoritative Table 1 and Table 2 CSVs.
- Trace-calibrated 0.01% fixed-path replay code and recorded action traces.
- Figure 3 and Figure 4 fixed case manifests, plot code, inputs, and outputs.
- Appendix B.1/B.2/C.1/C.2/C.3 analysis and rendering code, compact derived
  inputs, locked tables, dense 1–60 day fixed-window figures, and two-market
  Controller hold/switch figures.
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
5. Supply daily traces for the remaining Table 2 ablation rows if reviewers
   must recompute them from paths rather than compare frozen metrics. Appendix
   B.2 already includes compact growth/turnover replays for all 1–60 day
   fixed-window schedules.
6. Record hardware, approximate runtime, number of independent runs, candidate
   seeds, and the final seed-selection rule.

The appendix direct renderer does not use a checkpoint. The two final
`best_model.pth` files support main-paper evaluation. Four intermediate
Controller/Manager–Trader stage checkpoints were excluded because the public
release promises seed-locked five-stage retraining, not resumption from every
internal stage.

These are explicit release boundaries. Missing licensed data and third-party
traces are not replaced with synthetic or guessed material.
