# Expected Results

The authoritative expected values are stored in `expected/table1.csv` and
`expected/table2.csv`.

For the fixed-path cost replay, rounding to two decimals must give:

| Market | Cost | TR | SR | MDD | CR |
|---|---:|---:|---:|---:|---:|
| Nasdaq-100 | 0.01% | 262.49% | 1.14 | 18.66% | 1.41 |
| CSI-300 | 0.01% | 237.01% | 1.24 | 22.91% | 1.18 |

CSV-level floating-point differences below `1e-8` are acceptable. Paper table
comparisons use the displayed two-decimal values.

## Appendix headline results

- B.1 transaction-cost sensitivity:
  `appendix/expected/tables/transaction_cost_sensitivity.csv`
- B.2 dense fixed holding-window sensitivity:
  `appendix/expected/tables/fixed_window_{sensitivity,summary}.csv` and
  `appendix/expected/figures/fixed_window_sensitivity_{nasdaq100,csi300}.{pdf,png}`
- C.1 Controller cases:
  `appendix/expected/figures/controller_cases_{nas,sh}.{pdf,png}`
- C.2 Controller statistics:
  `appendix/expected/tables/controller_decision_validation.csv`
- C.3 Trader statistics:
  `appendix/expected/tables/trader_refinement_validation.csv`

Run `python appendix/code/run_appendix.py` to regenerate the directly packaged
appendix tables and case figures.

At the 0.01% paper cost, the B.2 Controller win counts against 60 fixed windows
are:

| Market | TR | SR | MDD | CR |
|---|---:|---:|---:|---:|
| Nasdaq-100 | 45/60 | 43/60 | 59/60 | 57/60 |
| CSI-300 | 59/60 | 59/60 | 43/60 | 58/60 |

These counts are recomputed from
`appendix/expected/tables/fixed_window_sensitivity.csv`, not hard-coded in the
plotting code.
