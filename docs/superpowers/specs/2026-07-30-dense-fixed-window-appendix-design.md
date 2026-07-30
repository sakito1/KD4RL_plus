# Dense Fixed Holding-Window Appendix Design

## Goal

Add the previously performed dense fixed holding-window experiment to the
paper-facing reproduction package, while removing only model code that is
provably unused by the paper-selected checkpoints and workflows.

The result belongs in:

`CMTFlow_AAAI27/appendix/`

It is Appendix B.2, not a new main-paper experiment.

## Authoritative experiment identity

- Markets: NASDAQ-100 and CSI-300.
- Paper-selected runs: NASDAQ-100 seed 49 and CSI-300 seed 90.
- Both markets use the paper's five-stage progressive training procedure.
- Training remains unchanged and uses a 0.005% transaction cost so that fees do
  not suppress the Trader's weak daily refinement signal.
- The Appendix comparison uses the main-paper evaluation cost of 0.01%.
- The model checkpoints, data splits, Manager, and Trader are frozen.
- Only the Controller is replaced by a deterministic fixed holding-window rule.

The seed is provenance metadata. It must not appear in public output filenames.

## Appendix B.2 protocol

Evaluate every integer holding window from 1 through 60 days for each market.
This produces 120 evaluation-only paths:

1. Reconstruct the candidate portfolio on the first evaluation day.
2. Hold it for the selected fixed number of trading days.
3. Reconstruct again when that holding period expires.
4. Keep the paper-selected Manager and Trader active throughout the path.
5. Replay the resulting fixed path from the 0.005% environment cost to the
   paper evaluation cost of 0.01%, using the recorded daily turnover.

For each fixed window, report total return (TR), Sharpe ratio (SR), maximum
drawdown (MDD), and Calmar ratio (CR). Compare the 60 fixed-window values with
the complete CMTFlow Controller result from the same market.

The summary reports:

- the Controller value for each metric;
- the best and median fixed-window values;
- the best fixed-window length;
- how many of the 60 fixed windows the Controller outperforms;
- the corresponding percentage/percentile.

The comparison is a schedule-sensitivity analysis. It supports the claim that
adaptive Controller timing is robust relative to dense deterministic schedules;
it does not claim that every Controller metric must dominate every fixed
window.

## Validation and fallback rule

Before the 120-path run, representative windows are rerun against the current
paper-locked models:

| Market | Window | Current rerun | Historical reference |
|---|---:|---:|---:|
| NASDAQ-100 | 5 days | TR 219.84% | TR 219.84% |
| NASDAQ-100 | 8 days | TR 336.91%, CR 1.52 | TR 336.91%, CR 1.52 |
| CSI-300 | 50 days | TR 292.25%, SR 1.40 | TR 292.16%, SR 1.40 |

The representative results agree within display-level tolerance, so the
current-model full rerun is authoritative. If the complete output later fails
structural checks or materially contradicts these anchors, the public expected
figure must not be overwritten; retain the previous figure and report the
failure.

## Public disclosure boundary

Publish only the files required to audit and render Appendix B.2:

```text
appendix/
├── code/
│   ├── analyze_fixed_window_sensitivity.py
│   └── plot_fixed_window_sensitivity.py
├── inputs/fixed_window/
│   ├── fixed_window_metrics.csv
│   ├── daily_replay_nasdaq100.csv
│   └── daily_replay_csi300.csv
└── expected/
    ├── tables/fixed_window_sensitivity.csv
    ├── tables/fixed_window_summary.csv
    └── figures/fixed_window_sensitivity_{nasdaq100,csi300}.{pdf,png}
```

Do not publish 120 action-level traces, temporary evaluator runtimes, duplicate
checkpoints, exploratory plots, or raw licensed market data. The input metrics
table has one row per market and fixed window at the original 0.005% evaluator
cost. Each compact daily replay contains only date plus the per-window net
growth and turnover columns needed to reprice the paths to 0.01% and redraw the
figure; asset weights and action diagnostics are excluded.

`run_appendix.py` regenerates the B.2 tables and figures together with the
existing Appendix outputs.

## Figure design

Use one identically structured figure per market and preserve the historical
figure's reading structure:

- the left panel overlays all 60 fixed-window wealth paths as thin purple lines;
- the complete Controller appears as a prominent red line;
- the 30-day fixed-HRL reference appears as a black dashed line;
- the left-panel box reports Controller win counts for TR, SR, MDD, and CR;
- the right panel reports the same four win percentages as horizontal bars.

The plot reads CSV inputs and contains no hard-coded final result values.
Expected outputs are supplied as both PDF and PNG.

## Dead-code cleanup boundary

Remove only code with zero runtime, checkpoint, configuration, or paper-analysis
dependency:

- the uninstantiated `CausalConv1dBlock`;
- unused constructor parameters internal to the selected model implementation;
- unused diagnostic attention assignments where no public analysis consumes
  them.

Keep all Manager, Trader, and Controller auxiliary prediction heads used by the
five-stage objectives. Keep checkpoint-dependent module branches and parser
arguments needed to load the locked command records. Do not rewrite or
sanitize the two final checkpoint files.

## Verification

Tests are written before implementation and cover:

- exactly 60 windows for each of the two markets;
- windows are the integers 1–60 with no gaps or duplicates;
- the 0.005% replay reconstructs the evaluator's original metrics;
- the 0.01% replay applies the declared turnover-based fee;
- summary counts and best/median values are recomputed from the public CSV;
- plots contain both Controller references and fixed-window results;
- one-command Appendix rendering creates all B.2 files;
- public filenames contain no seed labels;
- only the two paper-selected `.pth` files remain in the package;
- removed dead components have no remaining reference;
- all existing Appendix and root integrity tests still pass.
