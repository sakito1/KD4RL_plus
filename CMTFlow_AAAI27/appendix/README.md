# AAAI Appendix Reproduction

This directory contains only the code and derived inputs needed for Appendix A,
B.1, B.2, C.1, C.2, and C.3. B.3 remains absent because no
paper-authoritative multi-day refinement-frequency result has been selected.

## One-command direct reproduction

From the package root:

```bash
python appendix/code/run_appendix.py --output-dir appendix/outputs
```

This command does not load a neural-network checkpoint and does not require raw
market data. It regenerates:

- `tables/transaction_cost_sensitivity.csv`
- `tables/fixed_window_sensitivity.csv`
- `tables/fixed_window_summary.csv`
- `tables/fixed_window_wealth_nasdaq100.csv`
- `tables/fixed_window_wealth_csi300.csv`
- `tables/controller_decision_validation.csv`
- `tables/trader_refinement_validation.csv`
- `tables/appendix_tables.md`
- `tables/appendix_tables.tex`
- `figures/controller_cases_nas.{pdf,png}`
- `figures/controller_cases_sh.{pdf,png}`
- `figures/fixed_window_sensitivity_nasdaq100.{pdf,png}`
- `figures/fixed_window_sensitivity_csi300.{pdf,png}`

Locked reference copies are under `expected/`.

## What each public entry point does

| Entry point | Appendix target | Direct input |
|---|---|---|
| `code/analyze_transaction_cost.py` | B.1 | parent `traces/transaction_cost/tables/*_daily_replay.csv` |
| `code/analyze_fixed_window_sensitivity.py` | B.2 metrics | compact 1–60 day growth/turnover replays and the B.1 Controller replays |
| `code/plot_fixed_window_sensitivity.py` | B.2 figures | generated B.2 summary and wealth matrices |
| `code/plot_controller_cases.py` | C.1 | four JSON case records and `configs/controller_cases.json` |
| `code/analyze_controller_statistics.py` | C.2 full analysis | authorized horizon-30 action traces and market data |
| `code/analyze_trader_statistics.py` | C.3 full analysis | authorized market data, action traces, and optional evaluation artifacts |
| `code/render_statistical_tables.py` | B.1/C.2/C.3 paper tables | compact packaged derived CSVs |
| `code/run_appendix.py` | all directly packaged outputs | the inputs listed above |

The full Controller and Trader analyses live once under
`../src/paper_experiments/`; the appendix entry points import them instead of
duplicating approximately 100 KB of statistical implementation.

## Experiment identity

- NASDAQ-100: seed 49.
- CSI-300: seed 90.
- Both markets use the same five-stage progressive training procedure.
- Training cost: 0.005%.
- Main paper evaluation cost: 0.01%.
- B.1 cost sweep: 0.005%–0.05%.
- B.2: every fixed holding window from 1 through 60 trading days.

The lower training cost is deliberate. It prevents fees from overwhelming the
Trader's weak daily incremental reward signal. Alternative paper fees are
applied only in a fixed-action-path replay.

## B.2 dense fixed holding-window sensitivity

B.2 replaces only the Controller timing rule with deterministic holding windows
of 1–60 trading days. The paper-selected Manager and Trader remain active, and
no model is retrained. The 120 evaluation paths were generated at the original
0.005% environment cost and then mechanically repriced to the paper's 0.01%
evaluation cost from their recorded daily turnover.

The public inputs under `inputs/fixed_window/` contain only:

- a 120-row evaluator metric audit;
- date, daily net growth, and turnover for each fixed window.

They exclude action probabilities, portfolio weights, temporary evaluator
logs, and the 120 full action traces. The result supports high-percentile
adaptive timing without ex-post selection of one constant schedule; it does not
claim that the Controller is best on every metric against every fixed window.

## Model disclosure audit

The appendix itself contains no `.pth` file and none of its direct rendering
code imports PyTorch. Neural-network files remain in the parent package for the
main-paper training/evaluation workflow:

- `checkpoints/nasdaq100/checkpoints/best_model.pth`
- `checkpoints/csi300/checkpoints/best_model.pth`

These are the only two distributed model files. Intermediate stage-resume
checkpoints are not required to regenerate the appendix or evaluate the final
models, so they are excluded. The selected seeds remain recorded inside
configuration and provenance files, not in public filenames.

## Input disclosure

The package includes only:

- fixed-path daily cost replays already used by the main package;
- compact daily growth/turnover replays for the 1–60 day fixed-window analysis;
- four selected Controller cases;
- two Controller headline/statistical CSVs;
- two Trader headline/placebo CSVs.

It excludes full bootstrap draws, exploratory case searches, notebooks,
temporary runtimes, failed runs, and raw price data whose redistribution rights
have not been confirmed.

## Full raw-data reanalysis

The full statistical entry points expose `--help` with their required paths.
They need the licensed data layout described in `../data/README.md`. Direct
appendix rendering remains available without those private inputs.
