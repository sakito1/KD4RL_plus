# AAAI-27 Appendix Reproduction Package Design

## Goal

Extend `CMTFlow_AAAI27/` with a compact `appendix/` subtree that reproduces the
appendix transaction-cost table, Controller case figures, Controller statistics,
and Trader statistics without exposing unrelated exploratory code or caches.

## Authoritative experiment identity

- Dataset labels are `NASDAQ-100` and `CSI-300`.
- Paper-selected runs are NASDAQ-100 seed 49 and CSI-300 seed 90.
- Both markets use the same five-stage progressive training procedure:
  Manager warm-up, Trader warm-up, fixed-interval Manager–Trader stabilization,
  Controller training, and end-to-end alignment.
- Training uses a 0.005% proportional transaction cost. This deliberately
  preserves the Trader's weak daily incremental reward signal.
- The paper main backtest uses 0.01%. Appendix B.1 sweeps 0.005%, 0.01%,
  0.015%, 0.02%, and 0.05% on the locked execution path.
- The phrase “240 model” is not a dataset name. It only identifies the
  CSI-300 seed-90 run by its 240.13% return at the 0.005% reference cost.

## Included appendix scope

### Appendix A

`ARCHITECTURE_AND_TRAINING.md` maps A.1–A.5 to the minimal parent-package
source files and documents the five-stage training workflow. It does not copy
the training implementation into `appendix/`.

### Appendix B

B.1 is reproducible from the packaged daily fixed-path replay. The appendix
script recalculates TR, SR, MDD, CR, and delta TR from daily net log returns.
B.2 and B.3 are excluded because no final paper-authoritative results have been
selected.

### Appendix C

- C.1 produces one two-by-three Controller figure per market. Each row follows
  current state → Base/Adv decision → frozen counterfactual value.
- C.2 contains the Controller adaptive-horizon statistical analysis and renders
  the compact decision-validation table.
- C.3 contains the Trader refinement statistical analysis and renders the
  Active Share/risk-placebo table.

The expected case values use the currently locked cache:

| Market | Case | Date | p(switch) | 20d advantage | 30d advantage |
|---|---|---|---:|---:|---:|
| NASDAQ-100 | Correct Hold | 2025-05-05 | 14.61% | +7.64 pp | +13.17 pp |
| NASDAQ-100 | Correct Switch | 2020-07-06 | 52.39% | +3.50 pp | +3.64 pp |
| CSI-300 | Correct Hold | 2020-11-25 | 23.19% | +14.33 pp | +24.25 pp |
| CSI-300 | Correct Switch | 2021-07-07 | 51.31% | +14.61 pp | +20.56 pp |

## Directory layout

```text
CMTFlow_AAAI27/
└── appendix/
    ├── README.md
    ├── ARCHITECTURE_AND_TRAINING.md
    ├── CLAIM_BOUNDARIES.md
    ├── MODEL_VERSION.json
    ├── configs/controller_cases.json
    ├── code/
    │   ├── analyze_transaction_cost.py
    │   ├── analyze_controller_statistics.py
    │   ├── analyze_trader_statistics.py
    │   ├── plot_controller_cases.py
    │   ├── render_statistical_tables.py
    │   └── run_appendix.py
    ├── inputs/
    │   ├── controller_cases/
    │   ├── controller_statistics/
    │   └── trader_statistics/
    ├── expected/
    │   ├── figures/
    │   └── tables/
    └── tests/test_appendix_package.py
```

## Data flow and disclosure boundary

The directly runnable appendix renderer consumes only packaged derived traces:
daily fixed-path cost replay, four selected case records and curves, and the
minimum statistical summaries/daily series. Full raw market data remains
excluded pending redistribution approval. The full statistical source is
included so an authorized user with `data/full/` can rerun upstream analysis.

Expected figures are written as PDF and PNG. Expected tables are written as
CSV, Markdown, and LaTeX. Plotting code reads generated CSV/JSON inputs; it does
not hard-code final values in plot annotations.

## Statistical claim boundaries

- Cost sensitivity supports robustness of the locked complete execution path,
  not standalone Trader alpha.
- NASDAQ-100 Controller values are descriptive because their confidence
  intervals include zero.
- CSI-300 Controller chosen-action return and drawdown values are statistically
  supported by the reported HAC analysis.
- Trader refinement is non-random and reduces ex-ante risk relative to random
  within-support tilts; frozen-path direct alpha is not claimed.

## Verification

The appendix test checks folder completeness, market/seed identity, training
and evaluation fees, four case dates and probabilities, headline table values,
and generated file presence. The root manifest and integrity checker are
regenerated after appendix files are finalized.
