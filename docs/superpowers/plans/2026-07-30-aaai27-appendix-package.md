# AAAI-27 Appendix Reproduction Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a minimal, verified appendix reproduction subtree to the existing
`CMTFlow_AAAI27/` package.

**Architecture:** The appendix uses compact derived inputs for one-command
rendering while retaining the full analysis entry points needed for authorized
raw-data reruns. Generated tables are the sole data source for plotting and
LaTeX rendering.

**Tech Stack:** Python 3, NumPy, pandas, SciPy, statsmodels, matplotlib, pytest.

---

### Task 1: Lock package requirements with tests

**Files:**
- Create: `CMTFlow_AAAI27/appendix/tests/test_appendix_package.py`

- [ ] Write tests that assert the two markets/seeds, fee split, five training
  stages, four case dates/probabilities, and expected table values.
- [ ] Run `pytest CMTFlow_AAAI27/appendix/tests/test_appendix_package.py -v`
  and confirm it fails because `appendix/` does not yet exist.

### Task 2: Correct model and fee provenance

**Files:**
- Modify: `CMTFlow_AAAI27/MODEL_PROVENANCE.md`
- Modify: `CMTFlow_AAAI27/README.md`
- Create: `CMTFlow_AAAI27/appendix/MODEL_VERSION.json`
- Create: `CMTFlow_AAAI27/appendix/ARCHITECTURE_AND_TRAINING.md`

- [ ] Replace ambiguous “CSI 240 model” wording with CSI-300 seed 90 and state
  that 240.13% is only the 0.005% reference-return label.
- [ ] Record NASDAQ seed 49, CSI seed 90, five training stages,
  `training_cost_rate=0.00005`, and `paper_evaluation_cost_rate=0.0001`.
- [ ] Run the identity and provenance tests and confirm they pass.

### Task 3: Add appendix analyses and rendering

**Files:**
- Create: `CMTFlow_AAAI27/appendix/code/analyze_transaction_cost.py`
- Create: `CMTFlow_AAAI27/appendix/code/analyze_controller_statistics.py`
- Create: `CMTFlow_AAAI27/appendix/code/analyze_trader_statistics.py`
- Create: `CMTFlow_AAAI27/appendix/code/plot_controller_cases.py`
- Create: `CMTFlow_AAAI27/appendix/code/render_statistical_tables.py`
- Create: `CMTFlow_AAAI27/appendix/code/run_appendix.py`

- [ ] Implement cost aggregation from packaged `net_log_return_*` columns using
  annualized Sharpe, peak-to-trough MDD, and Calmar definitions.
- [ ] Restore the audited Controller adaptive-timing analysis source from commit
  `ea1491a` and keep its deterministic seeds and HAC/permutation logic.
- [ ] Copy the audited Trader refinement analysis source and keep only its
  public analysis entry point.
- [ ] Implement the two-market two-by-three case renderer from a case manifest.
- [ ] Implement CSV-to-Markdown/LaTeX renderers for B.1, C.2, and C.3.
- [ ] Implement `run_appendix.py` to generate all directly reproducible outputs.

### Task 4: Package only necessary derived inputs and expected outputs

**Files:**
- Create: `CMTFlow_AAAI27/appendix/configs/controller_cases.json`
- Create: `CMTFlow_AAAI27/appendix/inputs/controller_cases/*.csv`
- Create: `CMTFlow_AAAI27/appendix/inputs/controller_statistics/*.csv`
- Create: `CMTFlow_AAAI27/appendix/inputs/trader_statistics/*.csv`
- Create: `CMTFlow_AAAI27/appendix/expected/tables/*`
- Create: `CMTFlow_AAAI27/appendix/expected/figures/*`

- [ ] Extract only the four case rows, their pre-decision paths, Base/Adv
  decomposition, and 20/30-day frozen curves.
- [ ] Include Controller headline and placebo tables and Trader configuration
  and placebo tables; exclude bootstrap dumps and unrelated robustness tables.
- [ ] Run `python CMTFlow_AAAI27/appendix/code/run_appendix.py` and compare
  generated outputs with the locked values in the design spec.

### Task 5: Document, inventory, and verify the final folder

**Files:**
- Create: `CMTFlow_AAAI27/appendix/README.md`
- Create: `CMTFlow_AAAI27/appendix/CLAIM_BOUNDARIES.md`
- Modify: `CMTFlow_AAAI27/PACKAGE_STATUS.md`
- Modify: `CMTFlow_AAAI27/scripts/verify_package.py`
- Modify: `CMTFlow_AAAI27/MANIFEST.json`

- [ ] Document direct-render commands and full raw-data rerun boundaries.
- [ ] Add appendix semantic checks to the root verifier.
- [ ] Regenerate SHA-256 entries for every regular package file except the
  manifest itself.
- [ ] Run the appendix tests, root integrity checker, all appendix entry-point
  `--help` checks, and a full one-command render.
- [ ] Inspect the complete folder inventory for symlinks, absolute paths,
  oversized caches, secrets, and ambiguous CSI naming.

## Verification

```bash
pytest CMTFlow_AAAI27/appendix/tests/test_appendix_package.py -v
python CMTFlow_AAAI27/appendix/code/run_appendix.py
python CMTFlow_AAAI27/scripts/verify_package.py
find CMTFlow_AAAI27 -type l -print
rg -n "/home/|CSI-240|CSI 240" CMTFlow_AAAI27
```

Expected: tests pass; all tables and four-case figures are regenerated; the
manifest verifies; no symlinks, absolute private paths, or CSI-240 dataset label
remain.

**Next skill:** `$superpower-executing-plans`
