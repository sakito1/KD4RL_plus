# Reference-Like Case Typography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Redraw the four existing Controller and Inner-Actor case figures with restrained reference-like typography while preserving all figure content and geometry.

**Architecture:** Change only Matplotlib `fontsize` and `fontweight` arguments in the two authoritative case-plot functions. Lock the approved hierarchy with focused regression tests, render from existing cached traces, and copy the same four outputs to the stable manuscript paths.

**Tech Stack:** Python, Matplotlib, pandas, unittest, cached experiment CSV traces.

---

### Task 1: Lock the restrained typography contract

**Files:**
- Modify: `tests/test_paper_figure_readability.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Replace the oversized Controller assertions**

Assert an 18pt semibold suptitle, 13pt semibold panel titles, 11pt regular axis labels, 10pt regular ticks/legend, and 11pt semibold metric and endpoint annotations. Preserve the existing assertions for two rows, exact text, canvas height, legend separation, and endpoint containment.

- [ ] **Step 2: Replace the oversized Inner assertions**

Assert an 18pt semibold suptitle, 13pt semibold row titles, 10pt regular asset/tick labels, 11pt regular axis/colorbar labels, and 11pt semibold summary/bar annotations. Preserve the existing assertions for four rows, exact titles, canvas dimensions, colorbars, and summary placement.

- [ ] **Step 3: Run the two focused tests and confirm red state**

Run:

```bash
PYTHONPATH=. /home/tongwenxuan/conda/envs/xuangu/bin/python -m unittest \
  tests.test_paper_figure_readability.PaperFigureReadabilityTests.test_controller_case_has_two_rows_and_trading_day_axis \
  tests.test_paper_figure_readability.PaperFigureReadabilityTests.test_inner_case_keeps_four_rows_with_compact_titles -v
```

Expected: both fail because the scripts still use 20–38pt bold typography.

### Task 2: Apply the approved font hierarchy

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py:1006-1150`
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py:265-360`

- [ ] **Step 1: Restyle Controller text only**

In `plot_controller_case`, set suptitle/panel/axis/tick-legend/annotation sizes to `18/13/11/10/11` pt. Use `semibold` only for the suptitle, panel titles, endpoint labels, and metric annotations; use `normal` for axes, ticks, and legend. Do not modify strings, paths, fills, lines, annotations, axes limits, canvas geometry, spacing, cases, or output paths.

- [ ] **Step 2: Restyle Inner-Actor text only**

In `plot_market`, set suptitle/row-title/axis/asset-tick/annotation sizes to `18/13/11/10/11` pt. Use `semibold` only for the suptitle, row titles, summary, and bar annotations; use `normal` for axes, ticks, asset codes, and colorbars. Do not modify data, panels, color scales, bars, strings, canvas geometry, spacing, selected windows, or output paths.

- [ ] **Step 3: Run the focused test file**

Run:

```bash
PYTHONPATH=. /home/tongwenxuan/conda/envs/xuangu/bin/python -m unittest discover \
  -s tests -p 'test_paper_figure_readability.py' -v
```

Expected: all seven tests pass.

### Task 3: Render and verify the four stable figures

**Files:**
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png`
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png`
- Replace: `paper_full_evidence_edit/figures/explainability/inner_actor_nas.png`
- Replace: `paper_full_evidence_edit/figures/explainability/inner_actor_sh.png`

- [ ] **Step 1: Render Controller figures from cached counterfactual actions**

Use the existing selected-case CSVs and cached horizon-30 action CSVs for Nasdaq-100 seed 49 and CSI-300 seed 90. Do not run training or evaluation.

- [ ] **Step 2: Render Inner figures from cached traces**

Run `plot_inner_actor_base_adjustment.py --markets nas sh --seeds nas:49 sh:90` without `--force_eval`.

- [ ] **Step 3: Copy the first Controller case and both Inner outputs**

Replace only the four stable manuscript PNG paths listed above.

- [ ] **Step 4: Verify source and rendered outputs**

Run the focused seven-test suite, `py_compile` on both plotting scripts, and scoped `git diff --check`. Inspect all four PNGs at original resolution for clipping, overlap, and a restrained typography hierarchy. Leave all changes unstaged and uncommitted.

### Task 4: Compact the four case-figure layouts

**Files:**
- Modify: `tests/test_paper_figure_readability.py`
- Modify: `paper_experiments/run_paper_experiments_final.py:1006-1135`
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py:265-358`
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png`
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png`
- Replace: `paper_full_evidence_edit/figures/explainability/inner_actor_nas.png`
- Replace: `paper_full_evidence_edit/figures/explainability/inner_actor_sh.png`

- [ ] **Step 1: Write failing compact-geometry assertions**

Require Controller size `(7.2, 5.6)`, axes top above `0.78`, and the vertical gap between its two axes below `0.18`. Require Inner size `(11.5, 8.8)` and the maximum vertical gap between adjacent evidence axes below `0.10`. Retain all existing content and typography assertions.

- [ ] **Step 2: Run the focused test file and verify red state**

```bash
PYTHONPATH=. /home/tongwenxuan/conda/envs/xuangu/bin/python -m unittest discover \
  -s tests -p 'test_paper_figure_readability.py' -v
```

Expected: the Controller and Inner tests fail on the old `6.8/10.5` heights and wide subplot gaps.

- [ ] **Step 3: Apply the compact Controller geometry**

Set `figsize=(7.2, 5.6)`, GridSpec and final `hspace=0.34`, legend anchor near `0.88`, and subplot `top` near `0.80`. Keep the approved typography and all plotted content unchanged.

- [ ] **Step 4: Apply the compact Inner geometry**

Set `figsize=(11.5, 8.8)`, GridSpec and final `hspace=0.32`, and subplot `top` near `0.92`. Adjust only the fourth-row summary y-position if required to avoid a title collision. Keep the approved typography and all evidence content unchanged.

- [ ] **Step 5: Verify green state and regenerate cached outputs**

Run the seven focused tests, compile both scripts, render both markets from the existing cached traces, replace the same four stable PNGs, and inspect each at original resolution. Do not train, evaluate, stage, or commit.

### Task 5: Separate Nasdaq endpoint labels from the curves

**Files:**
- Modify: `tests/test_paper_figure_readability.py`
- Modify: `paper_experiments/run_paper_experiments_final.py:1042-1065`
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png`

- [ ] **Step 1: Write failing NAS offset assertions**

In the existing Nasdaq Controller test, assert that the Hold annotation has
`xyann == (-8, -22)` and the Switch annotation has `xyann == (-8, 12)`.
Retain the endpoint-containment assertions.

- [ ] **Step 2: Run the focused suite and verify red state**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /home/tongwenxuan/conda/envs/xuangu/bin/python -m unittest discover \
  -s tests -p 'test_paper_figure_readability.py' -v
```

Expected: the Nasdaq Controller test fails because the current offsets are
`(-8, -14)` and `(-8, 5)`.

- [ ] **Step 3: Apply market-specific endpoint offsets**

Use `hold_offset = -22` and `switch_offset = 12` only when `market == "nas"`.
Preserve the current conditional `-14/5` offset behavior for all other
markets. Do not change horizontal offsets, label text, typography, curves, or
geometry.

- [ ] **Step 4: Verify and export only the Nasdaq case**

Run the seven focused tests, compile the Controller plotting script to
`/dev/shm` because the root filesystem is full, regenerate the selected
Nasdaq-100 Controller cases from the seed-49 cached actions, replace only
`controller_switch_case_nas.png`, and visually confirm curve clearance and
axes containment. Leave changes unstaged and uncommitted.

## Verification

- Figure content is unchanged; compact canvas sizes are `7.2 x 5.6 in` for
  Controller and `11.5 x 8.8 in` for Inner Actor.
- Typography matches the approved `18/13/11/10/11` hierarchy.
- Seven focused tests pass and both scripts compile.
- Four stable manuscript PNGs exist and pass visual inspection.

**Next skill:** `$superpower-executing-plans`
