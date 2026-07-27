# Trader Controller-Style Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Reformat the combined Trader Refinement figure so each market is one row, the two metrics are columns, and the two column captions appear once below the figure.

**Architecture:** Keep the existing prepared market-panel dictionaries and shared scale calculation. Change only the axes-to-market mapping and figure-level labels inside `plot_combined_market_heatmaps`; preserve the canonical runner integration and output basename.

**Tech Stack:** Python, Matplotlib, pandas, pytest.

---

### Task 1: Lock the Controller-style layout in a regression test

**Files:**
- Modify: `paper_experiments/tests/test_trader_refinement_layout.py`

- [ ] **Step 1: Write the failing assertions**

Update the existing layout test so the four image axes are interpreted as
`CSI return`, `CSI tilt`, `Nasdaq return`, and `Nasdaq tilt`. Assert:

```python
assert top_left.get_position().x0 < top_right.get_position().x0
assert top_left.get_position().y0 > bottom_left.get_position().y0
assert not any(axis.get_title() for axis in image_axes)
assert figure_text.count("A. Future 7-day relative return") == 1
assert figure_text.count("B. Refinement tilt") == 1
assert figure_text.count("CSI-300") == 1
assert figure_text.count("Nasdaq-100") == 1
```

Also assert that left-column images share a color limit, right-column images
share a color limit, and both colorbar axes are outside the heatmap columns.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_trader_refinement_layout.py -q
```

Expected: failure because the current implementation maps markets to columns
and repeats metric titles.

### Task 2: Implement the row-market, column-metric layout

**Files:**
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py`
- Test: `paper_experiments/tests/test_trader_refinement_layout.py`

- [ ] **Step 1: Change the market order and axes mapping**

Use `("sh", "nas")` as the row order. Draw `fut_pct` into `axes[row, 0]` and
`tilt_pct` into `axes[row, 1]`. Place one horizontal shared colorbar above each
metric column without covering either heatmap.

- [ ] **Step 2: Replace repeated axes titles with figure labels**

Remove per-axis titles. Add each market label once to the left of its row, and
add the following centered below the two columns:

```python
f"A. Future {future_horizon}-day relative return"
"B. Refinement tilt"
```

Keep the existing enlarged font sizes, shared metric scales, date ticks for
both market rows, 240-DPI saving, and output basename.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_trader_refinement_layout.py -q
```

Expected: all tests pass.

### Task 3: Regenerate and verify the paper output

**Files:**
- Generated: `paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/trader_refinement_two_markets.png`
- Generated: `paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/trader_refinement_two_markets.pdf`

- [ ] **Step 1: Run the canonical entrypoint**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  run_paper_experiments_final.py \
  --markets sh nas --seeds sh:90 nas:49 --device cuda --skip_fixed_eval
```

Expected: exit code 0 and the final output summary.

- [ ] **Step 2: Inspect the PNG**

Confirm that CSI-300 is the upper row, Nasdaq-100 is the lower row, titles do
not overlap heatmaps, A/B captions appear once below the columns, and colorbars
do not clip.

- [ ] **Step 3: Run regression verification**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_trader_refinement_layout.py \
  paper_experiments/tests/test_controller_case_layout.py \
  paper_experiments/tests/test_root_paper_experiments_entry.py \
  paper_experiments/tests/test_final_formatting.py \
  paper_experiments/tests/test_switch_endpoint_distribution.py -q
git diff --check
```

Expected: all tests pass and `git diff --check` produces no output.

- [ ] **Step 4: Commit only the intended files**

```bash
git add \
  paper_experiments/plot_inner_actor_base_adjustment.py \
  paper_experiments/tests/test_trader_refinement_layout.py \
  docs/superpowers/specs/2026-07-27-trader-controller-style-layout-design.md \
  docs/superpowers/plans/2026-07-27-trader-controller-style-layout.md
git commit -m "refine: match trader layout to controller"
```
