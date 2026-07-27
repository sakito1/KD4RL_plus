# Controller Case 2×2 Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Generate four 2×2 controller figures from every pairing of two CSI-300 cases and two Nasdaq-100 cases while retaining the existing single-market figures.

**Architecture:** Refactor the existing single-market plot into reusable case-data and axis-rendering helpers. Use those helpers for both the existing 1×2 output and a new 2×2 combined plot, then form the SH/NAS Cartesian product after both markets' cases have been selected.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest

---

## File Structure

- Modify `paper_experiments/run_paper_experiments_final.py`: reusable controller-case rendering, combined plotting, and SH/NAS combination orchestration.
- Modify `paper_experiments/tests/test_controller_case_layout.py`: layout and four-combination regression tests.

### Task 1: Specify the Combined Layout and Pairing

**Files:**
- Modify: `paper_experiments/tests/test_controller_case_layout.py`

- [ ] **Step 1: Write the failing combined-layout test**

Add fixtures for one SH and one NAS case, then call:

```python
figures.plot_combined_controller_case(
    sh_case_id=1,
    sh_case=sh_case,
    sh_portfolio=sh_portfolio,
    sh_actions=sh_actions,
    nas_case_id=2,
    nas_case=nas_case,
    nas_portfolio=nas_portfolio,
    nas_actions=nas_actions,
    out_dir=tmp_path,
)
```

Capture `save_figure` and assert:

```python
assert len(fig.axes) == 4
top_left, top_right, bottom_left, bottom_right = fig.axes
assert top_left.get_position().width == pytest.approx(top_right.get_position().width)
assert top_left.get_position().width == pytest.approx(bottom_left.get_position().width)
assert top_left.get_position().y0 > bottom_left.get_position().y0
assert [text.get_text() for text in fig.texts].count("CSI-300") == 1
assert [text.get_text() for text in fig.texts].count("Nasdaq-100") == 1
assert [text.get_text() for text in fig.texts].count(
    "A. Future return after the switch decision"
) == 1
assert [text.get_text() for text in fig.texts].count(
    "B. Future drawdown under the same frozen window"
) == 1
assert len(fig.legends) == 1
assert captured["path"].name == "controller_case_combined_sh01_nas02"
```

- [ ] **Step 2: Write the failing Cartesian-product test**

Exercise a small pure helper:

```python
pairs = figures.controller_case_combinations(
    [(1, sh_case_1), (2, sh_case_2)],
    [(1, nas_case_1), (2, nas_case_2)],
)
assert [(sh_id, nas_id) for sh_id, _, nas_id, _ in pairs] == [
    (1, 1), (1, 2), (2, 1), (2, 2)
]
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_controller_case_layout.py -q
```

Expected: FAIL because `plot_combined_controller_case` and
`controller_case_combinations` do not exist.

- [ ] **Step 4: Commit the failing tests**

```bash
git add paper_experiments/tests/test_controller_case_layout.py
git commit -m "test: specify combined controller case grids"
```

### Task 2: Extract Reusable Controller Panel Rendering

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py:931-1087`
- Test: `paper_experiments/tests/test_controller_case_layout.py`

- [ ] **Step 1: Add a case-data preparation helper**

Create:

```python
def prepare_controller_case_plot_data(
    market: str,
    case_id: int,
    case: pd.Series,
    portfolio: pd.DataFrame,
    actions: pd.DataFrame,
) -> Dict[str, object]:
    ...
```

Move curve parsing, date alignment, day-axis creation, return/drawdown
calculation, and annotation metrics from `plot_controller_case` into this
helper. Return all arrays and labels needed for drawing, including
`date_range_label`, `plot_days`, `realized_horizon`, and the summary row.

- [ ] **Step 2: Add a row renderer**

Create:

```python
def draw_controller_case_panels(
    ax_return,
    ax_drawdown,
    data: Dict[str, object],
    *,
    show_ylabels: bool = True,
) -> Dict[str, object]:
    ...
```

Move the current curves, fills, endpoints, annotations, axis cleanup, ticks,
and date labels into this helper. Return legend handles keyed by label so the
caller can create one shared legend.

- [ ] **Step 3: Rebuild the existing single-market function from the helpers**

Keep the public signature and output name unchanged:

```python
def plot_controller_case(...):
    data = prepare_controller_case_plot_data(...)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    legend_items = draw_controller_case_panels(axes[0], axes[1], data)
    ...
    save_figure(fig, out_dir / f"controller_case_{market}_{case_id:02d}")
    return data["summary"]
```

- [ ] **Step 4: Run the existing single-market layout test**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_controller_case_layout.py::test_controller_case_uses_two_equal_side_by_side_panels_and_shared_legend -q
```

Expected: PASS.

### Task 3: Implement Combined Plotting and Four Pairings

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py`
- Test: `paper_experiments/tests/test_controller_case_layout.py`

- [ ] **Step 1: Implement deterministic pairing**

Add:

```python
def controller_case_combinations(sh_cases, nas_cases):
    return [
        (sh_id, sh_case, nas_id, nas_case)
        for sh_id, sh_case in sh_cases
        for nas_id, nas_case in nas_cases
    ]
```

- [ ] **Step 2: Implement the 2×2 combined figure**

Add `plot_combined_controller_case(...)`. It creates a `2 × 2` equal-size
grid, renders SH on the first row and NAS on the second, and places:

```python
fig.text(0.012, 0.64, "CSI-300", rotation=90, ...)
fig.text(0.012, 0.27, "Nasdaq-100", rotation=90, ...)
fig.text(0.29, 0.025, "A. Future return after the switch decision", ...)
fig.text(0.75, 0.025, "B. Future drawdown under the same frozen window", ...)
```

Use a single top legend and save as:

```python
out_dir / f"controller_case_combined_sh{sh_case_id:02d}_nas{nas_case_id:02d}"
```

- [ ] **Step 3: Integrate combination generation**

In `controller_experiment`, retain selected cases by market:

```python
selected_cases = {}
selected_inputs = {}
...
selected_cases[market] = [
    (idx, case) for idx, (_, case) in enumerate(cases.iterrows(), start=1)
]
selected_inputs[market] = (portfolio, case_actions)
```

After the market loop, when both `sh` and `nas` exist, iterate
`controller_case_combinations(selected_cases["sh"], selected_cases["nas"])`
and call `plot_combined_controller_case` for all four combinations.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_controller_case_layout.py \
  paper_experiments/tests/test_root_paper_experiments_entry.py \
  paper_experiments/tests/test_final_formatting.py \
  paper_experiments/tests/test_switch_endpoint_distribution.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit implementation**

```bash
git add paper_experiments/run_paper_experiments_final.py \
  paper_experiments/tests/test_controller_case_layout.py
git commit -m "feat: generate combined controller case grids"
```

### Task 4: Generate and Inspect the Four Candidate Figures

**Files:**
- Generated: `paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_combined_sh*_nas*.png`

- [ ] **Step 1: Generate figures with the confirmed models**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  run_paper_experiments_final.py \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --skip_fixed_eval
```

Expected output includes `controller cases: 4` and creates:

```text
controller_case_combined_sh01_nas01.png
controller_case_combined_sh01_nas02.png
controller_case_combined_sh02_nas01.png
controller_case_combined_sh02_nas02.png
```

- [ ] **Step 2: Inspect all four PNG files**

Verify that row labels, A/B captions, shared legend, date ranges, annotations,
and plot boundaries are visible without clipping or overlap.

- [ ] **Step 3: Run final repository checks**

Run:

```bash
git diff --check
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_controller_case_layout.py \
  paper_experiments/tests/test_root_paper_experiments_entry.py \
  paper_experiments/tests/test_final_formatting.py \
  paper_experiments/tests/test_switch_endpoint_distribution.py -q
```

Expected: no whitespace errors and all tests PASS.

