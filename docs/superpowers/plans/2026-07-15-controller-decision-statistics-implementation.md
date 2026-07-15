# Controller Decision Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Generate decision-level and realized-segment Hold-vs-Switch statistics from the final Nasdaq-100 and CSI-300 Controller traces without retraining models.

**Architecture:** Add one focused analysis module that parses the existing frozen counterfactual curves, computes one-step chosen-action advantage and next-revision segment outcomes, applies calendar-order moving-block bootstrap intervals, and writes CSV plus Markdown artifacts. Keep the analysis separate from the existing experiment-generation script so current paper outputs are untouched.

**Tech Stack:** Python 3, pandas, NumPy, pytest, existing CSV/JSON counterfactual traces.

---

### Task 1: Specify Event-Level Calculations

**Files:**
- Create: `tests/test_controller_decision_statistics.py`
- Create: `paper_experiments/controller_decision_statistics.py`

- [ ] **Step 1: Write failing tests for decision sign orientation**

Create synthetic Hold/Switch curves where Switch is better, assert that a Switch
decision has positive chosen-action advantage and a Hold decision has negative
chosen-action advantage.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_controller_decision_statistics.py -q
```

Expected: import failure because the analysis module does not exist.

- [ ] **Step 3: Implement curve parsing and decision events**

Implement `parse_curve`, `max_drawdown`, and `build_decision_events`. Filter to
`decision_type == "free_decision"`, use the first post-decision curve point, and
write both raw Switch-minus-Hold and action-oriented chosen advantage.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: decision tests pass.

### Task 2: Add Realized-Segment Events

**Files:**
- Modify: `tests/test_controller_decision_statistics.py`
- Modify: `paper_experiments/controller_decision_statistics.py`

- [ ] **Step 1: Write a failing next-revision boundary test**

Use synthetic decisions at steps 0, 2, and 5. Assert that a free switch at step 0
uses a two-day segment ending at the revision on step 2, including forced revisions
as valid endpoints.

- [ ] **Step 2: Verify the new test fails for missing behavior**

Run the focused pytest command and confirm failure at the segment API.

- [ ] **Step 3: Implement `build_segment_events`**

For each free switch, locate the next row with `is_switch == 1`, cap at available
curve length/test end, compute endpoint Switch-minus-Hold return and
Hold-minus-Switch MDD, and retain duration metadata.

- [ ] **Step 4: Verify the focused suite passes**

Run the focused pytest command. Expected: all event tests pass.

### Task 3: Add Summaries and Dependence-Aware Intervals

**Files:**
- Modify: `tests/test_controller_decision_statistics.py`
- Modify: `paper_experiments/controller_decision_statistics.py`

- [ ] **Step 1: Write failing deterministic-bootstrap and summary tests**

Assert fixed-seed interval reproducibility, separate Hold/Switch positive ratios,
balanced hit rate, event counts, and one-day segment fraction.

- [ ] **Step 2: Verify RED**

Run the focused pytest command and confirm the missing summary functions fail.

- [ ] **Step 3: Implement dense-calendar moving-block bootstrap summaries**

Implement deterministic 30-day circular block resampling with NaN-aware group
statistics. Add decision and segment summary builders with mean, median, positive
ratio, lower/upper 95% interval, duration quartiles, and one-day fraction.

- [ ] **Step 4: Verify GREEN**

Run the focused pytest command. Expected: all summary tests pass.

### Task 4: Generate Final Artifacts

**Files:**
- Modify: `paper_experiments/controller_decision_statistics.py`
- Create: `paper_experiments_outputs/controller_decision_statistics/*.csv`
- Create: `paper_experiments_outputs/controller_decision_statistics/CONTROLLER_DECISION_STATISTICS.md`

- [ ] **Step 1: Add a CLI with explicit final-trace inputs**

Default inputs are the final cached Nasdaq seed-49 and CSI-300 seed-90 action traces.
Expose output directory, bootstrap repetitions, block length, and bootstrap seed.

- [ ] **Step 2: Run the generator**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m paper_experiments.controller_decision_statistics \
  --output-dir paper_experiments_outputs/controller_decision_statistics \
  --bootstrap-reps 5000 --block-length 30 --bootstrap-seed 20260715
```

Expected: four CSV files and one Markdown report for both markets.

- [ ] **Step 3: Verify result integrity**

Check that decision counts equal all free decisions, segment counts equal free
switches with usable endpoints, all confidence bounds are finite, and rerunning the
command produces identical checksums.

- [ ] **Step 4: Run regression verification**

Run the focused tests plus `git diff --check` on the new source, test, specification,
plan, and generated Markdown/CSV artifacts.
