# Controller Base Filter Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Generate a verified two-market Base-filter case figure and evidence report.

**Architecture:** A standalone plotting script joins existing Controller traces
with frozen counterfactual trajectories, computes the exact Base–Adv
decomposition, renders one row per market, and writes a machine-readable summary
plus a Chinese report.

**Tech Stack:** Python, pandas, NumPy, SciPy, Matplotlib.

---

### Task 1: Case data extraction and invariants

**Files:**
- Create: `scripts/plot_controller_base_filter_cases.py`
- Create: `tests/test_plot_controller_base_filter_cases.py`

- [ ] Add pure helpers for the Adv correction, sigmoid probability, curve parsing,
  case extraction, and positive-Adv blocked counts.
- [ ] Test exact NASDAQ-100 and CSI-300 case values, formula identities, actions,
  and 20/30-day outcome directions.

### Task 2: Figure and report

**Files:**
- Modify: `scripts/plot_controller_base_filter_cases.py`
- Create: `reproduced_outputs/controller_base_filter_cases/*`

- [ ] Render the approved 2×3 layout with consistent colors and labels.
- [ ] Write the case-summary CSV and Chinese report with the population-evidence
  boundary stated explicitly.

### Task 3: Verification

- [ ] Run the focused tests.
- [ ] Run the plotting script from raw cached traces.
- [ ] Visually inspect the PNG.
- [ ] Assert formula errors below `5e-7`, both final probabilities below `0.5`,
  both 20/30-day candidate-minus-Hold gaps below zero, and output files nonempty.

