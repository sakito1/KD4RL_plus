# AAAI-27 Reproduction Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Build a self-contained `CMTFlow_AAAI27/` directory that reproduces the paper-facing 0.01% CMTFlow metrics and fixes the exact Figure 4 case windows while preserving the original training workflow.

**Architecture:** Add two focused, tested paper utilities: a drift-aware fixed-path transaction-cost replay and an optional fixed-case manifest for the Trader heatmap. Add a deterministic package builder that copies the exact paper code, real checkpoints, command records, traces, expected tables, and documentation into one directory and hashes every packaged artifact.

**Tech Stack:** Python 3.10, NumPy, pandas, matplotlib, PyTorch, pytest, JSON/CSV/Markdown.

---

### Task 1: Restore the 0.01% fixed-path replay source

**Files:**
- Create: `tests/test_transaction_cost_sensitivity.py`
- Create: `paper_experiments/analyze_transaction_cost_sensitivity.py`

- [ ] **Step 1: Write failing drift-aware replay tests**

Test a two-asset example where previous weights drift before the current rebalance,
then assert turnover and higher-cost monotonicity:

```python
def test_replay_uses_price_drifted_previous_weights():
    replay = replay_executed_weight_path(weights, prices, [0.0, 0.001])
    expected = abs(0.5 - 110.0 / 210.0) + abs(0.5 - 100.0 / 210.0)
    assert replay.loc[pd.Timestamp("2020-01-02"), "turnover"] == pytest.approx(expected)


def test_higher_cost_cannot_improve_fixed_path():
    replay = replay_executed_weight_path(weights, prices, [0.00005, 0.0005])
    assert (
        replay["net_growth_tc_0p0500pct"]
        <= replay["net_growth_tc_0p0050pct"] + 1e-15
    ).all()
```

- [ ] **Step 2: Run RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_transaction_cost_sensitivity.py -q
```

Expected: import failure because the analysis module does not exist.

- [ ] **Step 3: Implement the replay kernel and summary**

Implement `cost_label(rate)`, `replay_executed_weight_path(executed, prices,
cost_rates)`, and `summarize_replay(replay, market, seed, cost_rates,
reference_rate)`. The replay must normalize each executed row, drift the
previous target by the previous-to-current price relative, compute full L1
turnover, compute current-to-next gross growth, and apply
`(1 - rate * turnover) * gross_growth`. The summary must calculate TR, annualized
Sharpe, maximum drawdown, annualized-return Calmar, mean turnover, cumulative
charged cost rate, and deltas from the reference row.

The CLI must read the paper-selected action traces, adjusted prices, checkpoint
records, and write daily CSVs, a summary CSV, Markdown, and a hashed manifest.

- [ ] **Step 4: Run GREEN**

Run the focused tests and require all tests to pass.

### Task 2: Lock Figure 4 to the PDF case windows

**Files:**
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py`
- Create: `tests/test_inner_actor_case_manifest.py`
- Create: `configs/aaai27_figure4_cases.json`

- [ ] **Step 1: Write failing fixed-case tests**

Test that a case specification with start/end dates and ordered assets resolves
to exactly those rows and assets, and that an unavailable date or asset raises a
clear `ValueError`.

```python
case = {
    "start_date": "2024-05-13",
    "end_date": "2024-06-25",
    "assets": ["IDXX.O", "ADSK.O"],
}
window = resolve_case_window(tilt, future, case)
assert list(window["idx"]) == list(pd.bdate_range("2024-05-13", "2024-06-25"))
assert window["assets"] == ["IDXX.O", "ADSK.O"]
```

- [ ] **Step 2: Run RED**

Expected: import failure for `resolve_case_window`.

- [ ] **Step 3: Implement manifest support**

Add:

```python
parser.add_argument("--case_manifest", default=None)
```

Load a JSON mapping for `nas` and `sh`; pass the selected case into both the
single-market and combined heatmap data preparation. With no manifest, preserve
the current automatic selection behavior.

The committed manifest must encode:

- CSI-300: 2024-01-23 through 2024-03-12 and the 10 PDF assets.
- Nasdaq-100: 2024-05-13 through 2024-06-25 and the 10 PDF assets.

- [ ] **Step 4: Run GREEN**

Run the focused case-manifest test and the existing inner-actor plotting tests.

### Task 3: Build a deterministic paper-first package

**Files:**
- Create: `tools/build_aaai27_repro_package.py`
- Create: `tests/test_build_aaai27_repro_package.py`

- [ ] **Step 1: Write failing package-builder tests**

Create a temporary fixture repository and assert that:

```python
build_package(source_root, destination, artifact_roots=fixture_artifacts)
assert (destination / "README.md").is_file()
assert (destination / "src/run_hrl_training.py").is_file()
assert (destination / "expected/table1.csv").is_file()
assert (destination / "expected/table2.csv").is_file()
assert (destination / "MANIFEST.json").is_file()
assert not any(path.is_symlink() for path in destination.rglob("*"))
```

Also verify that `MANIFEST.json` contains relative POSIX paths and valid SHA256
values for every packaged regular file except itself.

- [ ] **Step 2: Run RED**

Expected: import failure because the builder does not exist.

- [ ] **Step 3: Implement the builder**

The builder must:

1. create `src/`, `scripts/`, `configs/`, `data/`, `checkpoints/`, `traces/`,
   `expected/`, and `third_party/licenses/`;
2. copy the paper-facing source closure and final command files;
3. copy real checkpoint bytes from the audited model directories;
4. copy exact Figure 3 manifests and paper-selected Figure 4 inputs;
5. write exact PDF Table 1 and Table 2 values;
6. write README, expected-results, data instructions, model manifest, and a
   legal-status note without choosing a license for the user;
7. generate a SHA256 manifest using only relative paths;
8. reject symlinks in the completed package.

- [ ] **Step 4: Run GREEN**

Run the focused builder tests and require all to pass.

### Task 4: Generate and verify `CMTFlow_AAAI27/`

**Files:**
- Create: `CMTFlow_AAAI27/`

- [ ] **Step 1: Re-run the selected cost replay**

Run the restored script at rates 0.005%, 0.01%, 0.015%, 0.02%, and 0.05%.
Verify that the 0.01% rows are Nasdaq 262.49% and CSI-300 237.01% within
rounding tolerance.

- [ ] **Step 2: Build the package**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  tools/build_aaai27_repro_package.py \
  --source-root . \
  --destination CMTFlow_AAAI27
```

- [ ] **Step 3: Verify structure and hashes**

Run the package verifier, ensure there are no symlinks or absolute home paths in
submitted text files, and verify the two final model SHA256 values.

- [ ] **Step 4: Run focused and relevant regression tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_transaction_cost_sensitivity.py \
  tests/test_inner_actor_case_manifest.py \
  tests/test_build_aaai27_repro_package.py \
  paper_experiments/tests/test_controller_case_layout.py -q
```

Do not commit automatically because the existing branch contains unrelated user
changes; report the new files and remaining legal/data-redistribution blockers.
