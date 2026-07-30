# Dense Fixed Holding-Window Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add a compact, reproducible Appendix B.2 for the paper-locked 1–60-day fixed holding-window experiment and remove only proven model dead code.

**Architecture:** Existing author-side evaluation code loads each selected final checkpoint once per market and evaluates all 60 deterministic Controller replacements while leaving Manager and Trader active. A compact public daily replay stores only date, 0.005%-cost net growth, and turnover; Appendix code reprices those paths to 0.01%, computes tables, and renders the historical two-panel figure without loading a checkpoint.

**Tech Stack:** Python 3.11, PyTorch 2.4 for author-side evaluation, pandas/numpy for compact data preparation, Python standard-library statistics for packaged metrics, matplotlib for PDF/PNG figures, unittest/pytest for verification.

---

## File structure

Create:

- `CMTFlow_AAAI27/appendix/code/analyze_fixed_window_sensitivity.py` — validate the compact daily replays, reprice 0.005% paths to 0.01%, compute per-window metrics and Controller-relative summaries.
- `CMTFlow_AAAI27/appendix/code/plot_fixed_window_sensitivity.py` — render the historical dense wealth-path plus Controller-percentile figure for each market.
- `CMTFlow_AAAI27/appendix/inputs/fixed_window/fixed_window_metrics.csv` — 120-row evaluator audit at the original 0.005% cost.
- `CMTFlow_AAAI27/appendix/inputs/fixed_window/daily_replay_nasdaq100.csv` — date plus 60 pairs of net-growth/turnover columns.
- `CMTFlow_AAAI27/appendix/inputs/fixed_window/daily_replay_csi300.csv` — date plus 60 pairs of net-growth/turnover columns.
- `CMTFlow_AAAI27/appendix/expected/tables/fixed_window_sensitivity.csv` — 120 recomputed rows at the paper's 0.01% evaluation cost.
- `CMTFlow_AAAI27/appendix/expected/tables/fixed_window_summary.csv` — two-market Controller win-count and best/median summary.
- `CMTFlow_AAAI27/appendix/expected/tables/fixed_window_wealth_nasdaq100.csv` — plotting matrix reconstructed at 0.01%.
- `CMTFlow_AAAI27/appendix/expected/tables/fixed_window_wealth_csi300.csv` — plotting matrix reconstructed at 0.01%.
- `CMTFlow_AAAI27/appendix/expected/figures/fixed_window_sensitivity_nasdaq100.{pdf,png}` — NASDAQ-100 figure.
- `CMTFlow_AAAI27/appendix/expected/figures/fixed_window_sensitivity_csi300.{pdf,png}` — CSI-300 figure.
- `CMTFlow_AAAI27/appendix/tests/test_fixed_window_sensitivity.py` — isolated fee-replay and metric tests.

Modify:

- `CMTFlow_AAAI27/appendix/tests/test_appendix_package.py` — assert B.2 inventory, 1–60 coverage, deterministic one-command outputs, and disclosure boundaries.
- `CMTFlow_AAAI27/appendix/code/run_appendix.py` — generate B.2 tables and figures.
- `CMTFlow_AAAI27/appendix/README.md` — document B.2, its evaluation-only command path, and public inputs.
- `CMTFlow_AAAI27/appendix/CLAIM_BOUNDARIES.md` — state the high-percentile/adaptive-timing claim without claiming universal dominance.
- `CMTFlow_AAAI27/EXPECTED_RESULTS.md` and `CMTFlow_AAAI27/PACKAGE_STATUS.md` — list the added paper-facing assets.
- `CMTFlow_AAAI27/src/Components/PPO_model.py` — remove only confirmed unused implementation.
- `CMTFlow_AAAI27/MANIFEST.json` — regenerate hashes after every public file is final.

The author-side 120 action traces remain under a temporary runtime and are not
copied into `CMTFlow_AAAI27/`.

### Task 1: Lock B.2 behavior with failing tests

**Files:**

- Create: `CMTFlow_AAAI27/appendix/tests/test_fixed_window_sensitivity.py`
- Modify: `CMTFlow_AAAI27/appendix/tests/test_appendix_package.py`

- [ ] **Step 1: Write a synthetic fee-replay test**

Import `reprice_growth` and `path_metrics` from the new analyzer and assert the
declared formula:

```python
def test_reprice_growth_recovers_gross_path_before_new_fee():
    original_growth = 1.01
    turnover = 0.4
    original_fee = 0.00005
    new_fee = 0.00010
    expected = original_growth / (1.0 - turnover * original_fee)
    expected *= 1.0 - turnover * new_fee
    assert reprice_growth(
        original_growth, turnover, original_fee, new_fee
    ) == pytest.approx(expected)
```

Add validation cases for nonpositive growth, negative turnover, and a charged
fee greater than or equal to one.

- [ ] **Step 2: Add package-level B.2 assertions**

Extend `test_required_public_files_are_present` with both scripts, three public
inputs, two expected tables, two wealth matrices, and four figure files. Add a
test that reads `fixed_window_sensitivity.csv` and asserts:

```python
self.assertEqual(len(rows), 120)
for market in ("NASDAQ-100", "CSI-300"):
    windows = sorted(
        int(row["fixed_window_days"])
        for row in rows
        if row["market"] == market
    )
    self.assertEqual(windows, list(range(1, 61)))
    self.assertTrue(all(float(row["evaluation_cost_pct"]) == 0.01 for row in rows))
```

Extend the one-command test with all B.2 outputs and compare their bytes with
the locked files under `appendix/expected/`.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python -m pytest \
  CMTFlow_AAAI27/appendix/tests/test_fixed_window_sensitivity.py \
  CMTFlow_AAAI27/appendix/tests/test_appendix_package.py -q
```

Expected: failures for the missing analyzer, inputs, figures, and tables; all
pre-existing Appendix assertions remain green.

### Task 2: Run and compact the 120 evaluation-only paths

**Files:**

- Create: the three files under `CMTFlow_AAAI27/appendix/inputs/fixed_window/`
- Read only: `paper_experiments/run_paper_experiments_final.py`
- Read only: the two paper-selected checkpoints and their command records

- [ ] **Step 1: Prepare a temporary results root**

Create a `mktemp -d` directory. Under its `results/`, add read-only symlinks
named exactly as the evaluator expects, pointing to:

```text
CMTFlow_AAAI27/checkpoints/nasdaq100/checkpoints/best_model.pth
CMTFlow_AAAI27/checkpoints/nasdaq100/five_stage_training_command.json
CMTFlow_AAAI27/checkpoints/csi300/checkpoints/best_model.pth
CMTFlow_AAAI27/checkpoints/csi300/five_stage_training_command.json
```

The temporary symlink names may contain selected-run identifiers; public
package paths may not.

- [ ] **Step 2: Evaluate windows 1 through 60**

Use `/home/tongwenxuan/conda/envs/xuangu/bin/python`, CUDA, and the existing
`ensure_fixed_window_eval` entry point. Construct an `argparse.Namespace` with:

```python
fixed_windows=list(range(1, 61))
markets=["nas", "sh"]
seeds=["nas:49", "sh:90"]
device="cuda"
test_max_days=None
force_fixed_eval=False
skip_fixed_eval=False
```

Call `ensure_dirs(temp_output)`, then
`ensure_fixed_window_eval(args, ["nas", "sh"], {"nas": 49, "sh": 90}, dirs)`.
Expected: 120 `status=ok` rows and, for each row, one portfolio CSV and one
action CSV in the temporary cache.

- [ ] **Step 3: Apply anchor and structural checks before publishing**

Assert:

```python
assert len(metrics) == 120
assert set(metrics.fixed_window_days) == set(range(1, 61))
assert abs(nas_window_5.total_return * 100 - 219.8407) < 0.02
assert abs(nas_window_8.total_return * 100 - 336.9108) < 0.02
assert abs(csi_window_50.total_return * 100 - 292.2515) < 0.12
```

If any assertion fails, stop without modifying expected figures. Retain the
historical figure and report the failed anchor.

- [ ] **Step 4: Compact the traces**

Write `fixed_window_metrics.csv` with the provenance-free public columns:

```text
market,market_key,fixed_window_days,training_cost_pct,total_return_pct,
sharpe,max_drawdown_pct,calmar,switch_count,turnover_sum
```

For each market, align all action traces by date and write one wide daily replay:

```text
date,net_growth_w01,turnover_w01,...,net_growth_w60,turnover_w60
```

Derive `net_growth_wNN = exp(exec_log_return)`. Reject duplicate dates,
different date grids, missing/nonfinite values, nonpositive growth, or negative
turnover. Do not copy weights, action probabilities, checkpoint identifiers, or
seed-labelled paths.

### Task 3: Implement 0.01% analysis and summary generation

**Files:**

- Create: `CMTFlow_AAAI27/appendix/code/analyze_fixed_window_sensitivity.py`
- Test: `CMTFlow_AAAI27/appendix/tests/test_fixed_window_sensitivity.py`

- [ ] **Step 1: Implement guarded fee replay**

Add:

```python
def reprice_growth(
    original_growth: float,
    turnover: float,
    original_fee: float = 0.00005,
    evaluation_fee: float = 0.00010,
) -> float:
    charged = turnover * original_fee
    replacement = turnover * evaluation_fee
    if original_growth <= 0 or turnover < 0:
        raise ValueError("growth must be positive and turnover non-negative")
    if charged >= 1 or replacement >= 1:
        raise ValueError("turnover-adjusted fee must be below one")
    return original_growth / (1.0 - charged) * (1.0 - replacement)
```

Reuse the paper definitions of TR, SR, MDD, and CR from
`analyze_transaction_cost.py`.

- [ ] **Step 2: Validate and analyze both replay matrices**

Implement:

```python
def build_fixed_window_outputs(
    input_dir: Path,
    controller_replay_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    ...
```

For every market and window, reprice daily growth to 0.01%, accumulate wealth,
and write:

```text
fixed_window_sensitivity.csv
fixed_window_wealth_nasdaq100.csv
fixed_window_wealth_csi300.csv
```

Obtain the complete Controller wealth and metrics from the existing
`traces/transaction_cost/tables/{nas,sh}_daily_replay.csv` 0.01% columns.

- [ ] **Step 3: Compute the two-market summary**

For TR, SR, and CR, a Controller win means the fixed-window value is lower. For
MDD, a Controller win means the fixed-window value is higher. Write one row per
market with Controller value, best fixed value/window, median fixed value,
win count, and win percentage for each metric. Ties are not counted as wins.

- [ ] **Step 4: Run unit tests and confirm GREEN**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python -m pytest \
  CMTFlow_AAAI27/appendix/tests/test_fixed_window_sensitivity.py -q
```

Expected: all synthetic replay and validation tests pass.

### Task 4: Recreate the historical figures and one-command entry point

**Files:**

- Create: `CMTFlow_AAAI27/appendix/code/plot_fixed_window_sensitivity.py`
- Modify: `CMTFlow_AAAI27/appendix/code/run_appendix.py`
- Create: B.2 expected tables and figures

- [ ] **Step 1: Implement the market figure renderer**

Read the generated wealth matrix and summary CSV. For each market:

```python
for window in range(1, 61):
    ax.plot(date, wealth[f"fixed_w{window:02d}"],
            color=purple(window), alpha=0.30, linewidth=0.76)
ax.plot(date, wealth["fixed_w30"], color="black",
        linestyle="--", linewidth=1.55, label="Fixed HRL (30d)")
ax.plot(date, wealth["controller"], color="#B63A4A",
        linewidth=2.85, label="Learned Controller")
```

The right panel contains four horizontal bars for Controller win percentages,
with a dashed 50% reference line. Use semantic output stems:
`fixed_window_sensitivity_nasdaq100` and
`fixed_window_sensitivity_csi300`.

- [ ] **Step 2: Extend `run_appendix.py`**

Call `build_fixed_window_outputs(...)` after B.1, then call the B.2 renderer.
The command must generate tables and figures only from packaged inputs; it must
not import PyTorch or load a model checkpoint.

- [ ] **Step 3: Generate locked expected outputs**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python \
  CMTFlow_AAAI27/appendix/code/run_appendix.py \
  --output-dir CMTFlow_AAAI27/appendix/expected
```

Expected: the existing B.1/C.1/C.2/C.3 files remain present and B.2 adds two
CSV result tables, two wealth matrices, and four figure files.

- [ ] **Step 4: Visually inspect both PNG files**

Confirm both figures have:

- 60 thin fixed-window paths;
- a readable 30-day dashed reference;
- a red Controller curve;
- four win-percentage bars;
- correct market title;
- no seed in the title or filename;
- no clipping or blank panels.

If the complete rerun is materially inconsistent with the representative
anchors or the figure structure differs from the historical design, do not
replace an existing public expected figure.

### Task 5: Remove only confirmed model dead code

**Files:**

- Modify: `CMTFlow_AAAI27/src/Components/PPO_model.py`
- Modify: `CMTFlow_AAAI27/appendix/tests/test_appendix_package.py`

- [ ] **Step 1: Add dead-code boundary assertions**

Assert that `PPO_model.py` no longer defines `CausalConv1dBlock`, no longer
stores `InnerAC.last_temporal_attn1/2`, and still contains the required
`pred_head`, Controller return/risk/switch-advantage heads, and
`fallback_projection`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python -m pytest \
  CMTFlow_AAAI27/appendix/tests/test_appendix_package.py \
  -k model_dead_code -q
```

Expected: failure because the unused class and diagnostic assignments remain.

- [ ] **Step 3: Apply the minimum cleanup**

Delete `CausalConv1dBlock`, the unused `tcn_kernel_size` and stored
`max_boundary` members in `InnerAC`, and the unused temporal-attention member
assignments. Remove unused internal `MonitorAC`/`FullModel` constructor
parameters only when all packaged call sites can be changed without altering
locked command parsing. Keep auxiliary prediction heads, checkpoint-dependent
branches, and CLI compatibility arguments.

- [ ] **Step 4: Verify checkpoint compatibility**

Load both final checkpoint `agent_net` state dictionaries into their packaged
model instances with the same compatibility mode used by the evaluator, then
rerun the three representative fixed-window anchors or compare exact state-key
loading diagnostics. Expected: no new missing/unexpected canonical keys and
unchanged anchor metrics.

### Task 6: Documentation, manifest, and full verification

**Files:**

- Modify: `CMTFlow_AAAI27/appendix/README.md`
- Modify: `CMTFlow_AAAI27/appendix/CLAIM_BOUNDARIES.md`
- Modify: `CMTFlow_AAAI27/EXPECTED_RESULTS.md`
- Modify: `CMTFlow_AAAI27/PACKAGE_STATUS.md`
- Modify: `CMTFlow_AAAI27/MANIFEST.json`

- [ ] **Step 1: Document B.2 and its boundary**

Replace the statement that B.2 is absent. Explain that it evaluates 60 fixed
holding windows per market, leaves Manager/Trader active, changes only the
Controller schedule, and mechanically reprices the evaluation paths to 0.01%.
State that the result supports high-percentile adaptive timing, not universal
dominance over every ex-post-selected window.

- [ ] **Step 2: Run all Appendix tests**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python -m pytest \
  CMTFlow_AAAI27/appendix/tests -q
```

Expected: all tests pass.

- [ ] **Step 3: Regenerate the root manifest**

Use the existing manifest-builder path from
`tests/test_build_aaai27_repro_package.py`, or run the package builder's
manifest-only helper. Every non-ignored public file except `MANIFEST.json`
must appear exactly once with its current SHA-256 and size.

- [ ] **Step 4: Run package and source audits**

Run:

```bash
/tmp/cmtflow-appendix-venv/bin/python CMTFlow_AAAI27/scripts/verify_package.py
find CMTFlow_AAAI27 -type f -name '*.pth' -printf '%P\n' | sort
find CMTFlow_AAAI27 -type f -printf '%f\n' |
  grep -E -i '(seed[_-]?[0-9]+|nas49|sh90)' || true
```

Expected: the verifier reports success; model inventory contains exactly the
two semantic `best_model.pth` paths; the public filename scan is empty.

- [ ] **Step 5: Reproduce into a clean temporary output**

Run `run_appendix.py` into a new temporary directory and compare all generated
tables and figures byte-for-byte with `appendix/expected/`. Expected: identical
outputs with no checkpoint load and no raw market-data dependency.

- [ ] **Step 6: Inspect final public inventory**

Confirm that no action-level fixed-window traces, temporary logs,
`__pycache__`, exploratory figures, extra models, or seed-labelled filenames
were added. Report the final B.2 headline statistics and direct package path to
the user.

Git commits are intentionally omitted because the workspace already contains
unrelated uncommitted user changes and the user did not request repository
history changes.

## Verification

The task is complete only when:

1. all 120 fixed-window paths pass the anchor and date-grid checks;
2. 0.005% replay reconstructs evaluator metrics within numerical tolerance;
3. 0.01% tables are recomputed from public growth/turnover inputs;
4. both figures match the historical two-panel structure on visual inspection;
5. existing Appendix outputs remain reproducible;
6. both final checkpoints still load after cleanup;
7. the package manifest, model inventory, and filename disclosure audits pass.

## Next skill

Use `$superpower-executing-plans` for inline execution in this session. The
developer constraint for this run disallows subagent delegation.
