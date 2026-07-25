# Paper Figure 9pt Capacity-Preserving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Replace each pair of tiny per-market paper images with one single-column composite whose source fonts are at least twice the current sizes and render at no less than 9pt in the manuscript.

**Architecture:** Add one focused composite-figure script rather than extending the already modified final-experiment driver. Separate archived-data loading from three renderers, enforce a shared 21pt minimum typography contract, keep the existing per-market artifacts, and change only the three LaTeX figure includes to the new `\columnwidth` composites.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib, unittest, existing archived experiment CSV/JSON traces, LaTeX.

---

## File Map

- Create `paper_experiments/plot_paper_figure_composites.py`: load archived final traces and render the three market-composite figures.
- Modify `tests/test_paper_figure_readability.py`: add synthetic-data structural and typography tests for all three composites.
- Modify `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`: replace three pairs of half-column images with three full-column composite images.
- Generate `paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_combined.{png,pdf}`.
- Generate `paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_switch_cases_combined.{png,pdf}`.
- Generate `paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/inner_actor_combined.{png,pdf}`.
- Copy the PDF main composite and PNG Controller/Inner composites to stable manuscript paths.

### Task 1: Specify the Composite Typography Contract

**Files:**
- Modify: `tests/test_paper_figure_readability.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Add a helper that enumerates every visible Matplotlib text object**

Add:

```python
def visible_font_sizes(fig):
    sizes = []
    for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text)):
        if text.get_visible() and text.get_text().strip():
            sizes.append(float(text.get_fontsize()))
    return sizes
```

- [ ] **Step 2: Add three failing renderer tests**

Import `paper_experiments.plot_paper_figure_composites as composites` and add:

```python
def assert_minimum_figure_font(self, fig):
    sizes = visible_font_sizes(fig)
    self.assertTrue(sizes)
    self.assertGreaterEqual(min(sizes), 21.0)
```

Use synthetic curves for `render_main_composite`, two synthetic Controller case dictionaries for `render_controller_composite`, and two six-asset Inner case dictionaries for `render_inner_composite`. Assert:

```python
self.assertEqual(len(main_fig.axes), 2)
self.assertEqual(len(controller_fig.axes), 4)
self.assertEqual(len(inner_fig.axes[:8]), 8)
self.assertEqual(main_fig.get_size_inches()[0], 7.0)
self.assertEqual(controller_fig.get_size_inches()[0], 7.0)
self.assertEqual(inner_fig.get_size_inches()[0], 7.0)
self.assertIn("Portfolio value (initial = 1.0)", [text.get_text() for text in main_fig.texts])
self.assertIn("Trading days after switch", [text.get_text() for text in controller_fig.texts])
self.assertIn("Inner-Actor Refinement", [text.get_text() for text in inner_fig.texts])
self.assert_minimum_figure_font(fig)
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_paper_figure_readability.py -v
```

Expected: import failure because `plot_paper_figure_composites.py` does not yet exist.

### Task 2: Implement Shared Typography and the Main Composite

**Files:**
- Create: `paper_experiments/plot_paper_figure_composites.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Define constants, output helpers, and the main renderer**

The new module must define:

```python
SOURCE_WIDTH = 7.0
FONT_MIN = 21.0
FONT_AXIS = 25.0
FONT_PANEL = 30.0
FONT_FIGURE = 36.0

def save_figure(fig, path_base):
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def render_main_composite(curves_by_market, path_base=None):
    ...
    return fig
```

`curves_by_market` maps `nas` and `sh` to lists of dictionaries with `date`, `wealth`, `label`, `color`, `linewidth`, and `zorder`. Render a 1-by-2 figure with a 7.0-inch source width, `Nasdaq-100`/`CSI-300` panel titles, one figure-level vertical label, reduced date ticks, and one shared legend. Use 30pt panel titles, 25pt shared y-label, and no text below 21pt. Keep the shared legend at four columns so long method names remain readable.

- [ ] **Step 2: Add the archived main-curve loader**

Define:

```python
def load_main_curves(manifest, end2end_dir, seeds):
    ...
```

Follow the existing `plot_main_equity` data path exactly: available matched baseline curves from the manifest plus `{market}_seed{seed}_full_controller_portfolio.csv`, preserving method labels, colors, and CMTFlow/`Ours` line emphasis.

- [ ] **Step 3: Run the main composite test and verify GREEN**

Run the main-specific unittest by name and confirm the minimum visible font is 21pt.

### Task 3: Implement the Controller 2-by-2 Composite

**Files:**
- Modify: `paper_experiments/plot_paper_figure_composites.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Normalize one archived selected case**

Define:

```python
def prepare_controller_case(case):
    ...
```

Parse `hold_curve_30` and `switch_curve_30`, compute frozen return paths, drawdown paths, terminal returns, return gap, maximum-drawdown reduction, date, and switch probability using the existing helper functions from `run_paper_experiments_final.py`.

- [ ] **Step 2: Render the four panels with shared text**

Define:

```python
def render_controller_composite(cases_by_market, path_base=None):
    ...
    return fig
```

Use a 2-by-2, 7.0-inch-wide figure. Markets are columns; return and drawdown are rows. Use 30pt market/date headings, 25pt row labels and shared x-label, 21pt ticks/legend/annotations, one shared `Hold`/`Switch` legend, protected annotation boxes, and no C row. Reduce x ticks to `0, 10, 20, 30`.

- [ ] **Step 3: Load the first selected case for each market**

Define:

```python
def load_controller_cases(output_root):
    ...
```

Read `03_controller_interpretability/selected_controller_cases_{market}.csv`, require at least one row, and prepare only row 1 because those are the manuscript cases already used.

- [ ] **Step 4: Run the Controller composite test and verify GREEN**

Run the named unittest and confirm four panel axes, shared x-label, and minimum 21pt text.

### Task 4: Implement the Inner 4-by-2 Composite

**Files:**
- Modify: `paper_experiments/plot_paper_figure_composites.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Prepare one Inner case without changing selection**

Define:

```python
def prepare_inner_case(market, actions, future_horizon=5):
    ...
```

Reuse `parse_matrix`, `load_prices`, `future_relative_return`, `select_window`, `mean_corr`, and `date_ticks` from `plot_inner_actor_base_adjustment.py`. Return the same selected assets/window, future-return matrix, tilt matrix, executed-weight matrix, asset alignment, hit rates, mean correlation, and positive-day ratio used by the current per-market figure.

- [ ] **Step 2: Render the eight panels**

Define:

```python
def render_inner_composite(cases_by_market, path_base=None):
    ...
    return fig
```

Use a 4-by-2, 7.0-inch-wide composite. Keep a 36pt figure title; use 30pt market headings, 25pt shared row titles, and 21pt asset labels/colorbar ticks/units/annotations. Use no date labels, at most three colorbar ticks, compact alignment strings, and symmetric alignment limits with sufficient padding. The first eight `fig.axes` entries must be the data panels; colorbars are appended afterward.

- [ ] **Step 3: Load cached Inner action traces**

Define:

```python
def load_inner_cases(output_root, seeds, future_horizon=5):
    ...
```

Read `_cache/inner_base_adjustment/{market}_seed{seed}_full_controller_inner_base_actions.csv`. If a cache file is absent, fail with a message instructing the caller to run `plot_inner_actor_base_adjustment.py` first; do not start evaluation or training implicitly.

- [ ] **Step 4: Run the Inner composite test and verify GREEN**

Run the named unittest and confirm eight data panels, retained four evidence rows, and minimum 21pt text.

### Task 5: Add a CLI and Generate the Composite Assets

**Files:**
- Modify: `paper_experiments/plot_paper_figure_composites.py`

- [ ] **Step 1: Add command-line arguments**

Support:

```text
--baseline_dir
--end2end_dir
--output_dir
--seeds nas:49 sh:90
--future_horizon 5
```

The CLI loads the existing manifest and archived traces, renders all three composites, and writes them under the existing experiment subdirectories.

- [ ] **Step 2: Run the complete focused test file**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_paper_figure_readability.py -v
```

Expected: all old readability tests and new composite tests pass.

- [ ] **Step 3: Generate the real composites**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python paper_experiments/plot_paper_figure_composites.py \
  --seeds nas:49 sh:90 \
  --future_horizon 5
```

Expected: six non-empty output files, PNG and PDF for each composite.

- [ ] **Step 4: Copy the selected formats into the manuscript**

Copy main PDF to `paper_full_evidence_edit/figures/main_equity_combined.pdf`, Controller PNG to `paper_full_evidence_edit/figures/explainability/controller_switch_cases_combined.png`, and Inner PNG to `paper_full_evidence_edit/figures/explainability/inner_actor_combined.png`.

### Task 6: Replace the Three LaTeX Image Pairs

**Files:**
- Modify: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`

- [ ] **Step 1: Replace each two-image block**

Use exactly one include per figure:

```tex
\includegraphics[width=\columnwidth]{figures/main_equity_combined.pdf}
```

```tex
\includegraphics[width=\columnwidth]{figures/explainability/controller_switch_cases_combined.png}
```

```tex
\includegraphics[width=\columnwidth]{figures/explainability/inner_actor_combined.png}
```

Keep the single-column `figure` environments and existing `\label` commands. Update caption panel-order wording only if needed.

- [ ] **Step 2: Verify no target figure uses `0.49\columnwidth`**

Run `rg` over the three labels and surrounding image blocks. Expected: one `\columnwidth` image per figure and no `figure*` conversion.

### Task 7: Visual and Mechanical Verification

**Files:**
- Test: `tests/test_paper_figure_readability.py`
- Build: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`

- [ ] **Step 1: Run fresh syntax and focused tests**

Run `py_compile` for the new script and the full readability unittest file. Expected: exit 0 and all tests pass.

- [ ] **Step 2: Inspect all three composites at original resolution**

Use `view_image` on the generated PNGs. Confirm no clipped labels, legends, colorbars, or annotations; verify the Controller has four panels and Inner has eight data panels.

- [ ] **Step 3: Inspect simulated manuscript-scale renders**

Resize copies to the pixel width corresponding to one manuscript column and inspect them at 100% scale. Confirm all remaining text is readable and the vertical growth is acceptable relative to the evidence density.

- [ ] **Step 4: Run existing paper tests**

Run `tests/test_paper_experiments.py`. Record the known pre-existing `_market_label` mismatch separately; do not change it as part of this task.

- [ ] **Step 5: Attempt the LaTeX build**

Run the four-pass `pdflatex`/`bibtex` sequence. If `newtxtext.sty` is still absent, record that environmental blocker while verifying the image references and assets independently.

- [ ] **Step 6: Check scoped diffs and leave them unstaged**

Run `git diff --check` only for the new script, tests, and manuscript. Do not stage or commit implementation changes because the shared worktree contains unrelated user edits.

## Verification Summary

- Contract: every visible source text object is at least 21pt and at least twice the old figure's corresponding size.
- Layout: single-column floats remain; market pairs are combined into one `\columnwidth` image.
- Evidence: curves, selected cases, Inner windows, statistics, and colors are unchanged.
- Safety: no retraining, case reselection, `figure*`, unrelated source edits, or implementation commits.
- Next skill: `$superpower-executing-plans` for inline implementation already authorized by the user.
