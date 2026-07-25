# Paper Figure Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Redraw the cumulative portfolio-value, representative Controller, and Inner-Actor refinement figures with larger fonts and less redundant text, then place the regenerated assets back into the full-evidence LaTeX paper.

**Architecture:** Keep the existing experiment traces, selected switch cases, selected Inner windows, and stable LaTeX asset names. Add focused Matplotlib-object tests first, change only the three plotting functions that own these figures, regenerate from archived outputs, copy only the selected assets into `paper_full_evidence_edit`, and compile the manuscript.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib, unittest/pytest, LaTeX (`pdflatex` and `bibtex`).

---

## File Map

- Create `tests/test_paper_figure_readability.py`: focused structural tests for titles, axes, labels, panel counts, and abbreviated text.
- Modify `paper_experiments/run_paper_experiments_final.py:320-345`: cumulative portfolio-value plot.
- Modify `paper_experiments/run_paper_experiments_final.py:931-1124`: two-row Controller case plot. This file already has unrelated user edits after line 1820; preserve them and inspect/stage only the intended hunks.
- Modify `paper_experiments/plot_inner_actor_base_adjustment.py:232-350`: four-row simplified Inner-Actor plot.
- Modify `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex:867-878,998-1010,1031-1043`: concise captions using the existing asset paths and labels.
- Regenerate selected files under `paper_experiments_outputs/paper_experiments_final/01_main_experiment/`, `03_controller_interpretability/`, and `04_inner_actor_interpretability/`.
- Replace six stable manuscript assets under `paper_full_evidence_edit/figures/` and `paper_full_evidence_edit/figures/explainability/`.

### Task 1: Add Figure-Structure Regression Tests

**Files:**
- Create: `tests/test_paper_figure_readability.py`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Write the focused failing tests**

Create the following test module. It captures figures through patched `save_figure` calls so the assertions check Matplotlib objects without writing production images.

```python
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_experiments import plot_inner_actor_base_adjustment as inner_plot
from paper_experiments import run_paper_experiments_final as final_plot


class PaperFigureReadabilityTests(unittest.TestCase):
    @patch.object(final_plot, "save_figure")
    @patch.object(final_plot, "read_curve")
    def test_main_equity_identifies_normalized_portfolio_value(self, read_curve, save_figure):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        read_curve.return_value = pd.DataFrame(
            {"date": dates, "wealth": [1.0, 1.1, 1.05, 1.2, 1.3]}
        )
        manifest = pd.DataFrame(
            [{"market": "nas", "curve_status": "available", "method": "buy_hold", "curve_path": "dummy.csv"}]
        )

        final_plot.plot_main_equity(manifest, Path("unused"), "nas", 49, Path("unused"))

        fig = save_figure.call_args.args[0]
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Nasdaq-100 Portfolio Value")
        self.assertEqual(ax.get_ylabel(), "Portfolio value (initial = 1.0)")
        self.assertGreaterEqual(ax.yaxis.label.get_fontsize(), 12)
        self.assertTrue(all(text.get_fontsize() >= 10 for text in ax.get_legend().get_texts()))
        plt.close(fig)

    @patch.object(final_plot, "save_figure")
    def test_controller_case_has_two_rows_and_trading_day_axis(self, save_figure):
        hold = np.linspace(1.0, 0.96, 31)
        switch = np.linspace(1.0, 1.04, 31)
        case = pd.Series(
            {
                "step": 20,
                "date": "2021-04-19",
                "exit_prob": 0.63,
                "hold_curve_30": json.dumps(hold.tolist()),
                "switch_curve_30": json.dumps(switch.tolist()),
            }
        )
        actions = pd.DataFrame(
            {"step": [20], "is_switch": [1], "is_free_switch": [1]}
        )

        final_plot.plot_controller_case(
            "nas", 1, case, pd.DataFrame({"step": [0, 40]}), actions, Path("unused")
        )

        fig = save_figure.call_args.args[0]
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(fig._suptitle.get_text(), "Nasdaq-100 switch on 2021-04-19 (p = 0.63)")
        self.assertEqual(fig.axes[0].get_title(), "A. Frozen portfolio return")
        self.assertEqual(fig.axes[1].get_title(), "B. Frozen portfolio drawdown")
        self.assertEqual(fig.axes[1].get_xlabel(), "Trading days after switch")
        all_text = " ".join(text.get_text() for ax in fig.axes for text in ax.texts)
        self.assertNotIn("Day-0 decision evidence", all_text)
        plt.close(fig)

    @patch.object(inner_plot, "save_figure")
    @patch.object(inner_plot, "select_window")
    @patch.object(inner_plot, "future_relative_return")
    @patch.object(inner_plot, "load_prices")
    @patch.object(inner_plot, "parse_matrix")
    def test_inner_case_keeps_four_rows_with_compact_titles(
        self, parse_matrix, load_prices, future_relative_return, select_window, save_figure
    ):
        dates = pd.bdate_range("2024-01-02", periods=35)
        assets = [f"A{i}" for i in range(6)]
        base = pd.DataFrame(1.0 / 6.0, index=dates, columns=assets)
        tilt = pd.DataFrame(
            np.linspace(-0.005, 0.005, len(dates) * len(assets)).reshape(len(dates), len(assets)),
            index=dates,
            columns=assets,
        )
        executed = base + tilt
        parse_matrix.side_effect = lambda _actions, column: {
            "base_weights_json": base,
            "exec_weights_json": executed,
            "inner_tilt_json": tilt,
        }[column]
        load_prices.return_value = base
        future_relative_return.return_value = tilt * 4.0
        select_window.return_value = {"start": 0, "end": 29, "assets": assets}

        inner_plot.plot_market("nas", pd.DataFrame(), Path("unused"), future_horizon=5)

        fig = save_figure.call_args.args[0]
        panels = fig.axes[:4]
        self.assertEqual(fig._suptitle.get_text(), "Nasdaq-100 Inner-Actor Refinement")
        self.assertEqual(
            [ax.get_title() for ax in panels],
            ["Future 5-day relative return", "Inner tilt", "Executed weights", "Tilt-return alignment"],
        )
        self.assertEqual(len(panels), 4)
        verbose_text = " ".join(ax.get_title() for ax in panels)
        self.assertNotIn("green =", verbose_text)
        self.assertNotIn("positive bars mean", verbose_text)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify the expected failures**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_paper_figure_readability.py -q
```

Expected: three failures showing the old `Wealth multiple` label, three-row Controller layout, and verbose Inner titles.

- [ ] **Step 3: Inspect the new test file without staging unrelated work**

```bash
git diff --check -- tests/test_paper_figure_readability.py
git diff -- tests/test_paper_figure_readability.py
```

Expected: only the new focused regression tests are present. Do not commit during
implementation unless the user separately requests a commit.

### Task 2: Clarify the Cumulative Portfolio-Value Plot

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py:320-345`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Replace the label and font-layout block in `plot_main_equity`**

Keep the existing data-loading and line-plot loops. Replace the title-through-save block with:

```python
    ax.set_title(
        f"{MARKET_LABELS[market]} Portfolio Value",
        fontsize=15.0,
        fontweight="semibold",
        pad=10,
    )
    ax.set_ylabel("Portfolio value (initial = 1.0)", fontsize=12.5)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=10.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    clean_axis(ax)
    ax.grid(True, axis="both", alpha=0.65)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
        fontsize=10.0,
        columnspacing=1.1,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.23)
    save_figure(fig, out_dir / f"main_equity_{market}")
```

- [ ] **Step 2: Run the cumulative-value test**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_paper_figure_readability.py::PaperFigureReadabilityTests::test_main_equity_identifies_normalized_portfolio_value -q
```

Expected: PASS.

- [ ] **Step 3: Inspect the scoped source diff**

```bash
git diff -- paper_experiments/run_paper_experiments_final.py | sed -n '1,180p'
```

Expected: the first new hunk changes only `plot_main_equity`; the pre-existing Inner cost-benefit changes later in the file remain untouched.

### Task 3: Reduce Controller Cases to Two Readable Rows

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py:931-1124`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Preserve data preparation and replace the three-row plotting block**

Keep all calculations through `switches = ...`, because the return dictionary still records the signal window. Replace the code from `fig = plt.figure(...)` through `save_figure(...)` with:

```python
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.42},
    )
    fig.suptitle(
        f"{MARKET_LABELS[market]} switch on {key_date_label.replace('/', '-')} (p = {exit_prob:.2f})",
        fontsize=15.0,
        fontweight="semibold",
        y=0.975,
        color="#1F2937",
    )

    ax0.plot(days, hold_ret_path, color=keep_color, lw=2.5, label="Hold")
    ax0.plot(days, switch_ret_path, color=switch_color, lw=2.8, label="Switch")
    ax0.fill_between(
        days,
        hold_ret_path,
        switch_ret_path,
        where=switch_ret_path >= hold_ret_path,
        color="#BFE6DD",
        alpha=0.58,
        interpolate=True,
    )
    ax0.fill_between(
        days,
        hold_ret_path,
        switch_ret_path,
        where=switch_ret_path < hold_ret_path,
        color="#F4C7BE",
        alpha=0.32,
        interpolate=True,
    )
    ax0.axhline(0, color="#CBD2DD", lw=1.0)
    ax0.scatter([realized_horizon], [hold_ret_path[-1]], color=keep_color, s=44, zorder=4, edgecolor="white", linewidth=0.8)
    ax0.scatter([realized_horizon], [switch_ret_path[-1]], color=switch_color, s=44, zorder=4, edgecolor="white", linewidth=0.8)
    hold_offset = -18 if hold_ret_path[-1] <= switch_ret_path[-1] else 9
    switch_offset = 9 if hold_ret_path[-1] <= switch_ret_path[-1] else -18
    ax0.annotate(
        f"Hold {hold_return * 100:+.2f}%",
        xy=(realized_horizon, hold_ret_path[-1]),
        xytext=(-8, hold_offset),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color=keep_color,
    )
    ax0.annotate(
        f"Switch {switch_return * 100:+.2f}%",
        xy=(realized_horizon, switch_ret_path[-1]),
        xytext=(-8, switch_offset),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        fontweight="semibold",
        color=switch_color,
    )
    ax0.text(
        0.02,
        0.06,
        f"Return gap {ret_gain * 100:+.2f} pp",
        transform=ax0.transAxes,
        fontsize=9.7,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D9DEE7", "alpha": 0.94},
    )
    ax0.set_title("A. Frozen portfolio return", loc="left", fontsize=11.5, pad=6)
    ax0.set_ylabel("Return (%)", fontsize=10.5)
    ax0.set_xlim(0, realized_horizon)
    ax0.legend(loc="upper left", ncol=2, frameon=False, fontsize=9.5)

    ax1.plot(days, hold_dd_path, color=keep_color, lw=2.3, label="Hold")
    ax1.plot(days, switch_dd_path, color=switch_color, lw=2.6, label="Switch")
    ax1.fill_between(
        days,
        switch_dd_path,
        hold_dd_path,
        where=hold_dd_path >= switch_dd_path,
        color="#F1B7AB",
        alpha=0.40,
        interpolate=True,
    )
    hold_mdd_day = int(np.nanargmax(hold_dd_path)) if len(hold_dd_path) else realized_horizon
    switch_mdd_day = int(np.nanargmax(switch_dd_path)) if len(switch_dd_path) else realized_horizon
    ax1.scatter([hold_mdd_day], [hold_dd_path[hold_mdd_day]], color=keep_color, s=42, zorder=4, edgecolor="white", linewidth=0.8)
    ax1.scatter([switch_mdd_day], [switch_dd_path[switch_mdd_day]], color=switch_color, s=42, zorder=4, edgecolor="white", linewidth=0.8)
    ax1.text(
        0.02,
        0.88,
        f"MDD reduction {mdd_gain * 100:+.2f} pp",
        transform=ax1.transAxes,
        va="top",
        fontsize=9.7,
        color="#1F2937",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D9DEE7", "alpha": 0.94},
    )
    ax1.set_title("B. Frozen portfolio drawdown", loc="left", fontsize=11.5, pad=6)
    ax1.set_ylabel("Drawdown (%)", fontsize=10.5)
    ax1.set_xlabel("Trading days after switch", fontsize=10.5)
    ax1.set_ylim(bottom=0)

    for ax in (ax0, ax1):
        clean_axis(ax)
        ax.grid(True, axis="both", alpha=0.60)
        ax.tick_params(axis="both", labelsize=9.5)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.12)
    save_figure(fig, out_dir / f"controller_case_{market}_{case_id:02d}")
```

- [ ] **Step 2: Remove the now-unused `FancyBboxPatch` import only if no other function uses it**

Run:

```bash
rg -n 'FancyBboxPatch' paper_experiments/run_paper_experiments_final.py
```

If the only remaining match is the import, delete that import with `apply_patch`; otherwise retain it.

- [ ] **Step 3: Run the Controller structure test**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_paper_figure_readability.py::PaperFigureReadabilityTests::test_controller_case_has_two_rows_and_trading_day_axis -q
```

Expected: PASS with two axes and no C-row content.

- [ ] **Step 4: Verify the existing result fields remain unchanged**

```bash
rg -n '"case_horizon"|"ret_gain_30"|"mdd_gain_30"|"exit_prob"|"score"' paper_experiments/run_paper_experiments_final.py | sed -n '1,40p'
```

Expected: all fields remain in the return dictionary after `save_figure`.

### Task 4: Simplify the Four-Row Inner-Actor Figure

**Files:**
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py:232-350`
- Test: `tests/test_paper_figure_readability.py`

- [ ] **Step 1: Tighten the figure-level layout and title**

Replace the figure construction and suptitle with:

```python
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.5, 8.6),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72], "hspace": 0.30},
    )
    fig.suptitle(
        f"{MARKET_LABELS[market]} Inner-Actor Refinement",
        fontsize=18,
        fontweight="semibold",
        y=0.985,
    )
```

- [ ] **Step 2: Replace verbose panel and colorbar text**

Use these exact labels:

```python
    panel_titles = [
        "Future 5-day relative return",
        "Inner tilt",
        "Executed weights",
        "Tilt-return alignment",
    ]
```

Set the first three titles with `fontsize=12.5, loc="left", pad=5`, set their asset tick labels to `fontsize=10.5`, and replace the three colorbar labels with:

```python
    c0.set_label("Return (%)", fontsize=10.5)
    c1.set_label("Tilt (pp)", fontsize=10.5)
    c2.set_label("Weight (%)", fontsize=10.5)
```

For all three colorbars, call `cbar.ax.tick_params(labelsize=9.5)`.

- [ ] **Step 3: Compact the fourth-row evidence labels**

Keep the bars and colors. Replace each bar label with:

```python
            f"{value:+.1f} | {hit:.0%}",
```

using `fontsize=9.8`. Replace the title, x-axis label, and summary annotation with:

```python
    axes[3].set_title("Tilt-return alignment", fontsize=12.5, loc="left", pad=5)
    axes[3].set_xlabel("Cumulative alignment score (bp proxy)", fontsize=11.0)
    axes[3].tick_params(axis="both", labelsize=10.0)
    axes[3].text(
        0.012,
        0.96,
        f"Mean r = {corr:.2f}  |  Positive days = {positive_ratio:.0%}",
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=10.2,
        color="#293241",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D8DEE9", "alpha": 0.94},
    )
```

Increase the fourth-row x-limits from `1.25` to `1.34` times the maximum absolute score to protect the rightmost label, and finish with:

```python
    fig.subplots_adjust(left=0.09, right=0.94, top=0.92, bottom=0.08)
```

- [ ] **Step 4: Run the Inner structure test**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_paper_figure_readability.py::PaperFigureReadabilityTests::test_inner_case_keeps_four_rows_with_compact_titles -q
```

Expected: PASS with four compact panel titles.

- [ ] **Step 5: Inspect the scoped Inner source diff**

```bash
git diff --check -- paper_experiments/plot_inner_actor_base_adjustment.py
git diff -- paper_experiments/plot_inner_actor_base_adjustment.py
```

Expected: changes are limited to the approved four-row presentation and labels.

### Task 5: Regenerate and Place the Selected Figure Assets

**Files:**
- Regenerate: `paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_{nas,sh}.{png,pdf}`
- Regenerate: `paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_{nas,sh}_{01,02}.{png,pdf}`
- Regenerate: `paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_{nas,sh}.{png,pdf}`
- Replace: `paper_full_evidence_edit/figures/main_equity_{nas,sh}.pdf`
- Replace: `paper_full_evidence_edit/figures/explainability/controller_switch_case_{nas,sh}.png`
- Replace: `paper_full_evidence_edit/figures/explainability/inner_actor_{nas,sh}.png`

- [ ] **Step 1: Run the existing final-paper generator from archived traces**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python paper_experiments/run_paper_experiments_final.py \
  --markets nas sh \
  --seeds nas:49 sh:90 \
  --skip_fixed_eval
```

Expected: exit 0 and `wrote final paper experiments to .../paper_experiments_final`. This is evaluation/plotting only; it must not start model training.

- [ ] **Step 2: Regenerate the selected Inner refinement figures**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python paper_experiments/plot_inner_actor_base_adjustment.py \
  --markets nas sh \
  --seeds nas:49 sh:90
```

Expected: exit 0 and updated `inner_actor_base_adjustment_future_return_{nas,sh}.{png,pdf}`.

- [ ] **Step 3: Copy only the selected assets to stable manuscript names**

```bash
cp paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_nas.pdf \
  paper_full_evidence_edit/figures/main_equity_nas.pdf
cp paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_sh.pdf \
  paper_full_evidence_edit/figures/main_equity_sh.pdf
cp paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_nas_01.png \
  paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png
cp paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_sh_01.png \
  paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png
cp paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_nas.png \
  paper_full_evidence_edit/figures/explainability/inner_actor_nas.png
cp paper_experiments_outputs/paper_experiments_final/04_inner_actor_interpretability/inner_actor_base_adjustment_future_return_sh.png \
  paper_full_evidence_edit/figures/explainability/inner_actor_sh.png
```

- [ ] **Step 4: Confirm all expected generated and manuscript assets exist**

```bash
test -f paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_nas.pdf
test -f paper_experiments_outputs/paper_experiments_final/01_main_experiment/main_equity_sh.pdf
test -f paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_nas_01.png
test -f paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_nas_02.png
test -f paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_sh_01.png
test -f paper_experiments_outputs/paper_experiments_final/03_controller_interpretability/controller_case_sh_02.png
test -f paper_full_evidence_edit/figures/main_equity_nas.pdf
test -f paper_full_evidence_edit/figures/main_equity_sh.pdf
test -f paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png
test -f paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png
test -f paper_full_evidence_edit/figures/explainability/inner_actor_nas.png
test -f paper_full_evidence_edit/figures/explainability/inner_actor_sh.png
```

Expected: exit 0 with no missing files.

### Task 6: Update Concise LaTeX Captions

**Files:**
- Modify: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex:875-877`
- Modify: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex:1006-1009`
- Modify: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex:1039-1042`

- [ ] **Step 1: Replace only the three figure captions**

Use:

```tex
\caption{Cumulative portfolio value, normalized to 1 at the beginning
of each test period. Left: Nasdaq-100. Right: CSI-300.}
```

```tex
\caption{Representative Controller switches. Titles report the actual
switch dates and probabilities; curves compare frozen 30-day hold and
switch paths. Left: Nasdaq-100. Right: CSI-300.}
```

```tex
\caption{Inner-Actor refinement cases. The four rows show future
relative returns, within-support tilts, executed weights, and
tilt--return alignment. Left: Nasdaq-100. Right: CSI-300.}
```

Keep the existing `\label` commands and `\includegraphics` paths unchanged.

- [ ] **Step 2: Verify the manuscript references stable filenames**

```bash
rg -n 'main_equity_(nas|sh)|controller_switch_case_(nas|sh)|inner_actor_(nas|sh)' \
  paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex
```

Expected: six existing image references with no new paths.

### Task 7: Full Verification and Scoped Integration

**Files:**
- Test: `tests/test_paper_figure_readability.py`
- Test: `tests/test_paper_experiments.py`
- Build: `paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`

- [ ] **Step 1: Run focused and existing paper-experiment tests**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_paper_figure_readability.py \
  tests/test_paper_experiments.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect the six final manuscript assets at original resolution**

Use `view_image` on the two Controller and two Inner PNGs. Render or inspect the two PDF pages/images for cumulative portfolio value. Confirm:

- no title, legend, endpoint, annotation, axis-label, or colorbar overlap;
- no C row in Controller images;
- dated Controller titles and a trading-day x-axis;
- four compact Inner rows;
- cumulative y-axis explicitly says initial value is 1.0.

- [ ] **Step 3: Compile the full-evidence manuscript**

Run from `paper_full_evidence_edit`:

```bash
pdflatex -interaction=nonstopmode anonymous-submission-latex-2026-full-evidence.tex
bibtex anonymous-submission-latex-2026-full-evidence
pdflatex -interaction=nonstopmode anonymous-submission-latex-2026-full-evidence.tex
pdflatex -interaction=nonstopmode anonymous-submission-latex-2026-full-evidence.tex
```

Expected: all four commands exit 0, the PDF is regenerated, and the log contains no missing figure errors.

- [ ] **Step 4: Check source and manuscript diffs without disturbing unrelated work**

```bash
git diff --check -- \
  paper_experiments/run_paper_experiments_final.py \
  paper_experiments/plot_inner_actor_base_adjustment.py \
  tests/test_paper_figure_readability.py \
  paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex
git diff -- paper_experiments/run_paper_experiments_final.py \
  paper_experiments/plot_inner_actor_base_adjustment.py \
  tests/test_paper_figure_readability.py \
  paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex
git status --short
```

Expected: no whitespace errors; intended figure hunks are visible; pre-existing unrelated changes remain present and unstaged.

- [ ] **Step 5: Leave the verified changes unstaged for user review**

Do not stage or commit by default. `run_paper_experiments_final.py` already contains
unrelated user work, so the handoff must list the intended source hunks, captions,
tests, and six replaced manuscript assets explicitly. Commit only if the user later
requests it and the intended hunks can be isolated safely.

## Verification Summary

- Goal: three figure groups are legible at final publication size with redundant prose removed.
- Key changes: normalized portfolio-value axis label, larger fonts, two-row Controller cases, compact four-row Inner cases, concise captions, regenerated stable assets.
- Required evidence: focused tests pass, final images inspected, LaTeX builds, and scoped diff contains no unrelated edits.
- Next skill: `$superpower-subagents` if the user explicitly requests delegated execution; otherwise `$superpower-executing-plans` for inline implementation.

## Approved 2026-07-14 Local-Asset Revision

This revision supersedes the earlier source-font values but keeps the original
per-market assets and LaTeX paths. It is executed entirely in the local
workspace: do not stage, commit, push, create a branch, or switch worktrees.

### Task A: Lock typography and geometry with failing tests

**Files:**
- Modify: `tests/test_paper_figure_readability.py`

- [ ] Require Main Equity title/y-label/ticks/legend to be at least
  `30/25/21/20` pt while retaining the `12.5 x 6.2 in` canvas.
- [ ] Require Controller title/panel/axis/tick/legend/annotation fonts to be at
  least twice their current values and require each axes rectangle to occupy
  less than 32% of the canvas height.
- [ ] Require Inner title/panel/asset/colorbar/axis/annotation fonts to be at
  least twice their current values, preserve the `11.5 in` width, and require
  the height to exceed `8.6 in`.
- [ ] Run the three named unit tests and verify they fail on the old font values.

### Task B: Enlarge Main Equity and Controller typography

**Files:**
- Modify: `paper_experiments/run_paper_experiments_final.py`

- [ ] In `plot_main_equity`, use 30pt title, 25pt y-label, 21pt ticks, and 20pt
  legend. Wrap the full legend to three columns and reserve bottom margin; keep
  the canvas and every curve unchanged.
- [ ] In `plot_controller_case`, use 30pt figure title, 23pt panel titles, 21pt
  axes, 19pt ticks/legend/endpoint labels, and 19.4pt metric annotations.
  Preserve the 7.2in width and reduce the axes rectangles with larger margins.
- [ ] Run the Main and Controller unit tests and verify both pass.

### Task C: Make Inner taller and double all typography

**Files:**
- Modify: `paper_experiments/plot_inner_actor_base_adjustment.py`

- [ ] Change only the height from `8.6` to `14.0 in`; keep width `11.5 in`.
- [ ] Use 36pt figure title, 25pt row titles, 21pt asset codes/colorbar labels,
  19pt colorbar ticks, 20pt bar-axis ticks, 22pt bar x-label, 19.6pt bar
  annotations, and 20.4pt summary text.
- [ ] Increase subplot spacing and margins only as needed to prevent clipping.
- [ ] Run the Inner unit test and verify it passes.

### Task D: Regenerate and validate six local assets

**Files:**
- Replace the six stable manuscript assets named in the approved design.

- [ ] Render Main and Controller from the existing manifest, traces, selected
  case CSVs, portfolio traces, and action traces; do not run evaluation or
  training.
- [ ] Run `plot_inner_actor_base_adjustment.py` without `--force_eval` so it
  reads the existing cached action traces.
- [ ] Copy only the two Main PDFs, two first Controller-case PNGs, and two Inner
  PNGs to their stable manuscript paths.
- [ ] Run the focused test file, verify dimensions/font contracts, and visually
  inspect all six assets for overlaps and clipping.

## Approved 2026-07-15 Shorter Case-Figure Revision

- [ ] Update the Controller regression test to require height `<= 7.0 in`, a
  34pt bold figure title, 25pt bold panel titles, and 22pt axes.
- [ ] Update the Inner regression test to require height `<= 10.8 in`, a 38pt
  bold figure title, 27pt bold row titles, and 22pt asset labels.
- [ ] Reduce Controller and Inner canvas heights and whitespace while retaining
  all current evidence rows, data, selected cases, and manuscript widths.
- [ ] Regenerate the four case assets from cached traces, replace the four stable
  manuscript PNGs, inspect them visually, and run the focused test file.
