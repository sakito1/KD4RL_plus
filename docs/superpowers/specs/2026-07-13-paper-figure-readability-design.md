# Paper Figure Readability Redesign

Date: 2026-07-13

## Objective

Improve the readability of the three figure groups used in
`paper_full_evidence_edit/anonymous-submission-latex-2026-full-evidence.tex`
without changing their underlying data, selected cases, metrics, colors, or
experimental meaning. The redesign removes explanatory prose that is already
provided by the paper and increases the visual hierarchy of the remaining
labels at the final two-column publication size.

## Scope

The change covers:

1. cumulative portfolio-value comparisons for Nasdaq-100 and CSI-300;
2. representative Controller switch cases for both markets;
3. Inner-Actor refinement cases for both markets;
4. regenerated figure assets and their placement in the full-evidence LaTeX
   manuscript.

It does not change experiment selection, traces, counterfactual horizons,
statistics, figure colors, table values, or model results.

## Design 1: Cumulative Portfolio Value

Source function:
`paper_experiments/run_paper_experiments_final.py::plot_main_equity`.

Requirements:

- Keep one market per image and the existing side-by-side LaTeX layout.
- Plot the same normalized cumulative curves and preserve the current method
  colors and relative line emphasis.
- Use the title `<Market> Portfolio Value`.
- Replace `Wealth multiple` with
  `Portfolio value (initial = 1.0)` on the vertical axis.
- Keep the horizontal axis date-based and avoid an unnecessary x-axis title.
- Increase title, axis-label, tick-label, and legend font sizes so they remain
  legible at `0.49\columnwidth`.
- Keep the legend outside the data area and ensure it does not collide with the
  date labels.
- Update the LaTeX caption to state explicitly that portfolio value is
  normalized to 1 at the beginning of each test period.

Outputs:

- `paper_full_evidence_edit/figures/main_equity_nas.pdf`
- `paper_full_evidence_edit/figures/main_equity_sh.pdf`

## Design 2: Representative Controller Switch Cases

Source function:
`paper_experiments/run_paper_experiments_final.py::plot_controller_case`.

Requirements:

- Preserve the selected switch dates and the fixed 30-trading-day frozen
  hold/switch paths.
- Reduce the figure from three rows to two rows; remove the current decision
  evidence card row entirely.
- Put the market, switch date, and switch probability directly in the title:
  `<Market> switch on YYYY-MM-DD (p = 0.xx)`.
- Remove the subtitle, bottom explanatory sentence, and other prose already
  provided by the manuscript.
- Use compact panel titles:
  - `A. Frozen portfolio return`
  - `B. Frozen portfolio drawdown`
- Label the horizontal axis `Trading days after switch`.
- Shorten the line labels to `Hold` and `Switch`. The shaded difference may
  remain visually present but does not need a separate verbose legend entry.
- Retain the essential numerical evidence: terminal hold/switch returns,
  return gap, and maximum-drawdown reduction.
- Place annotations using axis-relative or padded coordinates so labels remain
  inside the canvas and do not overlap curves, legends, titles, or each other.
- Increase the remaining title, panel-title, axis, tick, legend, and annotation
  fonts for display at `0.49\columnwidth`.
- Keep the LaTeX figure as two market panels and use a concise caption; the
  manuscript text remains responsible for explaining the counterfactual
  protocol.

Outputs:

- `paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png`
- `paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png`

## Design 3: Inner-Actor Refinement Cases

Source function:
`paper_experiments/plot_inner_actor_base_adjustment.py::plot_market`.

Requirements:

- Preserve all four evidence rows and the same selected windows/assets:
  future five-day relative return, Inner tilt, executed weights, and
  tilt-return alignment.
- Use the title `<Market> Inner-Actor Refinement`.
- Replace explanatory row titles with:
  - `Future 5-day relative return`
  - `Inner tilt`
  - `Executed weights`
  - `Tilt-return alignment`
- Remove parenthetical descriptions of color meanings and mechanism details.
- Shorten colorbar labels to units only: `Return (%)`, `Tilt (pp)`, and
  `Weight (%)`.
- Retain a compact one-line summary of the correlation and positive-alignment
  ratio in the fourth row because these are quantitative evidence rather than
  procedural explanation.
- Retain asset-level bar values and hit rates, but use compact formatting and
  sufficient right/left padding to prevent clipping.
- Increase titles, asset labels, ticks, colorbar labels, summary text, and bar
  annotations while reducing unused vertical whitespace.
- Keep the existing two-market side-by-side LaTeX layout and use a concise
  caption; detailed interpretation stays in the body text.

Outputs:

- `paper_full_evidence_edit/figures/explainability/inner_actor_nas.png`
- `paper_full_evidence_edit/figures/explainability/inner_actor_sh.png`

## Asset Flow and Manuscript Integration

The authoritative plotting scripts regenerate the figures under
`paper_experiments_outputs/paper_experiments_final/`. The selected final assets
are then copied to the stable filenames already referenced by the
full-evidence manuscript. Existing `\includegraphics` paths remain unchanged,
which avoids accidental changes to figure ordering or labels.

Only the three corresponding LaTeX captions may be revised for clarity. Figure
labels and surrounding result claims remain unchanged.

## Validation

1. Run focused plotting/test commands using the archived final traces; do not
   retrain models or change selected cases.
2. Confirm all ten expected output assets exist: two cumulative-value files,
   four Controller case files in the experiment output, and four final
   Controller/Inner manuscript images.
3. Inspect every final manuscript image at original resolution for overlaps,
   clipped annotations, unreadable labels, and excessive prose.
4. Confirm the cumulative-value y-axis contains
   `Portfolio value (initial = 1.0)`.
5. Confirm Controller figures contain two axes, no C row, a dated title, and a
   trading-day x-axis label.
6. Confirm Inner figures retain four rows with shortened titles.
7. Compile the full-evidence LaTeX manuscript and check that all figures resolve
   under their existing labels and fit without overfull figure content.

## Non-Goals and Safety

- Do not recompute or alter experiment metrics.
- Do not select new representative cases or Inner windows.
- Do not change unrelated user edits in the dirty worktree.
- Do not refactor unrelated plotting or experiment code.

## Approved Typography and Geometry Revision (2026-07-14)

The final per-market assets remain the authoritative targets for this revision.
Their data, selected cases, traces, colors, labels, and manuscript paths remain
unchanged.

### Controller switch cases

- Increase every visible source font to at least twice its current size,
  including the figure title, panel titles, axes, ticks, legend, endpoint
  labels, and metric annotations.
- Keep the current source width.
- Make the plotted curve regions smaller by increasing the space reserved for
  typography and by reducing the axes rectangles within the canvas.
- Preserve both evidence rows and all plotted paths and statistics.

### Inner-Actor refinement cases

- Keep the current `11.5 in` source width and increase only the figure height.
- Increase every visible source font to at least twice its current size,
  including asset codes, titles, ticks, colorbars, summary text, axis labels,
  and bar annotations.
- Use the additional vertical space to prevent overlap and clipping; do not
  widen the figure.
- Preserve the four evidence rows, selected assets and window, heatmap values,
  alignment bars, correlations, and hit rates.

### Cumulative portfolio-value plots

- Keep the existing per-market canvas geometry and curves.
- Increase every visible source font by at least two times, including titles,
  axis labels, tick labels, and legend labels.
- The legend may wrap to additional rows only when required to prevent text
  collisions; no curve, method, or label may be removed.

### Outputs and validation

Regenerate and replace exactly these six stable manuscript assets:

- `paper_full_evidence_edit/figures/main_equity_nas.pdf`
- `paper_full_evidence_edit/figures/main_equity_sh.pdf`
- `paper_full_evidence_edit/figures/explainability/controller_switch_case_nas.png`
- `paper_full_evidence_edit/figures/explainability/controller_switch_case_sh.png`
- `paper_full_evidence_edit/figures/explainability/inner_actor_nas.png`
- `paper_full_evidence_edit/figures/explainability/inner_actor_sh.png`

Validation must compare the revised Matplotlib font settings with the current
settings, confirm the Inner canvas width is unchanged and its height is larger,
and visually inspect all six final assets for clipping and overlap.
