# Paper Figure 9pt Capacity-Preserving Redesign

Date: 2026-07-13

## Objective

Make every visible text element in the three paper figure groups at least twice
its current source size and no smaller than 9pt after LaTeX scaling, while
avoiding the two-column `figure*` layout and minimizing the loss of manuscript
text capacity.
The redesign must preserve all underlying traces, selected cases, statistics,
colors, and experimental conclusions.

## Constraint and Layout Rationale

The current manuscript places two independent source images at
`0.49\columnwidth` inside a single-column `figure`. Their source canvases are
7.2--12.5 inches wide, so the manuscript shrinks the 9.5--18pt source text to
far below 9pt. Merely multiplying the source fonts by two is therefore
insufficient.

The capacity-preserving solution is to generate one composite image per figure
group and include it once at `\columnwidth`. The two markets remain side by
side inside the composite, while repeated titles, legends, axis labels, and
colorbar information are shared wherever the evidence permits. A source canvas
near 7 inches wide and a minimum source font near 21--22pt yields at least 9pt
after single-column scaling. Every individual font value must also be at least
twice its current value.

The figures remain single-column floats. Some vertical growth is unavoidable,
especially for the eight-panel Inner figure, but shared elements and reduced
tick density should keep this growth substantially below a naive stacking of
the two market images.

## Typography Contract

- Main or figure-level titles: at least 30pt; the Inner title is at least 36pt
  because its current value is 18pt.
- Panel titles: at least 23--25pt.
- Axis labels: at least 21--25pt.
- Tick labels, legends, colorbars, and numerical annotations: at least 21pt.
- No visible text may use a source size below 21pt.
- Line widths, markers, legend handles, annotation padding, and subplot spacing
  increase proportionally so the enlarged type does not look detached from the
  plotted evidence.
- The final manuscript PDF is the authoritative check: extracted/rendered text
  must be at least 9pt at `\columnwidth`.

## Cumulative Portfolio-Value Composite

Source:
`paper_experiments/run_paper_experiments_final.py`.

- Generate a single two-panel image containing Nasdaq-100 and CSI-300.
- Use compact market panel titles rather than repeating the full figure title.
- Use one shared vertical label:
  `Portfolio value (initial = 1.0)`.
- Keep date ticks but reduce their density to avoid overlap.
- Use one shared legend below both panels. The legend may span multiple rows;
  method names and colors must remain unchanged.
- Preserve all cumulative curves and the stronger visual emphasis on the
  CMTFlow/`Ours` curve.
- Target a short, wide composite so the main plot consumes as little additional
  vertical manuscript space as possible.

Output:
`paper_full_evidence_edit/figures/main_equity_combined.pdf`.

## Controller Switch-Case Composite

Source:
`paper_experiments/run_paper_experiments_final.py`.

- Generate a 2-by-2 composite: markets are columns; frozen return and drawdown
  are rows.
- Put each market, switch date, and probability in a compact column heading.
- Use shared row labels for Return and Drawdown, and one shared bottom label for
  trading days after switch.
- Use one shared `Hold`/`Switch` legend.
- Preserve endpoint returns, return gap, and MDD reduction, but shorten their
  formatting and place them in protected annotation zones.
- Keep the C row removed.
- Reduce tick density and omit duplicate axis labels on the right column.

Output:
`paper_full_evidence_edit/figures/explainability/controller_switch_cases_combined.png`.

## Inner-Actor Composite

Source:
`paper_experiments/plot_inner_actor_base_adjustment.py`.

- Generate a 4-by-2 composite: markets are columns and the four evidence types
  are rows.
- Use market names as column headings and the four compact evidence names as
  shared row titles.
- Keep the same selected dates, assets, heatmap values, alignment bars,
  correlations, and positive-day ratios.
- Avoid duplicate explanatory titles and x-axis date labels.
- Keep asset names, colorbar units, alignment scores, and hit rates at 9pt or
  larger after manuscript scaling.
- Use compact colorbars and reduced tick counts. Color normalization must remain
  explicit and must not change the numerical evidence.
- Reserve sufficient horizontal space for alignment annotations; no label may
  be clipped.

Output:
`paper_full_evidence_edit/figures/explainability/inner_actor_combined.png`.

## Manuscript Integration

Replace each pair of `0.49\columnwidth` images with one composite image at
`\columnwidth`. Keep the existing single-column `figure` environments, labels,
captions, and surrounding claims. Update only filenames and any caption wording
needed to describe the composite panel order.

The old per-market images remain available as experiment artifacts but are no
longer included in this manuscript version.

## Validation

1. Add structural tests for composite panel counts, shared labels, shared
   legends, and minimum source font sizes.
2. Verify the tests fail against the current separate-image implementation.
3. Generate all three composites from the archived traces without retraining or
   selecting new cases.
4. Inspect the composite assets at original resolution and at simulated
   `\columnwidth` scale for overlap, clipping, and balance.
5. Verify every visible text object has a source size of at least 21pt and at
   least twice the corresponding current size.
6. Verify the manuscript uses one `\columnwidth` image per figure group and
   remains in single-column `figure` environments.
7. Compile the manuscript when `newtxtext.sty` is available. If the local TeX
   dependency remains absent, report the environmental blocker and validate the
   assets and LaTeX references independently.

## Non-Goals and Safety

- Do not change experiment data, metrics, cases, windows, or model results.
- Do not move the figures to `figure*`.
- Do not independently retrain or rerun portfolio evaluation.
- Do not alter unrelated manuscript prose or user changes in the dirty
  worktree.
- Do not stage or commit implementation changes unless separately requested;
  only this design document is committed as required by the design workflow.
