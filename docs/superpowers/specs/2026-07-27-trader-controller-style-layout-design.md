# Trader Controller-Style Layout

## Goal

Make the Trader Refinement figure use the same reading order as the Controller
case figure.

## Layout

- Use a 2-by-2 grid.
- Rows are markets: CSI-300 on top and Nasdaq-100 below.
- Columns are metrics:
  - left: future 5-day relative return;
  - right: refinement tilt.
- Put the market name once at the far left of each row.
- Put the column captions once below the figure:
  - `A. Future 5-day relative return`
  - `B. Refinement tilt`
- Remove all repeated titles above individual heatmaps.
- Retain one shared colorbar for each metric column.

## Presentation

- Preserve the current enlarged fonts and 240-DPI PNG/PDF output.
- Keep both markets on the same color scale within each metric column.
- Keep date labels on the bottom row only to avoid repetition.
- Preserve the output basename `trader_refinement_two_markets`.

## Verification

- A layout test checks row/column ordering, unique bottom captions, market labels,
  shared scales, colorbar placement, and font sizes.
- Regenerate the PNG and PDF through the canonical root entrypoint and inspect
  the PNG for overlap or clipping.
