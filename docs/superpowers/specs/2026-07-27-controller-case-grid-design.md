# Controller Case 2×2 Grid Design

## Goal

Generate four candidate controller figures from the Cartesian product of the two
selected CSI-300 cases and the two selected Nasdaq-100 cases. Each candidate is
a 2×2 figure that makes the two markets directly comparable:

|              | A. Future return | B. Future drawdown |
|--------------|------------------|--------------------|
| CSI-300      | selected SH case | selected SH case   |
| Nasdaq-100   | selected NAS case| selected NAS case  |

The existing single-market controller figures remain available for independent
formatting and inspection.

## Case Pairing and Outputs

With the default `--case_count 2`, create these four combinations:

1. SH case 1 + NAS case 1
2. SH case 1 + NAS case 2
3. SH case 2 + NAS case 1
4. SH case 2 + NAS case 2

Use stable output names:

- `controller_case_combined_sh01_nas01`
- `controller_case_combined_sh01_nas02`
- `controller_case_combined_sh02_nas01`
- `controller_case_combined_sh02_nas02`

PNG and PDF variants follow the existing `save_figure` behavior.

## Layout and Formatting

- Use two equal-width columns and two equal-height rows.
- Put `CSI-300` at the far left of the first row and `Nasdaq-100` at the far
  left of the second row.
- The left column shows future return and the right column shows future
  drawdown.
- Show the A/B captions once per column at the bottom of the full figure.
- Use one shared legend at the top of the full figure.
- Each panel uses horizon ticks `1, 5, 10, 15, 20, 25, 30`.
- Each panel retains its own `YYYY-MM-DD—YYYY-MM-DD` date-range label because
  the selected SH and NAS cases can have different dates.
- Preserve the current curve colors, shaded advantage regions, endpoint
  annotations, and drawdown summary annotations.

## Code Structure

Extract the plotting of one return/drawdown row into reusable helpers so the
single-market 1×2 figure and combined 2×2 figure use the same calculations and
visual formatting. The controller experiment first selects and stores cases for
both markets, continues generating the existing single-market figures, and then
generates every SH/NAS case combination when both markets are present.

If either market is absent, no combined figure is generated; existing
single-market behavior remains unchanged.

## Verification

Automated layout tests will verify:

- four combined figures are requested for two cases per market;
- each combined figure has exactly four axes;
- rows and columns are aligned and equally sized;
- the two market labels appear at the far left;
- one shared legend is present;
- A/B captions appear once at the bottom;
- horizon ticks and per-row date ranges remain correct.

The final generated PNGs will also be inspected for clipping, overlap, and
readability.
