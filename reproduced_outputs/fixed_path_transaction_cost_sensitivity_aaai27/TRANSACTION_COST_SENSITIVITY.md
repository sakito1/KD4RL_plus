# Fixed-Path Transaction-Cost Sensitivity

## Material Passport

- Verification Status: ANALYZED
- Scope: paper-selected environment-recorded paths held fixed
- Reference cost: 0.005%
- Replay: recover pre-cost growth from recorded net return and turnover
- Limitation: policy actions and Controller switches are not recomputed

## Results

| Market | Cost | TR | SR | MDD | CR | ΔTR (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Nasdaq-100 | 0.005% | 265.53% | 1.150 | 18.62% | 1.424 | +0.00 |
| Nasdaq-100 | 0.010% | 262.49% | 1.144 | 18.66% | 1.412 | -3.04 |
| Nasdaq-100 | 0.015% | 259.48% | 1.137 | 18.70% | 1.400 | -6.05 |
| Nasdaq-100 | 0.020% | 256.49% | 1.130 | 18.75% | 1.389 | -9.04 |
| Nasdaq-100 | 0.050% | 239.06% | 1.090 | 19.01% | 1.321 | -26.46 |
| CSI-300 | 0.005% | 240.13% | 1.246 | 22.70% | 1.194 | +0.00 |
| CSI-300 | 0.010% | 237.01% | 1.237 | 22.91% | 1.175 | -3.11 |
| CSI-300 | 0.015% | 233.92% | 1.228 | 23.12% | 1.157 | -6.20 |
| CSI-300 | 0.020% | 230.87% | 1.220 | 23.32% | 1.138 | -9.26 |
| CSI-300 | 0.050% | 213.09% | 1.168 | 24.56% | 1.036 | -27.03 |

## Interpretation Boundary

This table isolates the mechanical effect of charging alternative costs to the same environment-recorded path. It does not represent inference or training under the alternative cost rates.
