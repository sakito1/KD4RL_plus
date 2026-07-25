# Population-Level Mechanism Statistics Implementation Plan

1. Add failing unit tests for full-horizon Controller filtering, drawdown sign,
   active-support Inner IC, tilt quintiles, and completed holding segments.
2. Implement a standalone analysis script that reads the existing final-model
   action traces and local market price files.
3. Generate detailed CSV tables and a Chinese Markdown report with a compact
   paper-candidate summary table.
4. Run focused tests, the analysis script, output integrity checks, and the
   existing module-coordination test suite.
