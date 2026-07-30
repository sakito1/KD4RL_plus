#!/usr/bin/env python3
"""Public Appendix C.3 entry point for Trader statistical validation."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paper_experiments.analyze_inner_outer_statistical_validation import *  # noqa: F403
from paper_experiments.analyze_inner_outer_statistical_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
