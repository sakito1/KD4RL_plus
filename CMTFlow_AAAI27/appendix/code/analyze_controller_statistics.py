#!/usr/bin/env python3
"""Public Appendix C.2 entry point for Controller statistical validation."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PACKAGE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paper_experiments.analyze_controller_adaptive_timing import *  # noqa: F403
from paper_experiments.analyze_controller_adaptive_timing import main


if __name__ == "__main__":
    raise SystemExit(main())
