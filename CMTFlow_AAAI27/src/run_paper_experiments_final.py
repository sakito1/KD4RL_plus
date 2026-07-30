#!/usr/bin/env python3
"""Run the canonical final paper experiment entry point."""

import os
from pathlib import Path

from paper_experiments.run_paper_experiments_final import main


def run() -> None:
    os.chdir(Path(__file__).resolve().parent)
    main()


if __name__ == "__main__":
    run()
