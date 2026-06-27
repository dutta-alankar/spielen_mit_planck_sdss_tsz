#!/usr/bin/env python
"""Convenience wrapper to run the full planck_cmb pipeline.

Examples
--------
    # quick, fully offline demo on synthetic data
    uv run python scripts/run_pipeline.py --simulate

    # real Planck + SDSS data (downloads several GB on first run)
    uv run python scripts/run_pipeline.py
"""

import sys

from planck_cmb.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["all", *argv]
    sys.exit(main(argv))
