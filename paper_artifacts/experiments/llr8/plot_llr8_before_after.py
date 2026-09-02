"""The llr8 connected before/after scatter: every kernel's point without skills joined to its point with.

Usage:  python3 plot_llr8_before_after.py [--data data/kernels.csv] [--out figures]
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import beforeafter  # noqa: E402  -- the artifact is run from a clone, not installed

ROOT = pathlib.Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(beforeafter.main(ROOT / "data" / "kernels.csv", ROOT / "figures", "llr8"))
