"""The three llr9 figures: speed-up per kernel, success rate, token cost per kernel.

Usage:  python3 plot_llr9_kernels.py [--data data/kernels.csv] [--out figures]
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import aggregate_llr9  # noqa: E402  -- sibling: the experiment's tag size lives with its rules
from benchlib import dumbbell  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.exit(dumbbell.main(ROOT / "data" / "kernels.csv", ROOT / "figures", "llr9", aggregate_llr9.TAG_SIZE))
