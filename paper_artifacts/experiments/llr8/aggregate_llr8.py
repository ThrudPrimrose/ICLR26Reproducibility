"""Pool every collected llr8 wave into ONE tidy table, one row per (model, language, skills, kernel).

The pooling rules live in :mod:`benchlib.kernels`; this file is the llr8 experiment's two decisions
about them -- what the tag holds, and what is dropped from it.

Usage:  python3 aggregate_llr8.py [--data data] [--out data/kernels.csv]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import kernels  # noqa: E402  -- the artifact is run from a clone, not installed

#: The llr-focus40 tag as the llr8 waves drew it, and the denominator of every success rate. Not
#: read off any one arm, because a completion arm draws a SUBSET of the tag and would report its
#: own gap list as the experiment's denominator.
TAG_SIZE = 40

#: Dropped from every figure. ``tsvc_2_s2233`` took 296 judge calls across the campaign and graded
#: ok ZERO times, in every arm of every wave -- a kernel no arm can score measures the harness, not
#: the model (the pass/fail is size- and thread-dependent; see the open harness bug). It was swapped
#: out of the tag for ``tsvc_2_s232`` on 2026-08-30, which is the only reason a pooled cell could
#: show 41 distinct kernels over a 40-kernel tag. Excluding it makes the pooled set exactly the tag,
#: costs no numerator anywhere (it was never solved), and stops its 296 unscoreable calls from
#: inflating the token cost of the kernels that ARE scoreable.
EXCLUDED = frozenset({"tsvc_2_s2233"})

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data", help="collected wave directories")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "kernels.csv", help="tidy table")
    args = parser.parse_args()
    return kernels.run(args.data, args.out, EXCLUDED, "collect_llr8.py")


if __name__ == "__main__":
    sys.exit(main())
