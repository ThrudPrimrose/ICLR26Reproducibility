"""Pool every collected llr9 wave into ONE tidy table, one row per (model, language, skills, kernel).

The pooling rules live in :mod:`benchlib.kernels`; this file is the llr9 experiment's two decisions
about them -- what the tag holds, and what is dropped from it.

Usage:  python3 aggregate_llr9.py [--data data] [--out data/kernels.csv]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import kernels  # noqa: E402  -- the artifact is run from a clone, not installed

#: The llr9 kernel set, and the denominator of every success rate: llr8's 40, less the duplicate
#: ``tsvc_2_s13110``, plus the five kernels authored on 2026-09-01. ``argmax_with_index`` is in both
#: counts -- it is re-measured, not added -- so it moves the numbers without moving the size.
#:
#: It is NOT the ``llr-focus40`` tag as the benchmark repository currently holds it. That tag was
#: re-cut on 2026-09-01 to stay at 40: it took the five new kernels in, put ``tsvc_2_s2233`` back,
#: and untagged ``ext_break_find_first``, ``ext_break_post_body``, ``tsvc_2_s232``, ``wavefront2d``
#: and ``wf_north_west``. Those five were measured by llr8 and are kept here, because dropping a
#: measured kernel to match a tag re-cut after the measurement changes what the experiment reports
#: without anyone deciding to.
TAG_SIZE = 44

#: Dropped from every figure. ``tsvc_2_s2233`` took 296 judge calls across llr8 and graded ok ZERO
#: times, in every arm of every wave -- a kernel no arm can score measures the harness, not the
#: model (the pass/fail is size- and thread-dependent; see the open harness bug). llr40v9 did not
#: draw it, so it contributes nothing to llr9 either way.
EXCLUDED = frozenset({"tsvc_2_s2233"})

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data", help="collected wave directories")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "kernels.csv", help="tidy table")
    args = parser.parse_args()
    return kernels.run(args.data, args.out, EXCLUDED, "collect_llr9.py")


if __name__ == "__main__":
    sys.exit(main())
