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

#: The ``llr-focus40`` tag as the benchmark repository holds it after the 2026-09-01 re-cut, and the
#: denominator of every success rate. The re-cut kept the tag at forty: it took the five kernels
#: authored that day in, put ``tsvc_2_s2233`` back, deleted the duplicate ``tsvc_2_s13110``, and
#: untagged the five kernels those replaced. llr9 is the experiment over THAT roster, so the size is
#: the tag's, not llr8's forty plus whatever llr8 happened to have measured.
#:
#: The pooled table holds THIRTY-NINE kernels against this forty, and the missing one is
#: ``tsvc_2_s2233`` below. That is deliberate: an arm was given forty kernels, one of which the
#: harness cannot score, so the denominator counts it and the numerator cannot. Reporting 39 here
#: instead would quietly credit every leg for a kernel it was never able to solve.
TAG_SIZE = 40

#: Dropped from every figure. ``tsvc_2_s2233`` took 296 judge calls across llr8 and graded ok ZERO
#: times, in every arm of every wave -- a kernel no arm can score measures the harness, not the
#: model (the pass/fail is size- and thread-dependent; see the open harness bug). It is back IN the
#: re-cut tag, so it counts towards :data:`TAG_SIZE`; it is kept out of the table so that its 296
#: unscoreable calls do not inflate the token cost of the kernels that ARE scoreable.
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
