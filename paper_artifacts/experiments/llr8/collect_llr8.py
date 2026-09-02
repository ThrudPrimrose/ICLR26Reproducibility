"""Collect every llr8 wave of the llr-focus40 campaign, end to end, in one command.

WAVE 1 IS NOT HERE, and its absence is deliberate. The ``llr8`` campaign (jobs 608446-608987) and
the same-day databases under ``scratch-s353/llr8-results`` ran on 2026-08-25, before the C
reference fix of 08-26: 208 of 298 ``_reference.c`` were verbatim TSVC, so an agent that followed
the reference it was shown built a library that could not load and the judge recorded that as
``incorrect``. Those C results measure the corpus, not the model.

THE EXPERIMENT IS THE UNION OF ITS WAVES, never one wave directory. The campaign does not fit in
one job, so it is deliberately broken into submission batches: w3, w4, w6, w7 and w12-w15 are
COMPLETION waves that re-run only the kernels an earlier arm never submitted, and w8/w9 (and
w10/w11) are two HALVES of one 40-kernel draw split to fit the node budget. A kernel can therefore
appear in several waves; ``aggregate_llr8.py`` de-duplicates it by the judge's stamp, never by
directory. The per-wave directories stay so the union remains auditable.

The run is DETERMINISTIC: the same shards produce byte-identical CSVs. Nothing here samples,
shuffles or stamps a time, the arms are visited in registry order, and every shard glob is sorted.

It exits NON-ZERO when a registered arm produced no calls. That state is invisible afterwards -- an
arm with no rows is an arm with no row -- so a wave collected while one of its arms was still
grading would quietly become a figure with a bar missing and nothing to say so.

Usage:  python3 collect_llr8.py [--run-root RUN_ROOT] [--out data] [--sources artifacts/sources]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import shards  # noqa: E402  -- the artifact is run from a clone, not installed
from benchlib import sources  # noqa: E402

#: The kernel every llr8 arm reached and no llr8 arm ever scored; see ``aggregate_llr8.EXCLUDED``.
EXCLUDED = frozenset({"tsvc_2_s2233"})

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=pathlib.Path, default=shards.RUN_ROOT, help="tree holding <jobid>/judge/")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data", help="one directory per wave")
    parser.add_argument("--sources",
                        type=pathlib.Path,
                        default=ROOT / "artifacts" / "sources",
                        help="where each cell's final submitted source is written")
    args = parser.parse_args()

    waves = [(campaign, campaign.removeprefix("llr8")) for campaign in shards.wave_campaigns("llr8")]
    lines, empty = shards.collect_waves(args.run_root, args.out, waves)
    sources.export(args.run_root, args.out, args.sources, EXCLUDED)

    print("\ncollected:")
    for line in lines:
        print(line)
    if empty:
        print("\nregistered arms with NO calls: " + ", ".join(empty), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
