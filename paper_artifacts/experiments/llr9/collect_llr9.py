"""Collect llr9: the llr8 waves for every kernel they still speak for, plus the llr40v9 re-measure.

WHAT LLR9 IS. The same experiment as llr8 -- three models, two languages, the skills packet in the
prompt or absent -- over the ``llr-focus40`` tag AS THE BENCHMARK REPOSITORY NOW HOLDS IT. The tag
was re-cut on 2026-09-01 and llr9 is the experiment over the re-cut roster, not over the roster llr8
happened to draw: six of its forty kernels are taken from the ``llr40v9`` campaign, six of llr8's
are dropped because they are no longer in the tag, and the remaining thirty-four are llr8's
unchanged, because re-running a kernel nothing changed about would cost a day of nodes to measure
the same thing twice.

THE TAG IS AUTHORITATIVE, NOT THE MEASUREMENT HISTORY. The five untagged kernels below were REPLACED
deliberately -- three ext_break kernels collapsed into one, four wavefronts into two, and
``tsvc_2_s232`` removed as agent hyper-specialisation -- so carrying them because llr8 happened to
measure them would report a roster the benchmark no longer has.

    REFRESHED, taken from llr40v9 and filtered out of the inherited waves:

      compact_threshold_pack, scan_affine_decay, scatter_accum_dup, segment_reduce_ragged and
      versioned_distance_update are NEW -- authored 2026-09-01 -- and have no llr8 history at all,
      so for them the filter removes nothing and the whole cell comes from llr40v9.

      argmax_with_index is an OLD kernel RE-MEASURED. Its Fortran arm was unwinnable until
      2026-09-01: ``out_index`` is declared ``index_array: true`` and the ABI seam rebases index
      buffers in both directions, so a Fortran submission must store the ONE-BASED position.
      Nothing in the task said so and ``skills/lang-fortran/SKILL.md`` stated the opposite. Measured
      with the same loop body and only the stored base differing, 1-based grades ``correct=True``
      5/5 hidden and 0-based grades ``correct=False, out_index: integer mismatch: 1 of 1
      elements``. Its pre-2026-09-01 rows therefore measure the prompt, not the model. The C and
      C++ rows are unaffected (``index_base`` is 0 for those) but are dropped with the rest: half a
      kernel measured under two prompts is not one kernel, and llr40v9 re-ran C as well.

    DROPPED as a duplicate:

      tsvc_2_s13110 was removed from the benchmark repository as a byte-identical duplicate of
      tsvc_2_s3110 -- same ``_numpy.py``, same ``_dace.py``, byte-identical ``_reference.c``, same
      presets and sizes. Its 80 calls and 21 llr8 submissions are dropped rather than folded into
      s3110: an agent that drew both drew the same code twice as two independent problems, and
      merging the two cells would report one kernel's two attempts as one attempt at one kernel
      while inflating that kernel's token cost. It is a deletion, not a rename.

    DROPPED as replaced:

      ext_break_find_first, ext_break_post_body, tsvc_2_s232, wavefront2d and wf_north_west were
      untagged from llr-focus40 in the same re-cut that took the five new kernels in. They were not
      retired for being uninteresting: the three ext_break kernels collapsed into one, the four
      wavefronts into two, and s232 came out because agents were hyper-specialising on it. Keeping
      their llr8 rows would let llr9 report a forty-five kernel roster the benchmark does not have.

THE EXPERIMENT IS THE UNION OF ITS WAVES. llr9 inherits llr8's thirteen submission batches and adds
one of its own, and the per-wave directories stay so that union is auditable: the wave a row came
from is in the directory it sits in, and ``aggregate_llr9.py`` de-duplicates a kernel that appears
in several by the judge's stamp, never by picking a directory.

Usage:  python3 collect_llr9.py [--run-root RUN_ROOT] [--out data] [--sources artifacts/sources]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import shards  # noqa: E402  -- the artifact is run from a clone, not installed
from benchlib import sources  # noqa: E402

#: Measured fresh by llr40v9 and therefore filtered out of every inherited llr8 wave. Five have no
#: llr8 rows to filter; ``argmax_with_index`` has 122 calls and 21 submissions that do come out.
REFRESHED = frozenset({
    "argmax_with_index",
    "compact_threshold_pack",
    "scan_affine_decay",
    "scatter_accum_dup",
    "segment_reduce_ragged",
    "versioned_distance_update",
})

#: Removed from the benchmark repository as a duplicate; dropped from the inherited waves with it.
DUPLICATE = frozenset({"tsvc_2_s13110"})

#: Untagged from ``llr-focus40`` in the 2026-09-01 re-cut, having been REPLACED by the kernels that
#: came in with it. llr8 measured all five; llr9 drops them, because the tag is the roster and a
#: kernel the benchmark no longer offers is not part of what this experiment is over.
REPLACED = frozenset({
    "ext_break_find_first",
    "ext_break_post_body",
    "tsvc_2_s232",
    "wavefront2d",
    "wf_north_west",
})

#: The kernel no arm has ever scored; see ``aggregate_llr9.EXCLUDED``.
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

    inherited = [(campaign, campaign.removeprefix("llr8")) for campaign in shards.wave_campaigns("llr8")]
    lines, empty = shards.collect_waves(args.run_root, args.out, inherited, REFRESHED | DUPLICATE | REPLACED)
    fresh, fresh_empty = shards.collect_waves(args.run_root, args.out, [("llr40v9", "v9")])
    sources.export(args.run_root, args.out, args.sources, EXCLUDED)

    print("\ncollected:")
    for line in lines + fresh:
        print(line)
    if empty or fresh_empty:
        print("\nregistered arms with NO calls: " + ", ".join(empty + fresh_empty), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
