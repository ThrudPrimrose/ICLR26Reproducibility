"""Collect every llr8 wave of the llr-focus40 campaign, end to end, with no arguments.

One command for the whole campaign, because the collection was a hand-typed
``collect.py --campaign llr8wN --out data-llr8wN`` per wave and the waves that got missed are the
ones nobody typed: data-llr8w6 sat frozen at two arms for a day after its kimi pair finished, and
w7, w8 and w9 had never been collected at all. The wave list comes from :data:`collect.ARMS`, so
registering an arm is the only step; there is no second list to keep in step with it.

The run is DETERMINISTIC: the same shards produce byte-identical CSVs. Nothing here samples,
shuffles or stamps a time, the arms are visited in registry order, and every shard glob is sorted.
Re-running it overwrites the same files with the same bytes.

It exits NON-ZERO when a registered arm produced no calls. That state is invisible afterwards -- an
arm with no rows is an arm with no row -- so a wave collected while one of its arms was still
grading would quietly become a figure with a bar missing and nothing to say so.

Usage:  python3 collect_llr40.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

import collect


def waves() -> list[str]:
    """Registered ``llr8w<N>`` campaigns, in wave order.

    Numeric, not lexicographic: ``llr8w10`` sorts before ``llr8w2`` as text, and a wave printed out
    of order in the summary is the kind of thing that gets read as a missing wave.
    """
    names = {str(a["campaign"]) for a in collect.ARMS if collect.llr8_wave(str(a["campaign"]))}
    return sorted(names, key=lambda n: int(n[len("llr8w"):]))


def wave_root(arms: list[dict[str, object]]) -> pathlib.Path:
    """The directory this wave's jobs sit in: ``RUN_ROOT`` itself, or a campaign-family subdirectory.

    A completion wave inherits its RUN_ROOT from the env file it was derived from, so waves 4, 6 and
    7 all write under ``llr8w4-20260829/``, wave 2 under ``llr8w1-20260827/`` and waves 8 and 9 under
    ``llr8w8-20260830/``. The directory name is therefore NOT the wave name and cannot be built from
    it. The job id is the only reliable handle, so this looks for the parent that holds one.
    """
    roots = [collect.RUN_ROOT, *sorted(p for p in collect.RUN_ROOT.iterdir() if p.is_dir())]
    for arm in arms:
        for root in roots:
            if (root / str(arm["job"])).is_dir():
                return root
    raise SystemExit(f"no job of {arms[0]['campaign']} has a run directory under {collect.RUN_ROOT}")


def summary_rows(out: pathlib.Path) -> list[dict[str, str]]:
    with (out / "summary.csv").open() as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    empty: list[str] = []
    lines: list[str] = []
    for wave in waves():
        arms = [a for a in collect.ARMS if a.get("campaign") == wave]
        out = root / f"data-{wave}"
        runs = wave_root(arms)
        print(f"=== {wave}: {len(arms)} registered arms, runs under {runs} -> {out}")
        empty += collect.collect_campaign(runs, out, arms)
        rows = summary_rows(out)
        kernels = sum(int(r["speedup_kernels"]) for r in rows)
        solved = sum(int(r["solved"]) for r in rows)
        problems = sum(int(r["problems"]) for r in rows)
        lines.append(f"  {wave:8s} arms={len(rows):2d}  solved={solved:3d}/{problems:3d}  timed kernels={kernels:3d}")

    print("\ncollected:")
    for line in lines:
        print(line)
    if empty:
        print("\nregistered arms with NO calls: " + ", ".join(empty), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
