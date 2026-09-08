"""Collect one canon sweep's per-rank CSVs into this experiment's single tidy table.

Cluster only: it reads a run directory written by ``containers/cluster/example-script/canon_column.sh``,
which shards by rank (``<column>.rank<N>.csv``) because two ranks appending to one file interleave
partial lines. The shards of a column are disjoint kernel sets, so concatenating them is the whole
merge.

Every row of the output names the run it came from. A speedup is only meaningful against a baseline
measured on the SAME node under the SAME configuration -- the columns of one sweep share a job, a
node and a preset, and columns from two sweeps do not.

Usage:  python3 collect_canon.py --run-dir <dir> [--label <name>] [--out data/canon_llr40.csv]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

#: Column order on every figure. cc is the baseline the others are divided by, and it is kept in the
#: table (as a constant 1.0) so a reader can see it was measured rather than assumed.
COLUMNS = ("cc", "cc_autopar", "numba", "dace_cpu", "dace_cpu_canonicalize", "dace_gpu",
           "dace_gpu_canonicalize")

FIELDS = ("run", "column", "kernel", "preset", "datatype", "median_ms", "validated")


def rows_for(run_dir: pathlib.Path, column: str, run: str) -> list[dict[str, str]]:
    """Every rank shard of ``column``, as tidy rows. Missing shards are not an error: a column that
    was not part of a sweep simply contributes nothing."""
    out: list[dict[str, str]] = []
    for shard in sorted(run_dir.glob(f"{column}.rank*.csv")):
        with shard.open() as fh:
            for row in csv.DictReader(fh):
                out.append({
                    "run": run,
                    "column": column,
                    "kernel": row["kernel"],
                    "preset": row.get("preset", ""),
                    "datatype": row.get("datatype", ""),
                    "median_ms": row.get("median_ms", ""),
                    "validated": row.get("validated", ""),
                })
    return out


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--label", default=None, help="run name recorded in the table (default: dir name)")
    ap.add_argument("--out", type=pathlib.Path, default=here / "data" / "canon_llr40.csv")
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"no such run directory: {args.run_dir}", file=sys.stderr)
        return 2
    label = args.label or args.run_dir.name
    rows = [r for column in COLUMNS for r in rows_for(args.run_dir, column, label)]
    if not rows:
        print(f"{args.run_dir} holds no <column>.rank*.csv shards", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["column"], r["kernel"])))
    per = {c: sum(1 for r in rows if r["column"] == c) for c in COLUMNS}
    print(f"{args.out}: {len(rows)} rows  " + "  ".join(f"{c}={n}" for c, n in per.items() if n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
