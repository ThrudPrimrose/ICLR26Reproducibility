"""Emit a problems file per arm holding only the kernels that arm never reached.

Arms did not all get through the 242-kernel set: gpt-oss-120b with skills reached 130 while its
skills-off pair reached 192, on the same token spend, because the skills packet makes the prompt
250x larger (task field: 23,249 chars vs 93). That turns solved/242 into a measure of throughput
as much as capability -- the two denominators disagree only in that cell. Re-running just the
unreached kernels closes the gap so both denominators mean the same thing.

Usage:  python3 make_fill_problems.py [--data data] [--problems problems] [--out fill]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--problems", type=pathlib.Path, default=pathlib.Path("problems"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("fill"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reached: dict[str, set[str]] = collections.defaultdict(set)
    with (args.data / "calls.csv").open() as handle:
        for row in csv.DictReader(handle):
            reached[row["arm"]].add(row["benchmark"])

    with (args.data / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))

    print(f"{'arm':<28} {'reached':>8} {'missing':>8}  {'file':<44}")
    for row in summary:
        arm, language = row["arm"], row["language"]
        skills = row["skills"] == "1"
        source = args.problems / (f"problems-llr4-{language}-skills.jsonl"
                                  if skills else f"problems-llr2-{language}.jsonl")
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        # The judge records a benchmark by its bare name; the problems file carries the full
        # track/kernel path. The 242 tails are unique, verified before relying on this.
        missing = [r for r in records if r["kernel"].rsplit("/", 1)[-1] not in reached[arm]]
        if not missing:
            print(f"{arm:<28} {len(reached[arm]):>8} {0:>8}  (complete)")
            continue
        # Renumber so ids stay contiguous, as make_problems.py emits them.
        for index, record in enumerate(missing):
            record["id"] = index
        path = args.out / f"problems-fill-{arm}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in missing))
        print(f"{arm:<28} {len(reached[arm]):>8} {len(missing):>8}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
