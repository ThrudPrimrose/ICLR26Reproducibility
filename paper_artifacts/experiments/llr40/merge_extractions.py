"""Fold a fresh extraction into the committed observations, keeping arms whose runs are gone.

``extract_llr40.py`` rebuilds ``llr40_observations.csv`` from the run roots it is pointed at, so an
arm whose run root no longer exists simply vanishes from the output. Twelve GPU arms are in that
position -- ``gpuv2-llr40-20260906`` and ``gpuv3-llr40-20260907`` were deleted from scratch after
their rows were committed, and 4,386 observations exist nowhere else. For those arms the committed
CSV IS the source of truth, and re-running the extraction over the surviving CPU roots would
silently drop them.

This keeps every committed row whose arm the new extraction does not produce, and takes the new
extraction's rows for every arm it does -- so a re-measured arm is replaced whole rather than
duplicated. Headers must match exactly; a schema change means the two cannot be merged and the
mismatch is reported instead of papered over.
"""
import argparse
import collections
import csv
import pathlib


def read(path: pathlib.Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def merge(committed: pathlib.Path, fresh: pathlib.Path, out: pathlib.Path, arm_column: str) -> None:
    old_cols, old_rows = read(committed)
    new_cols, new_rows = read(fresh)
    if old_cols != new_cols:
        raise SystemExit(f"schema differs: only in committed {set(old_cols) - set(new_cols)}, "
                         f"only in fresh {set(new_cols) - set(old_cols)}")
    fresh_arms = {row[arm_column] for row in new_rows}
    kept = [row for row in old_rows if row[arm_column] not in fresh_arms]
    kept_arms = collections.Counter(row[arm_column] for row in kept)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_cols)
        writer.writeheader()
        writer.writerows(kept + new_rows)
    print(f"kept {len(kept)} committed rows across {len(kept_arms)} arms the fresh run does not cover:")
    for arm, n in sorted(kept_arms.items()):
        print(f"    {arm:44s} {n}")
    print(f"took {len(new_rows)} fresh rows across {len(fresh_arms)} arms")
    print(f"wrote {len(kept) + len(new_rows)} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("committed", type=pathlib.Path)
    parser.add_argument("fresh", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--arm-column", default="arm")
    args = parser.parse_args()
    merge(args.committed, args.fresh, args.out, args.arm_column)


if __name__ == "__main__":
    main()
