"""Find speed-up values shared by more than one (arm, kernel) across the judge databases.

A graded speed-up is a per-run measurement. Two unrelated kernels agreeing on one to 17 significant
digits is not a coincidence, so any value carried by more than one (arm, kernel) is a defect
candidate. Prints the per-family exposure and the worst values. Read-only.
"""

import collections
import pathlib
import sqlite3
import sys


def family(arm: str) -> str:
    for prefix in ("llrblind", "git-scicomp"):
        if arm.startswith(prefix):
            return prefix
    return arm.split("-focus40")[0] if "-focus40" in arm else "other"


def graded_rows(root: pathlib.Path) -> list[tuple[float, str, str, int]]:
    rows: list[tuple[float, str, str, int]] = []
    for db in root.glob("*/*/judge/rank-*/hpcagent_bench*.db"):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
            if not {"submissions", "runs"} <= tables:
                continue
            arms = {r[0]: (r[1] or "?") for r in conn.execute("select run_id, arm from runs")}
            for run_id, bench, speedup, suspect in conn.execute(
                "select run_id, benchmark, speedup, suspect from submissions where speedup is not null"
            ):
                rows.append((float(speedup), arms.get(run_id, "?"), str(bench).split("/")[-1], int(suspect or 0)))
        except sqlite3.Error:
            continue
    return rows


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "hpcagent-bench-runs")
    rows = graded_rows(root)
    by_value: dict[float, list] = collections.defaultdict(list)
    for row in rows:
        by_value[row[0]].append(row)
    # 1.0 is the score of a failed submission and legitimately repeats.
    colliding = {v for v, g in by_value.items() if v != 1.0 and len({(a, k) for _, a, k, _ in g}) > 1}
    seen = collections.Counter(family(r[1]) for r in rows)
    hit = collections.Counter(family(r[1]) for r in rows if r[0] in colliding and r[3] == 0)
    print(f"{len(rows)} graded rows, {len(by_value)} distinct speed-ups, {len(colliding)} colliding values")
    print(f"{'family':16s} {'rows':>7s} {'credited colliding':>19s} {'share':>7s}")
    for name in sorted(seen, key=lambda n: -seen[n]):
        print(f"{name:16s} {seen[name]:>7d} {hit[name]:>19d} {hit[name] / seen[name]:>6.1%}")
    for value in sorted(colliding, key=lambda v: -len(by_value[v]))[:5]:
        print(f"\n  {value!r}  rows={len(by_value[value])}")
        for _, arm, kernel, suspect in sorted({r for r in by_value[value]})[:6]:
            print(f"      {arm} / {kernel}  suspect={suspect}")


if __name__ == "__main__":
    main()
