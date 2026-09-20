"""Scratch: pull the graded candidate text of the best submission per (arm, kernel).

Walks the judge databases under the run roots and writes each arm's best-scoring candidate to
<out>/<arm>/<kernel>.<ext> with a one-line header naming the speed-up. Read-only.

PAIRING. A run grades the same kernel many times, and `sources` carries one row per graded
candidate. Keying sources by (run_id, benchmark) into a dict makes the LAST candidate win while
the speed-up still comes from the BEST round, so the header names a number the body never
produced. Every file the earlier version wrote was mislabelled that way. Each submission is
matched instead to the last source row at or before its own timestamp.
"""

import argparse
import bisect
import pathlib
import re
import sqlite3
import sys

EXT = {"c": "c", "cpp": "cpp", "fortran": "f90", "python": "py", "triton": "py", "hip": "hip"}


def judge_dbs(root: pathlib.Path):
    yield from sorted(root.glob("*/judge/rank-*/hpcagent_bench*.db"))


def blob_store(db: pathlib.Path) -> pathlib.Path:
    return db.with_name(db.stem + "_prompts")


def source_timeline(conn: sqlite3.Connection) -> dict:
    """(run_id, kernel) -> ([ts], [(language, path)]), both in ascending ts order."""
    timeline: dict = {}
    for run_id, ts, bench, lang, path in conn.execute(
        "select run_id, ts, benchmark, language, path from sources order by ts"
    ):
        key = (run_id, str(bench).split("/")[-1])
        stamps, rows = timeline.setdefault(key, ([], []))
        stamps.append(ts)
        rows.append((lang, path))
    return timeline


def harvest(db: pathlib.Path, arm_pattern: re.Pattern[str], best: dict, store_by_key: dict) -> None:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        if not {"submissions", "sources", "runs"} <= tables:
            return
        runs = {r[0]: (r[1] or "") for r in conn.execute("select run_id, arm from runs")}
        timeline = source_timeline(conn)
        cols = {r[1] for r in conn.execute("pragma table_info(submissions)")}
        if "speedup" not in cols:
            return
        for run_id, ts, bench, sp, suspect in conn.execute(
            "select run_id, ts, benchmark, speedup, suspect from submissions where speedup is not null"
        ):
            arm = runs.get(run_id, "")
            if not arm_pattern.search(arm):
                continue
            kernel = str(bench).split("/")[-1]
            key = (arm, kernel)
            if key in best and best[key][0] >= float(sp):
                continue
            found = timeline.get((run_id, kernel))
            if found is None:
                continue
            stamps, rows = found
            at = bisect.bisect_right(stamps, ts) - 1
            if at < 0:
                continue
            best[key] = (float(sp), rows[at][0], rows[at][1], int(suspect or 0), run_id)
            store_by_key[key] = blob_store(db)
    except sqlite3.Error:
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=pathlib.Path, required=True)
    ap.add_argument("--glob", default="gpu-llr-focus40-*")
    ap.add_argument("--arms", default=".", help="regex an arm must match")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    pattern = re.compile(args.arms)
    best: dict = {}
    stores: dict = {}
    for root in sorted(args.runs.glob(args.glob)):
        for db in judge_dbs(root):
            harvest(db, pattern, best, stores)
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for (arm, kernel), (sp, lang, path, suspect, run_id) in sorted(best.items()):
        blob = stores[(arm, kernel)] / path
        if not blob.is_file():
            continue
        target = args.out / arm
        target.mkdir(parents=True, exist_ok=True)
        text = blob.read_text(errors="replace")
        ext = EXT.get(str(lang), "txt")
        flag = " SUSPECT" if suspect else ""
        head = f"// speedup={sp:.3f} arm={arm} kernel={kernel} run={run_id} blob={path}{flag}\n"
        (target / f"{kernel}.{ext}").write_text(head + text)
        written += 1
    print(f"{written} candidate(s) -> {args.out}", file=sys.stderr)
    print(f"{len(best)} (arm,kernel) pairs seen", file=sys.stderr)


if __name__ == "__main__":
    main()
