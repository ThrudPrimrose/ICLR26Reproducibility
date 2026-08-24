"""Pull per-kernel results out of the judge SQLite DBs for a set of arms.

Runs on the login node: sqlite3 only, no numpy, no hpcagent_bench import (neither is
installed for python3.11 there). Emits one JSON blob the plotting step consumes, so the
DBs are read exactly once.
"""
import collections
import glob
import json
import pathlib
import sqlite3
import sys

RUNS = pathlib.Path("/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs")

#: job id -> (model, language, skills). llr6v10 only: one skills revision, so the arms are
#: comparable to each other and nothing older is mixed in.
ARMS = {
    605458: ("qwen30b", "c", False),
    605459: ("qwen30b", "c", True),
    605460: ("oss120b", "c", False),
    605461: ("oss120b", "c", True),
    605696: ("qwen30b", "fortran", False),
    605697: ("qwen30b", "fortran", True),
    605700: ("kimi27-sglang", "c", False),
}


def arm_rows(job: int) -> dict[str, object]:
    best: dict[str, float] = {}
    subs = 0
    status: collections.Counter[str] = collections.Counter()
    rounds: list[int] = []
    for db in glob.glob(str(RUNS / str(job) / "judge" / "rank-*" / "*.db")):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for bench, speedup in con.execute(
                "select benchmark, speedup from submissions where speedup is not null"):
            subs += 1
            if bench not in best or speedup > best[bench]:
                best[bench] = speedup
        for bench, st, rnd in con.execute("select benchmark, status, round from calls"):
            status[st] += 1
            if rnd is not None:
                rounds.append(rnd)
        con.close()
    return {
        "best": best,
        "submissions": subs,
        "status": dict(status),
        "max_round": max(rounds) if rounds else 0,
    }


def main() -> None:
    out = {}
    for job, (model, lang, skills) in ARMS.items():
        data = arm_rows(job)
        data.update(job=job, model=model, language=lang, skills=skills)
        out[str(job)] = data
        print(f"{job} {model:14s} {lang:8s} skills={str(skills):5s} "
              f"kernels={len(data['best']):3d} subs={data['submissions']:4d} "
              f"status={data['status']}", file=sys.stderr)
    dest = pathlib.Path(__file__).with_name("arms.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
