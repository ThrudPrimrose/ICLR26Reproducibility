"""Extract per-arm results from the judge shard databases into tidy CSVs.

Each campaign arm writes one SQLite shard per judge rank under
<RUN_ROOT>/<jobid>/judge/rank-*/hpcagent_bench*.db. The shards together are ~3 GB, far too large
to version, so this reduces them to two small CSVs that every figure and number in the paper is
derived from.

Two routes appear in `calls`. `score` is the agent iterating against the judge and says nothing
about the final answer; `submit` is the answer, re-timed on a second seed. Only `submit` rows
count towards success and speedup, while token cost sums over every call the arm made.

Usage:  python3 collect.py [--run-root PATH] [--out data]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import pathlib
import math
import sqlite3
import sys

# Job ids are the record of which Slurm run produced which arm; see experiments/<arm>/README.md.
ARMS: list[dict[str, object]] = [
    {
        "job": 601850,
        "model": "qwen30b",
        "language": "c",
        "skills": False
    },
    {
        "job": 601851,
        "model": "qwen30b",
        "language": "c",
        "skills": True
    },
    {
        "job": 601852,
        "model": "qwen30b",
        "language": "fortran",
        "skills": False
    },
    {
        "job": 602070,
        "model": "oss120b",
        "language": "c",
        "skills": False
    },
    {
        "job": 602071,
        "model": "oss120b",
        "language": "c",
        "skills": True
    },
    {
        "job": 602072,
        "model": "oss120b",
        "language": "fortran",
        "skills": False
    },
    {
        "job": 602073,
        "model": "oss120b",
        "language": "fortran",
        "skills": True
    },
]

# Every arm draws from the same 242-kernel loop_level_reasoning set, so success is reported
# against it rather than against however many kernels an arm managed to reach.
PROBLEM_COUNT = 242

CALL_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "route", "status", "correct", "tokens", "speedup",
    "compiler"
]
SUBMISSION_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "preset", "baseline_ns", "native_ns", "speedup", "suspect"
]


def arm_name(arm: dict[str, object]) -> str:
    suffix = "-skills" if arm["skills"] else ""
    return f"llr4-{arm['model']}-{arm['language']}{suffix}"


def shards(run_root: pathlib.Path, job: int) -> list[str]:
    found = sorted(glob.glob(str(run_root / str(job) / "judge" / "rank-*" / "*.db")))
    if not found:
        raise SystemExit(f"no judge shards for job {job} under {run_root}; pass --run-root")
    return found


def read_arm(run_root: pathlib.Path, arm: dict[str, object]) -> tuple[list[list], list[list]]:
    name = arm_name(arm)
    tag = [name, arm["model"], arm["language"], int(bool(arm["skills"])), arm["job"]]
    calls: list[list] = []
    submissions: list[list] = []
    for shard in shards(run_root, int(arm["job"])):
        con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        calls += [
            tag + list(row)
            for row in con.execute("select benchmark, route, status, correct, tokens, speedup, compiler from calls")
        ]
        submissions += [
            tag + list(row) for row in con.execute(
                "select benchmark, preset, baseline_ns, native_ns, speedup, suspect from submissions")
        ]
        con.close()
    return calls, submissions


def summarise(calls: list[list], submissions: list[list]) -> list[dict[str, object]]:
    by_arm: dict[str, dict[str, object]] = collections.OrderedDict()
    for row in calls:
        rec = dict(zip(CALL_COLUMNS, row, strict=True))
        arm = by_arm.setdefault(
            str(rec["arm"]), {
                "arm": rec["arm"],
                "model": rec["model"],
                "language": rec["language"],
                "skills": rec["skills"],
                "job": rec["job"],
                "calls": 0,
                "tokens": 0,
                "attempted": set(),
                "solved": set(),
                "submits": 0,
                "submits_ok": 0
            })
        arm["calls"] = int(arm["calls"]) + 1
        arm["tokens"] = int(arm["tokens"]) + int(rec["tokens"] or 0)
        arm["attempted"].add(rec["benchmark"])  # type: ignore[union-attr]
        if rec["route"] == "submit":
            arm["submits"] = int(arm["submits"]) + 1
            if rec["status"] == "ok":
                arm["submits_ok"] = int(arm["submits_ok"]) + 1
                arm["solved"].add(rec["benchmark"])  # type: ignore[union-attr]

    speedups: dict[str, list[float]] = collections.defaultdict(list)
    for row in submissions:
        rec = dict(zip(SUBMISSION_COLUMNS, row, strict=True))
        if not rec["suspect"] and rec["speedup"]:
            speedups[str(rec["arm"])].append(float(rec["speedup"]))

    out: list[dict[str, object]] = []
    for name, arm in by_arm.items():
        values = sorted(speedups[name])
        solved = len(arm["solved"])  # type: ignore[arg-type]
        out.append({
            "arm": name,
            "model": arm["model"],
            "language": arm["language"],
            "skills": arm["skills"],
            "job": arm["job"],
            "problems": PROBLEM_COUNT,
            "attempted": len(arm["attempted"]),  # type: ignore[arg-type]
            "solved": solved,
            "success_rate": round(solved / PROBLEM_COUNT, 4),
            # Two denominators, because they answer different questions and can disagree.
            # solved/242 is yield at a fixed budget; solved/attempted is per-problem capability.
            # For oss120b C they point opposite ways: the skills packet costs ~33% more tokens per
            # call, so that arm reached 130 problems where its pair reached 192 on the same token
            # spend. Reporting only solved/242 would read as "skills hurt" when the arm simply got
            # through less of the set.
            "success_rate_attempted": round(solved / len(arm["attempted"]), 4) if arm["attempted"] else 0.0,
            "calls": arm["calls"],
            "submits": arm["submits"],
            "submits_ok": arm["submits_ok"],
            "tokens": arm["tokens"],
            "tokens_per_solved": int(int(arm["tokens"]) / solved) if solved else 0,
            "speedup_n": len(values),
            "speedup_median": round(values[len(values) // 2], 4) if values else 0.0,
            "speedup_mean": round(sum(values) / len(values), 4) if values else 0.0,
            "speedup_max": round(max(values), 4) if values else 0.0,
            # The median sits at 1.00 for most arms, so the headline number is how often an arm
            # found a real speedup at all rather than the middle of a mostly-flat distribution.
            "frac_speedup_gt_1_1": round(sum(1 for v in values if v > 1.1) / len(values), 4) if values else 0.0,
        })
    return out


def matched_pairs(calls: list[list], submissions: list[list]) -> list[dict[str, object]]:
    """Per (model, language), compare the two arms only on kernels BOTH of them reached.

    Arms covered different amounts of the set, so solved/242 mixes capability with throughput and
    solved/attempted compares two different problem sets that need not be equally hard. The
    intersection is the only comparison where both sides face the same kernels.
    """
    reached: dict[str, set[str]] = collections.defaultdict(set)
    solved: dict[str, set[str]] = collections.defaultdict(set)
    meta: dict[str, tuple[str, str, str]] = {}
    for row in calls:
        rec = dict(zip(CALL_COLUMNS, row, strict=True))
        arm = str(rec["arm"])
        meta[arm] = (str(rec["model"]), str(rec["language"]), str(rec["skills"]))
        reached[arm].add(str(rec["benchmark"]))
        if rec["route"] == "submit" and rec["status"] == "ok":
            solved[arm].add(str(rec["benchmark"]))

    # fast_p: fraction of problems solved AND reaching speedup >= p. At p=1 it is the success
    # rate, so one metric carries correctness and performance together and cannot be gamed by
    # returning a correct but unoptimised kernel.
    best: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in submissions:
        rec = dict(zip(SUBMISSION_COLUMNS, row, strict=True))
        if rec["speedup"] and not rec["suspect"]:
            arm, bench = str(rec["arm"]), str(rec["benchmark"])
            best[arm][bench] = max(best[arm].get(bench, 0.0), float(rec["speedup"]))

    cells: dict[tuple[str, str], dict[str, str]] = collections.defaultdict(dict)
    for arm, (model, language, skills) in meta.items():
        cells[(model, language)][skills] = arm

    out: list[dict[str, object]] = []
    for (model, language), arms in cells.items():
        if "0" not in arms or "1" not in arms:
            continue
        off, on = arms["0"], arms["1"]
        common = reached[off] & reached[on]
        if not common:
            continue
        off_solved = len(solved[off] & common)
        on_solved = len(solved[on] & common)
        # Same kernels, two conditions, binary outcome: the discordant pairs are the evidence, so
        # McNemar's exact test rather than a two-sample proportion test on 84-217 problems.
        only_off = len((solved[off] & common) - solved[on])
        only_on = len((solved[on] & common) - solved[off])
        n_disc = only_off + only_on
        p_value = (min(1.0, 2.0 * sum(math.comb(n_disc, k)
                                      for k in range(min(only_off, only_on) + 1)) / 2**n_disc) if n_disc else 1.0)
        out.append({
            "model": model,
            "language": language,
            "arm_off": off,
            "arm_skills": on,
            "reached_off": len(reached[off]),
            "reached_skills": len(reached[on]),
            "common": len(common),
            "solved_off": off_solved,
            "solved_skills": on_solved,
            "rate_off": round(off_solved / len(common), 4),
            "rate_skills": round(on_solved / len(common), 4),
            "delta_pp": round(100.0 * (on_solved - off_solved) / len(common), 2),
            "only_off": only_off,
            "only_skills": only_on,
            "discordant": n_disc,
            "mcnemar_p": round(p_value, 4),
            **{f"fast{p:g}_{side}": round(
                sum(1 for b in common if best[arm].get(b, 0.0) >= p) / len(common), 4)
               for p in (1.0, 2.0, 4.0) for side, arm in (("off", off), ("skills", on))},
        })
    return out


def write_csv(path: pathlib.Path, columns: list[str], rows: list) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"  {path}  rows={len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",
                        type=pathlib.Path,
                        default=pathlib.Path("/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_calls: list[list] = []
    all_submissions: list[list] = []
    for arm in ARMS:
        calls, submissions = read_arm(args.run_root, arm)
        print(f"{arm_name(arm)}: calls={len(calls)} submissions={len(submissions)}")
        all_calls += calls
        all_submissions += submissions

    write_csv(args.out / "calls.csv", CALL_COLUMNS, all_calls)
    write_csv(args.out / "submissions.csv", SUBMISSION_COLUMNS, all_submissions)

    matched = matched_pairs(all_calls, all_submissions)
    if matched:
        columns = list(matched[0])
        write_csv(args.out / "matched.csv", columns, [[r[c] for c in columns] for r in matched])

    summary = summarise(all_calls, all_submissions)
    columns = list(summary[0])
    write_csv(args.out / "summary.csv", columns, [[row[c] for c in columns] for row in summary])
    return 0


if __name__ == "__main__":
    sys.exit(main())
