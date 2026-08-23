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
#
# Every arm before llr6v10 was dropped from this list on purpose, and none of them is comparable
# to what is here. Their judge step never asked for a width, and Slurm's default for a step that
# does not is ONE core: measured on this machine as `cpus_allowed=0,96`, a single physical core
# plus its SMT sibling, on a node with 96 of them. A threaded submission cannot outrun a serial
# one on one core, and a race on the parallelised axis has too few threads to show itself, so
# those arms measured serial optimisation and scored some racy code as correct. The fixed step
# measures `cpus_allowed=0-23`, 24 physical cores.
# They also timed the call itself: a submission that started asynchronous work and returned was
# charged for none of it, and its outputs were read while they were still being written. Both were
# fixed before this campaign (optarena ca8d9514). Git history holds the dropped entries.
#
# 605434-37 were cancelled mid-run and replaced by 605458-61, for a defect of the same shape one
# layer up: role_srun asked for --cpus-per-task on the JUDGE step only, so the INFERENCE and AGENT
# steps also took Slurm's one-core default. Every vLLM worker came up pinned to `0,96` -- and with
# four workers to a node they shared it, three of the four running on a socket that does not own
# their GPU's memory. The agent node ran ~240 processes there: 605434 recorded load average 61.9
# at 1.5% node utilisation. Fixed in optarena 9baa70d7; the workers now take a socket each
# (`0-23,96-119` ... `72-95,168-191`). Nothing measured before that fix is comparable to what
# follows it, throughput and submission counts least of all.
ARMS: list[dict[str, object]] = [
    {
        "campaign": "llr6v10",
        "job": 605458,
        "model": "qwen30b",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr6v10",
        "job": 605459,
        "model": "qwen30b",
        "language": "c",
        "skills": True
    },
    {
        "campaign": "llr6v10",
        "job": 605460,
        "model": "oss120b",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr6v10",
        "job": 605461,
        "model": "oss120b",
        "language": "c",
        "skills": True
    },
]

# Success is reported against the kernel set the arm DREW FROM, not against however many it
# managed to reach. llr6v10 draws from the llr-focus40 tag, 40 kernels sampled three times each.
PROBLEM_COUNT = {"llr6v10": 40}

CALL_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "route", "status", "correct", "tokens", "speedup",
    "compiler"
]
SUBMISSION_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "preset", "baseline_ns", "native_ns", "speedup", "suspect"
]


def arm_name(arm: dict[str, object]) -> str:
    suffix = "-skills" if arm["skills"] else ""
    return f"{arm.get('campaign', 'llr4')}-{arm['model']}-{arm['language']}{suffix}"


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
            str(rec["arm"]),
            {
                "arm": rec["arm"],
                "model": rec["model"],
                "language": rec["language"],
                "skills": rec["skills"],
                "job": rec["job"],
                "calls": 0,
                # Per-kernel HIGH WATER, not a running sum: the tokens field on a judge call is the
                # agent's CUMULATIVE usage at that moment, so adding the calls up counts the same
                # tokens once per call. The last call on a kernel is what that kernel cost.
                "spend": collections.defaultdict(int),
                "attempted": set(),
                "solved": set(),
                "submits": 0,
                "submits_ok": 0
            })
        arm["calls"] = int(arm["calls"]) + 1
        spend = arm["spend"]
        key = str(rec["benchmark"])
        spend[key] = max(spend[key], int(rec["tokens"] or 0))  # type: ignore[index]
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
        reached = len(arm["attempted"])  # type: ignore[arg-type]
        tokens = sum(arm["spend"].values())  # type: ignore[union-attr]
        out.append({
            "arm": name,
            "model": arm["model"],
            "language": arm["language"],
            "skills": arm["skills"],
            "job": arm["job"],
            "problems": PROBLEM_COUNT[name.split("-", 1)[0]],
            "attempted": reached,
            "solved": solved,
            "success_rate": round(solved / PROBLEM_COUNT[name.split("-", 1)[0]], 4),
            # Two denominators, because they answer different questions and can disagree.
            # solved/242 is yield at a fixed budget; solved/attempted is per-problem capability.
            # For oss120b C they point opposite ways: the skills packet costs ~33% more tokens per
            # call, so that arm reached 130 problems where its pair reached 192 on the same token
            # spend. Reporting only solved/242 would read as "skills hurt" when the arm simply got
            # through less of the set.
            "success_rate_attempted": round(solved / reached, 4) if reached else 0.0,
            "calls": arm["calls"],
            "submits": arm["submits"],
            "submits_ok": arm["submits_ok"],
            "tokens": tokens,
            # The budget question: what one kernel costs an arm from first turn to last grade. This
            # is the term the skills packet moves, and it is why the two success denominators
            # disagree -- a dearer kernel means fewer kernels reached at a fixed budget.
            "tokens_per_kernel": int(tokens / reached) if reached else 0,
            "tokens_per_solved": int(tokens / solved) if solved else 0,
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
            **{
                f"fast{p:g}_{side}": round(sum(1 for b in common if best[arm].get(b, 0.0) >= p) / len(common), 4)
                for p in (1.0, 2.0, 4.0)
                for side, arm in (("off", off), ("skills", on))
            },
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
    # Campaigns draw from DIFFERENT kernel sets, so pooling them into one figure invents a
    # distribution nothing was measured on: llr4 ran the 242-kernel track, llr6 the 40-kernel
    # llr-focus40 tag. Collect them into separate data directories and plot each on its own.
    parser.add_argument("--campaign", default="", help="only arms of this campaign (llr4, llr6)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_calls: list[list] = []
    all_submissions: list[list] = []
    selected = [a for a in ARMS if not args.campaign or a.get("campaign", "llr4") == args.campaign]
    if not selected:
        raise SystemExit(f"no arms for campaign {args.campaign!r}")
    for arm in selected:
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
