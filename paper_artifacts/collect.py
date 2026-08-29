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

import plot_llr8w2
from analysis import constructs

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
    # llr8: the same focus40 tag at --repeat 1. The llr6v10 arms drew each kernel three times,
    # which is agent multiplicity rather than sampling -- it tripled inference, agent and judge
    # load for coverage the grader already provides -- so an llr8 arm is 40 agents, one per kernel.
    {
        "campaign": "llr8",
        "job": 608446,
        "model": "qwen30b",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr8",
        "job": 608447,
        "model": "oss120b",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr8",
        "job": 608448,
        "model": "kimi27sglang",
        "language": "c",
        "skills": False,
        "batch": "a",
        "problems": 10
    },
    {
        "campaign": "llr8",
        "job": 608449,
        "model": "kimi27sglang",
        "language": "c",
        "skills": False,
        "batch": "b",
        "problems": 10
    },
    {
        "campaign": "llr8",
        "job": 608987,
        "model": "oss120b",
        "language": "c",
        "skills": True
    },
    {
        "campaign": "llr8",
        "job": 608988,
        "model": "qwen30b",
        "language": "c",
        "skills": True
    },
    # llr8w2: the C-vs-Fortran factorial. Wave 1's qwen arms were void (the qwen3_coder tool parser
    # ate turn-1 tool calls) and its C references were TSVC-shaped; both were fixed before this wave,
    # so these arms are not comparable to the llr8 rows above and carry their own campaign name.
    # The kimi arm is a 10-kernel batch like its wave-1 siblings, and its unskilled twin failed, so
    # it has no pair and cannot appear in the matched table.
    {
        "campaign": "llr8w2",
        "job": 610669,
        "model": "qwen38",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr8w2",
        "job": 610671,
        "model": "qwen38",
        "language": "c",
        "skills": True
    },
    {
        "campaign": "llr8w2",
        "job": 610670,
        "model": "qwen38",
        "language": "fortran",
        "skills": False
    },
    {
        "campaign": "llr8w2",
        "job": 610672,
        "model": "qwen38",
        "language": "fortran",
        "skills": True
    },
    {
        "campaign": "llr8w2",
        "job": 610668,
        "model": "oss120b",
        "language": "fortran",
        "skills": False
    },
    {
        "campaign": "llr8w2",
        "job": 610653,
        "model": "oss120b",
        "language": "fortran",
        "skills": True
    },
    {
        "campaign": "llr8w2",
        "job": 610662,
        "model": "kimi27sglang",
        "language": "c",
        "skills": True,
        "batch": "r1",
        "problems": 10
    },
    # llr8w3: a COMPLETION wave. Four arms re-run only the kernels their wave-2 twin never
    # submitted, so their denominator is the hand-written list, not 40 -- reporting them out of 40
    # would charge them for kernels they were never given. Coverage is the UNION with wave 2; these
    # rows are not an independent sample of the 40. The other four are full 40-kernel arms for the
    # two models wave 2 never ran in C.
    {
        "campaign": "llr8w3",
        "job": 611560,
        "model": "qwen38",
        "language": "c",
        "skills": False,
        "problems": 8
    },
    {
        "campaign": "llr8w3",
        "job": 611561,
        "model": "qwen38",
        "language": "c",
        "skills": True,
        "problems": 6
    },
    {
        "campaign": "llr8w3",
        "job": 611562,
        "model": "qwen38",
        "language": "fortran",
        "skills": False,
        "problems": 7
    },
    {
        "campaign": "llr8w3",
        "job": 611563,
        "model": "qwen38",
        "language": "fortran",
        "skills": True,
        "problems": 5
    },
    {
        "campaign": "llr8w3",
        "job": 611564,
        "model": "oss120b",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr8w3",
        "job": 611565,
        "model": "oss120b",
        "language": "c",
        "skills": True
    },
    {
        "campaign": "llr8w3",
        "job": 611566,
        "model": "kimi27sglang",
        "language": "c",
        "skills": False
    },
    {
        "campaign": "llr8w3",
        "job": 611567,
        "model": "kimi27sglang",
        "language": "c",
        "skills": True
    },
    # llr8w4: the second COMPLETION wave, and the first whose lists were COMPUTED rather than
    # written by hand (make_gap_kernels.py). Each arm re-ran exactly the kernels its wave-2/3
    # twin had never produced a scored submission for, so the denominator is that gap, not 40.
    # Wave 5 re-ran the Fortran arms' residue into this same run directory under the same
    # CAMPAIGN_ARM, which is why those jobs are absent here: they are the same arm continuing.
    {
        "campaign": "llr8w4",
        "job": 612042,
        "model": "qwen38",
        "language": "c",
        "skills": False,
        "problems": 13
    },
    {
        "campaign": "llr8w4",
        "job": 612043,
        "model": "qwen38",
        "language": "c",
        "skills": True,
        "problems": 9
    },
    {
        "campaign": "llr8w4",
        "job": 612044,
        "model": "qwen38",
        "language": "fortran",
        "skills": False,
        "problems": 15
    },
    {
        "campaign": "llr8w4",
        "job": 612045,
        "model": "qwen38",
        "language": "fortran",
        "skills": True,
        "problems": 11
    },
    {
        "campaign": "llr8w4",
        "job": 612046,
        "model": "oss120b",
        "language": "c",
        "skills": False,
        "problems": 9
    },
    {
        "campaign": "llr8w4",
        "job": 612047,
        "model": "oss120b",
        "language": "c",
        "skills": True,
        "problems": 10
    },
    {
        "campaign": "llr8w4",
        "job": 612048,
        "model": "oss120b",
        "language": "fortran",
        "skills": False,
        "problems": 15
    },
    {
        "campaign": "llr8w4",
        "job": 612049,
        "model": "oss120b",
        "language": "fortran",
        "skills": True,
        "problems": 16
    },
    {
        "campaign": "llr8w4",
        "job": 612050,
        "model": "kimi27sglang",
        "language": "c",
        "skills": False,
        "problems": 25
    },
    {
        "campaign": "llr8w4",
        "job": 612051,
        "model": "kimi27sglang",
        "language": "c",
        "skills": True,
        "problems": 25
    },
    # llr8w6: the C completion wave. The two oss120b arms were cut short 13 minutes in, after each
    # had scored 5 of its 7 kernels; their shards hold that work and the rest moved to a later wave
    # that runs under the prompt fix (no -ffast-math in the agents' own local builds).
    {
        "campaign": "llr8w6",
        "job": 612293,
        "model": "oss120b",
        "language": "c",
        "skills": False,
        "problems": 7
    },
    {
        "campaign": "llr8w6",
        "job": 612294,
        "model": "oss120b",
        "language": "c",
        "skills": True,
        "problems": 7
    },
]

# Success is reported against the kernel set the arm DREW FROM, not against however many it
# managed to reach. llr6v10 draws from the llr-focus40 tag, 40 kernels sampled three times each.
#: Kernels each campaign DREW FROM. llr8w2 points at problems-llr6-{c,fortran}.jsonl, which
#: holds 120 entries, but the launcher dispatches 40 of them per arm -- the highest problem
#: index in every wave-2 run_id is p39, and the two oss120b arms dispatched all 40. The POOL
#: is not the denominator; taking it as one reported a 55% success rate as 18%.
PROBLEM_COUNT = {"llr6v10": 40, "llr8": 40, "llr8w2": 40, "llr8w3": 40, "llr8w4": 40, "llr8w6": 40}

CALL_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "route", "status", "correct", "tokens", "speedup",
    "compiler"
]
SUBMISSION_COLUMNS = [
    "arm", "model", "language", "skills", "job", "benchmark", "preset", "baseline_ns", "native_ns", "speedup", "suspect"
]


def discover_arms(run_root: pathlib.Path, campaign: str) -> list[dict[str, object]]:
    """Build the arm table by READING the runs, instead of requiring one to be typed in.

    The launcher already stamps every row with ``run_id = <arm>.n<node>.p<problem>.w<worker>``, and
    the arm name carries the model, the language and the skills leg. So the registry above is a
    second copy of something the data already says, and keeping the two in step by hand is the step
    that gets skipped -- llr6 and llr8 were both collected late for exactly that reason.

    Discovery covers a run whose shards exist; :data:`ARMS` stays as the record of which jobs BELONG
    to a campaign, including ones that produced nothing, because "this arm ran and returned no rows"
    and "this arm was never in the campaign" are different claims and only the registry can tell
    them apart. Use ``--discover`` for a quick look at a directory, the registry for a paper number.
    """
    found: dict[str, tuple[int, dict[str, object]]] = {}
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        shard_paths = sorted(run_dir.glob("judge/rank-*/hpcagent_bench*.db"))
        if not shard_paths:
            continue
        names: set[str] = set()
        for shard in shard_paths:
            con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
            try:
                names |= {str(r[0]).split(".", 1)[0] for r in con.execute("select distinct run_id from calls") if r[0]}
            except sqlite3.DatabaseError:
                continue
            finally:
                con.close()
        for name in sorted(names):
            parts = name.split("-")
            if len(parts) < 3:
                continue  # not an arm run_id (a hand-run probe writes e.g. `adhoc`)
            skills = "skills" in parts
            batch = parts[-1] if parts[-1] not in ("skills", ) and parts[-1].startswith(("r", "a", "b")) and len(
                parts[-1]) <= 2 else ""
            core = [p for p in parts[1:] if p != "skills" and p != batch]
            if len(core) < 2:
                continue
            arm = {
                "campaign": campaign or parts[0],
                "job": int(run_dir.name),
                "model": "-".join(core[:-1]),
                "language": core[-1],
                "skills": skills,
            }
            if batch:
                arm["batch"] = batch
            # An arm that was submitted, failed and resubmitted appears under SEVERAL job ids, and
            # the dead attempts carry rows. Keep the job with the most calls and say which ones were
            # dropped: silently merging them would pool two different runs into one arm, and
            # silently keeping all of them would report the same arm several times.
            weight = sum(
                con.execute("select count(*) from calls where run_id like ?", (f"{name}.%", )).fetchone()[0]
                for con in [sqlite3.connect(f"file:{sh}?mode=ro", uri=True) for sh in shard_paths])
            previous = found.get(name)
            if previous is None or weight > previous[0]:
                if previous is not None:
                    print(f"{name}: job {previous[1]['job']} superseded by {arm['job']} "
                          f"({previous[0]} -> {weight} calls)",
                          file=sys.stderr)
                found[name] = (weight, arm)
            else:
                print(f"{name}: job {arm['job']} has fewer calls than {previous[1]['job']}, skipped", file=sys.stderr)
    return sorted((a for _, a in found.values()),
                  key=lambda a: (str(a["model"]), str(a["language"]), bool(a["skills"])))


def arm_run_pattern(arm: dict[str, object]) -> str:
    """A SQL LIKE pattern matching the ``run_id`` values this arm wrote.

    Anchored on the arm identity and the dot that ends it, not on the campaign token: the launcher
    writes the campaign name it was given (``llr8-qwen38-c`` even for a wave-2 arm this file calls
    ``llr8w2-qwen38-c``), so matching the wave-qualified name selects nothing and reports every arm
    as empty. The trailing dot is what keeps ``...-c.`` from also matching ``...-c-skills.``.
    """
    suffix = "-skills" if arm["skills"] else ""
    batch = f"-{arm['batch']}" if arm.get("batch") else ""
    return f"%{arm['model']}-{arm['language']}{suffix}{batch}.%"


def arm_name(arm: dict[str, object]) -> str:
    """``<campaign>-<model>-<language>[-skills][-<batch>]``.

    The batch letter is not decoration: a kimi arm is one BATCH of the tag, and two batches of the
    same model, language and leg would otherwise key to one name and have their rows merged into a
    single arm scored against a denominator neither of them drew from.
    """
    suffix = "-skills" if arm["skills"] else ""
    batch = f"-{arm['batch']}" if arm.get("batch") else ""
    return f"{arm.get('campaign', 'llr4')}-{arm['model']}-{arm['language']}{suffix}{batch}"


#: Arms that drew from a BATCH of their tag rather than the whole of it, and so carry their own
#: success denominator: 10 solved of the 10 a kimi batch drew is not 10 of 40.
BATCH_COUNT = {arm_name(a): int(a["problems"]) for a in ARMS if a.get("problems")}


def problem_count(name: str) -> int:
    """How many kernels the arm DREW FROM -- the denominator both success rates are taken over."""
    return BATCH_COUNT.get(name) or PROBLEM_COUNT[name.split("-", 1)[0]]


def shards(run_root: pathlib.Path, job: int) -> list[str]:
    """The job's judge shards, or an empty list when the job has not run yet.

    The two empty cases are NOT the same and must not be handled the same way. A run directory that
    does not exist is an arm still queued -- the registry names every arm of a campaign, including
    ones whose results are still coming -- and collecting the rest is the right answer. A run
    directory that exists with no shards under it means the root is wrong or the run broke, and
    silently reporting zero for it would put a fabricated row in every figure downstream.
    """
    run = run_root / str(job)
    found = sorted(glob.glob(str(run / "judge" / "rank-*" / "*.db")))
    if not found and run.is_dir():
        raise SystemExit(f"no judge shards under {run}; the run is there but its judge wrote nothing")
    return found


def read_arm(run_root: pathlib.Path, arm: dict[str, object]) -> tuple[list[list], list[list]]:
    name = arm_name(arm)
    tag = [name, arm["model"], arm["language"], int(bool(arm["skills"])), arm["job"]]
    calls: list[list] = []
    submissions: list[list] = []
    found = shards(run_root, int(arm["job"]))
    if not found:
        print(f"{name}: job {arm['job']} has not run yet, skipped", file=sys.stderr)
        return [], []
    # A shard is read for the rows THIS ARM wrote, not for everything in the file. The launcher
    # stamps every campaign row with `run_id = <arm>.n<node>.p<problem>.w<worker>`; a row carrying
    # anything else (`adhoc`, or a null) came from a hand-run probe that happened to share the
    # results directory, and attributing it here credits the arm with a measurement it never made.
    # Measured: seven such rows sat in the llr8w2 shards, one of them a 24x outlier.
    row_filter = "where run_id like ?"
    prefix = arm_run_pattern(arm)
    for shard in found:
        con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        calls += [
            tag + list(row) for row in con.execute(
                f"select benchmark, route, status, correct, tokens, speedup, compiler from calls {row_filter}", (
                    prefix, ))
        ]
        submissions += [
            tag + list(row) for row in con.execute(
                "select benchmark, preset, baseline_ns, native_ns, speedup, suspect "
                f"from submissions {row_filter}", (prefix, ))
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
            "problems": problem_count(name),
            "attempted": reached,
            "solved": solved,
            "success_rate": round(solved / problem_count(name), 4),
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
            # The HEADLINE. Speed-up is a ratio, so the arm-level figure is the geometric mean: it
            # is the ratio whose product over the set matches, and it is symmetric in speed-up and
            # slowdown (2x and 0.5x cancel to 1). The arithmetic mean is dragged by one 50x kernel
            # past anything the arm does normally, and the median throws away the size of every win
            # -- both are kept below as spread cues, neither is the number to quote.
            "speedup_geomean": geomean(values),
            "speedup_median": round(values[len(values) // 2], 4) if values else 0.0,
            "speedup_mean": round(sum(values) / len(values), 4) if values else 0.0,
            "speedup_max": round(max(values), 4) if values else 0.0,
            # The median sits at 1.00 for most arms, so the headline number is how often an arm
            # found a real speedup at all rather than the middle of a mostly-flat distribution.
            "frac_speedup_gt_1_1": round(sum(1 for v in values if v > 1.1) / len(values), 4) if values else 0.0,
        })
    return out


def geomean(values: list[float]) -> float:
    """Geometric mean of a speed-up set; ``0.0`` when it is empty.

    Non-positive entries are dropped rather than clamped: a speed-up at or below zero is a missing
    measurement, not a slow one, and an epsilon would drag the geomean toward zero and read as a
    collapse that never happened.
    """
    usable = [v for v in values if v > 0]
    if not usable:
        return 0.0
    return round(math.exp(sum(math.log(v) for v in usable) / len(usable)), 4)


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
            **paired_speedup(common, best[off], best[on]),
        })
    return out


def paired_speedup(common: set[str], off: dict[str, float], on: dict[str, float]) -> dict[str, object]:
    """Geomean speed-up of each side, and their ratio, over the kernels BOTH sides actually timed.

    The arm-level geomean in ``summary.csv`` is taken over whatever each arm reached, and the two
    arms reached different kernels -- so a difference between those two numbers is partly a
    difference in which problems were attempted. This is the paired form: only kernels with a
    timed, non-suspect submission on BOTH sides, so the ratio is the skills effect and nothing else.

    ``ratio`` is the geomean of the PER-KERNEL ratios, which is the same as the ratio of the two
    geomeans; the sign test is on those per-kernel ratios, and it is what says whether the direction
    is real rather than one kernel carrying the arm.
    """
    both = sorted(b for b in common if off.get(b, 0.0) > 0 and on.get(b, 0.0) > 0)
    if not both:
        # Every key, always: a caller formatting this row must not have to know which branch ran,
        # and a missing key here crashed the figure for a pair with no commonly-timed kernel.
        return {
            "paired_n": 0,
            "paired_geo_off": 0.0,
            "paired_geo_skills": 0.0,
            "paired_ratio": 0.0,
            "sign_wins": 0,
            "sign_losses": 0,
            "sign_p": 1.0,
        }
    ratios = [on[b] / off[b] for b in both]
    wins = sum(1 for r in ratios if r > 1.0)
    losses = sum(1 for r in ratios if r < 1.0)
    n = wins + losses
    p_value = (min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0)
    return {
        "paired_n": len(both),
        "paired_geo_off": round(math.exp(sum(math.log(off[b]) for b in both) / len(both)), 4),
        "paired_geo_skills": round(math.exp(sum(math.log(on[b]) for b in both) / len(both)), 4),
        "paired_ratio": round(math.exp(sum(math.log(r) for r in ratios) / len(ratios)), 4),
        "sign_wins": wins,
        "sign_losses": losses,
        "sign_p": round(p_value, 4),
    }


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
    parser.add_argument("--discover",
                        action="store_true",
                        help="read the arms out of the run directories instead of the ARMS registry")
    parser.add_argument("--plot", action="store_true", help="also render the figures for this campaign")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_calls: list[list] = []
    all_submissions: list[list] = []
    if args.discover:
        selected = discover_arms(args.run_root, args.campaign)
        print(f"discovered {len(selected)} arms under {args.run_root}")
    else:
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

    # The construct census reads the SAME shards this pass just opened, so it belongs on the same
    # command rather than as a second thing to remember; a campaign collected without it is how the
    # "what did the agents actually write" table went missing for two waves.
    constructs.write_census(args.run_root, selected, args.out)
    if args.plot:
        plot_llr8w2.render(args.out, args.out.parent / f"figures-{args.out.name.removeprefix('data-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
