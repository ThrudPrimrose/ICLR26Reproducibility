"""Extract the git-scicomp repo-vs-kernel A/B into a reproducibility artifact.

The experiment asks whether framing a task as a git repository with an ISSUE.md makes a coding
agent do better or worse than handing it a bare kernel name. Four arms -- two models crossed with
two framings -- over the same ten scientific-computing kernels, three independent agents each.

This script reads the per-run judge databases the campaign left under its run roots and writes one
folder per arm plus a combined CSV. Every database is opened READ-ONLY (``mode=ro``): the run roots
are the only copy of the campaign and re-running a reader must never be able to damage them.

The unit of the artifact is a CELL -- one (arm, kernel, attempt) triple. The launcher gives each
cell its own worker, so the cell grid is fixed by the problem list at 10 kernels x 3 attempts per
arm and exists whether or not the agent in it ever reached the judge. A cell with no database rows
is written out as a row saying so, never dropped, because an arm that silently loses a kernel
becomes a figure with a bar missing and nothing to say why.

Which grade counts, per the campaign owner: if a cell has SEVERAL submissions the LAST one is the
result; if it has NONE, the last state left in the agent workspace is what ``submit`` would have
taken. Both sources are recorded with a provenance column rather than blended, so a reader can
always tell a graded text from a recovered one.

Two provenance columns carry the honesty of the artifact:

``original_provenance``   task_repo_seed      the repo arm's seed, read from the seed COMMIT
                          task_repo_worktree  the checked-out seed, used only when there is no
                                              commit to read; may carry agent edits
                          task_reference      the kernel arm's ``<kernel>_reference.*``, whose
                                              suffix varies: .c for six kernels, .py for
                                              mandelbrot1, .cu for spgemm_hash
                          missing             no reference was served (laplacian_stencil_3d), or
                                              the run kept none
``submitted_provenance``  graded_submission  the exact text of the LAST graded submission
                          last_graded        the last text the judge graded on any route
                          last_saved         the last file in the agent workspace, NOT necessarily
                                             what was submitted
                          missing            nothing is recoverable

Re-running over unchanged inputs reproduces byte-identical output: every glob is sorted, nothing
samples or stamps a wall-clock time.

    python3 collect_git.py [--runs GLOB]... [--problems JSONL] [--out DIR]
"""

import argparse
import collections
import csv
import datetime
import functools
import hashlib
import json
import math
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

#: Terminal graded rows. ``submissions`` holds the accepted submits and ``attempts`` the rejected
#: ones; together they are exactly the ``route='submit'`` calls, so neither alone is the outcome.
TERMINAL_TABLES = ("submissions", "attempts")

#: Pseudo run id the harness writes for a grade it could not tie to a campaign worker. The job
#: directory still identifies the ARM, but nothing identifies the cell, so these rows are reported
#: separately instead of being credited to a kernel they may not belong to.
ADHOC_RUN_ID = "adhoc"

#: ``<arm>.n<node>.p<problem>.w<worker>`` -- the launcher stamps it on every campaign row.
RUN_ID_RE = re.compile(r"^(?P<arm>.+)\.n(?P<node>\d+)\.p(?P<problem>\d+)\.w(?P<worker>\d+)$")

#: Campaign prefix every arm name carries; stripped to name the arm's output folder.
ARM_PREFIX = "git-scicomp-"

RESULT_FIELDS = ("experiment", "arm", "arm_dir", "model", "framing", "kernel", "level", "dwarf", "attempt",
                 "worker_index", "problem_index", "run_id", "attribution", "job", "run_root", "db_paths", "verdict",
                 "correct", "submitted", "reason", "speedup", "baseline_ns", "native_ns", "runtime_s", "baseline_s",
                 "suspect", "preset", "datatype", "language", "baseline", "compiler", "execution", "cpu",
                 "best_score_speedup", "n_calls", "n_submit_calls", "n_submissions", "n_failed_submits", "turns",
                 "tokens", "ts_first_ms", "ts_last_ms", "ts_first_iso", "ts_last_iso", "wall_clock_s",
                 "off_task_benchmarks", "original_path", "original_sha256", "original_provenance", "original_origin",
                 "submitted_path", "submitted_sha256", "submitted_provenance")

SUMMARY_FIELDS = ("arm", "arm_dir", "model", "framing", "job", "cells", "cells_with_data", "cells_submitted",
                  "cells_accepted", "kernels_expected", "kernels_with_data", "kernels_missing", "correct_rate_of_cells",
                  "correct_rate_of_cells_with_data", "geomean_speedup_accepted", "geomean_speedup_best_score",
                  "total_tokens", "adhoc_rows", "off_task_rows")


class Problem(NamedTuple):
    """One cell of the grid, as the launcher defined it before any agent ran."""
    problem_index: int
    kernel: str
    dwarf: str
    manifest: str
    attempt: int


class Job(NamedTuple):
    """One arm's run directory and the labels every row it yields is stamped with."""
    run_root: pathlib.Path
    job: str
    arm: str
    model: str
    framing: str
    shards: tuple[pathlib.Path, ...]


class Cell(NamedTuple):
    """Every database row belonging to one (arm, kernel, attempt), already split by table."""
    calls: list[dict[str, Any]]
    submissions: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    sources: list[dict[str, Any]]


def parse_args(argv: list[str]) -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve()
    # <scratch>/ICLR26Reproducibility/paper_artifacts/experiments/git/collect_git.py -- the sibling
    # optarena checkout and the run roots are read from the same scratch tree.
    scratch = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs",
                        action="append",
                        default=None,
                        help="glob of run roots holding <job>/judge/ (repeatable)")
    parser.add_argument("--problems",
                        type=pathlib.Path,
                        default=scratch / "optarena/containers/cluster/example-script/problems-git-scicomp.jsonl",
                        help="launcher problem list; defines the cell grid")
    parser.add_argument("--benchmarks",
                        type=pathlib.Path,
                        default=scratch / "optarena/hpcagent_bench/benchmarks",
                        help="benchmark manifests, read for the level field")
    parser.add_argument("--out", type=pathlib.Path, default=here.parent, help="artifact directory to write")
    args = parser.parse_args(argv)
    if args.runs is None:
        args.runs = [str(scratch / "hpcagent-bench-runs" / "git-scicomp-*")]
    return args


def load_problems(path: pathlib.Path) -> list[Problem]:
    """The cell grid, from the launcher's own problem list rather than a list retyped here.

    The attempt number is the rank of a problem id among the ids sharing its kernel, which is what
    makes three independent agents on one kernel readable as attempt 0, 1 and 2.
    """
    raw: list[tuple[int, str, str, str]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            parts = str(entry["kernel"]).split("/")
            raw.append((int(entry["id"]), parts[-1], parts[1], str(entry["kernel"])))
    seen: collections.Counter[str] = collections.Counter()
    problems: list[Problem] = []
    for index, kernel, dwarf, manifest in sorted(raw):
        problems.append(Problem(index, kernel, dwarf, manifest, seen[kernel]))
        seen[kernel] += 1
    return problems


@functools.lru_cache(maxsize=None, typed=True)
def kernel_level(benchmarks: pathlib.Path, manifest: str) -> str:
    """The ``level`` field of a benchmark manifest, read as text.

    Read, never written: the manifests are correctness oracles. A regex rather than a YAML parse
    keeps the artifact runnable without a third-party import.
    """
    path = benchmarks / f"{manifest}.yaml"
    if not path.is_file():
        return ""
    for line in path.read_text().splitlines():
        found = re.match(r"^level:\s*(\S+)", line)
        if found:
            return found.group(1)
    return ""


def find_shards(run_root: pathlib.Path, job: str) -> tuple[pathlib.Path, ...]:
    return tuple(sorted((run_root / job).glob("judge/rank-*/*.db")))


def store_root(shard: pathlib.Path) -> pathlib.Path:
    """Where a shard keeps its content-addressed blobs; the ``path`` column is relative to it."""
    return shard.parent / f"{shard.stem}_prompts"


def read_rows(shard: pathlib.Path, table: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in con.execute(f"select * from {table}")]
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    for row in rows:
        row["db"] = str(shard)
    return rows


def arm_of_job(shards: tuple[pathlib.Path, ...]) -> str:
    """The arm a job ran, taken from the dotted prefix every campaign run id carries.

    Cross-checked rather than trusted: a job that yielded two arm prefixes would mean two campaigns
    shared a run directory, and every per-arm number after that point would be a blend of both.
    """
    arms = set()
    for shard in shards:
        for table in ("calls", *TERMINAL_TABLES):
            for row in read_rows(shard, table):
                found = RUN_ID_RE.match(str(row["run_id"]))
                if found:
                    arms.add(found.group("arm"))
    if len(arms) > 1:
        raise SystemExit(f"{shards[0].parents[2]}: rows from more than one arm: {sorted(arms)}")
    return arms.pop() if arms else ""


def discover_jobs(patterns: list[str]) -> list[Job]:
    """Every job directory under the run roots that graded something, labelled with its arm."""
    jobs: list[Job] = []
    roots: list[pathlib.Path] = []
    for pattern in patterns:
        roots += [p for p in sorted(pathlib.Path("/").glob(pattern.lstrip("/"))) if p.is_dir()]
    for run_root in roots:
        for run in sorted(p for p in run_root.iterdir() if p.is_dir()):
            shards = find_shards(run_root, run.name)
            if not shards:
                print(f"{run}: judge wrote no shards, skipped", file=sys.stderr)
                continue
            arm = arm_of_job(shards)
            if not arm.startswith(ARM_PREFIX):
                print(f"{run}: arm {arm!r} is not a git-scicomp arm, skipped", file=sys.stderr)
                continue
            tag = arm[len(ARM_PREFIX):]
            model, _, framing = tag.rpartition("-")
            jobs.append(Job(run_root, run.name, arm, model, framing, shards))
    return jobs


def collect_cells(job: Job) -> tuple[dict[int, Cell], list[dict[str, Any]]]:
    """Split one job's rows by worker index; return the unattributable ``adhoc`` rows beside them."""
    cells: dict[int, Cell] = collections.defaultdict(lambda: Cell([], [], [], []))
    adhoc: list[dict[str, Any]] = []
    for shard in job.shards:
        for table in ("calls", "submissions", "attempts", "sources"):
            for row in read_rows(shard, table):
                row["table"] = table
                found = RUN_ID_RE.match(str(row["run_id"]))
                if not found:
                    adhoc.append(row)
                    continue
                bucket = cells[int(found.group("worker"))]
                by_table = {
                    "calls": bucket.calls,
                    "submissions": bucket.submissions,
                    "attempts": bucket.attempts,
                    "sources": bucket.sources
                }
                by_table[table].append(row)
    return cells, adhoc


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_path(row: dict[str, Any]) -> pathlib.Path:
    return store_root(pathlib.Path(str(row["db"]))) / str(row["path"])


@functools.lru_cache(maxsize=None, typed=True)
def repo_seed(repo: pathlib.Path, rel: str) -> bytes | None:
    """The seed text as COMMITTED, or None when the repo has no such commit.

    Cached because the three agents on a kernel share one template, and each miss is a subprocess.
    """
    if not (repo / ".git").is_dir():
        return None
    done = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"], capture_output=True, check=False)
    return done.stdout if done.returncode == 0 else None


def original_source(job: Job, run_root: pathlib.Path, kernel: str) -> tuple[bytes | None, str, str, str]:
    """The starting point the agent was handed, which is not the same file in the two framings.

    The repo arm is seeded with a naive ``repo/src/<kernel>.c`` committed on ``main``; the kernel
    arm is shown ``<kernel>_reference.*``. Handing back one file for both would misreport half the
    experiment, so the framing picks the file.

    For the repo arm the seed is read from the COMMIT, never from the working tree. Agents can and
    did write into the shared task templates -- in the oss120b repo arm the spgemm_hash and
    mandelbrot1 templates are left with modified sources -- so the checked-out file is not reliably
    the text anyone started from, while ``HEAD`` still is.
    """
    task = run_root / job.job / "shared" / "tasks" / kernel
    if job.framing == "repo":
        repo = task / "repo"
        rel = f"src/{kernel}.c"
        seed = repo_seed(repo, rel)
        if seed is not None:
            return seed, "task_repo_seed", ".c", f"{repo}@HEAD:{rel}"
        worktree = repo / rel
        if worktree.is_file():
            return worktree.read_bytes(), "task_repo_worktree", ".c", str(worktree)
        return None, "missing", "", ""
    # The kernel arm's reference is NOT always C: mandelbrot1 ships a .py reference and
    # spgemm_hash a .cu one, and cg is served as a bare cg.c with no _reference infix. Assuming
    # one suffix silently reported four of the ten kernels as having no starting point at all.
    # Only a `_reference` file counts. A bare `<kernel>.c` sitting in a task directory is NOT a
    # served starting point: agents can write into the shared task tree, and the cg task dir picked
    # up an agent's cg.c and cg.o an hour after provisioning. Treating that as the original would
    # report an agent's own output back as the text it started from.
    references = sorted(p for p in task.glob(f"{kernel}_reference.*") if p.is_file())
    if not references:
        return None, "missing", "", ""
    return references[0].read_bytes(), "task_reference", references[0].suffix, str(references[0])


def workspace_source(job: Job, run_root: pathlib.Path, worker: int, kernel: str) -> pathlib.Path | None:
    """The last state of the kernel file in the agent's workspace -- what ``submit`` would take.

    Agents litter the workspace with probes, backups and renamed variants, so the search matches
    the kernel's exact filename and prefers a repo checkout's tracked ``src/`` copy over a loose
    one. Preference order is fixed and the glob is sorted, so the choice is reproducible.
    """
    agent = run_root / job.job / "shared" / f"agent-{worker}"
    if not agent.is_dir():
        return None
    hits = [p for p in sorted(agent.rglob(f"{kernel}.c")) if ".git" not in p.parts]
    if not hits:
        return None
    preferences: tuple[Callable[[pathlib.Path], bool], ...] = (
        lambda p: p.parent.name == "src" and p.parent.parent.parent == agent,
        lambda p: p.parent.name == "src",
        lambda p: p.parent == agent,
    )
    for wanted in preferences:
        for hit in hits:
            if wanted(hit):
                return hit
    return hits[0]


def geomean(values: list[float]) -> str:
    """Geometric mean, the only aggregate a ratio like speedup may be summarised with."""
    usable = [v for v in values if v and v > 0]
    if not usable:
        return ""
    return f"{math.exp(sum(math.log(v) for v in usable) / len(usable)):.6f}"


def iso(ms: Any) -> str:
    if not ms:
        return ""
    return datetime.datetime.fromtimestamp(int(ms) / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def num(value: Any) -> str:
    return "" if value is None else str(value)


def write_source(out: pathlib.Path, rel: str, data: bytes) -> str:
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


def cell_row(job: Job, problem: Problem, cell: Cell, benchmarks: pathlib.Path, out: pathlib.Path, arm_dir: str,
             model: str) -> dict[str, str]:
    """One artifact row: the cell's outcome, its cost, and the two sources saved beside it."""
    kernel = problem.kernel
    worker = problem.problem_index
    on_task = [r for r in cell.calls if r["benchmark"] == kernel]
    terminal = sorted([r for r in cell.submissions + cell.attempts if r["benchmark"] == kernel],
                      key=lambda r: (int(r["ts"]), int(r["id"])))
    last = terminal[-1] if terminal else None
    every = cell.calls + cell.submissions + cell.attempts
    stamps = [int(r["ts"]) for r in every]
    scores = [
        float(r["speedup"] or 0.0) for r in on_task
        if r.get("route") == "score" and r.get("status") == "ok" and r.get("correct")
    ]
    off_task = sorted({str(r["benchmark"]) for r in every if r["benchmark"] != kernel})

    accepted = bool(last is not None and last["table"] == "submissions")
    if last is None:
        verdict = "no_submission" if every else "no_data"
    else:
        verdict = "ok" if accepted else str(last.get("reason") or "rejected")

    original, original_from, original_suffix, original_origin = original_source(job, job.run_root, kernel)
    # Relative to the experiment directory, so `submitted_path` in results.csv is a path a reader
    # can follow from where the CSV sits.
    stem = f"artifacts/{arm_dir}/sources/{kernel}__a{problem.attempt}"
    original_rel = ""
    original_hash = ""
    if original is not None:
        original_rel = write_source(out, f"{stem}__original{original_suffix}", original)
        original_hash = sha256_of(original)

    # The LAST submission is the result. Its exact graded text is preferred; the workspace file is
    # the documented fallback for a cell that never submitted, and is labelled as such.
    candidate: bytes | None = None
    candidate_from = "missing"
    graded = {(int(r["ts"]), str(r["benchmark"])): r for r in cell.sources}
    if last is not None:
        blob = graded.get((int(last["ts"]), kernel))
        if blob is not None and blob_path(blob).is_file():
            candidate = blob_path(blob).read_bytes()
            candidate_from = "graded_submission"
    if candidate is None:
        on_kernel = sorted([r for r in cell.sources if r["benchmark"] == kernel], key=lambda r: (int(r["ts"]), r["id"]))
        if on_kernel and blob_path(on_kernel[-1]).is_file():
            candidate = blob_path(on_kernel[-1]).read_bytes()
            candidate_from = "last_graded"
    if candidate is None:
        saved = workspace_source(job, job.run_root, worker, kernel)
        if saved is not None:
            candidate = saved.read_bytes()
            candidate_from = "last_saved"
    candidate_rel = ""
    candidate_hash = ""
    if candidate is not None:
        candidate_rel = write_source(out, f"{stem}__submitted.c", candidate)
        candidate_hash = sha256_of(candidate)

    sub = last if accepted else None
    return {
        "experiment": "git-scicomp",
        "arm": job.arm,
        "arm_dir": arm_dir,
        "model": model,
        "framing": job.framing,
        "kernel": kernel,
        "level": kernel_level(benchmarks, problem.manifest),
        "dwarf": problem.dwarf,
        "attempt": str(problem.attempt),
        "worker_index": str(worker),
        "problem_index": str(problem.problem_index),
        "run_id": f"{job.arm}.n0.p{worker}.w{worker}",
        "attribution": "run_id",
        "job": job.job,
        "run_root": str(job.run_root),
        "db_paths": ";".join(sorted({str(r["db"])
                                     for r in every})),
        "verdict": verdict,
        "correct": "1" if accepted else ("0" if last is not None else ""),
        "submitted": "1" if terminal else "0",
        "reason": "" if last is None else str(last.get("reason") or ""),
        "speedup": num(sub["speedup"]) if sub else "",
        "baseline_ns": num(sub["baseline_ns"]) if sub else "",
        "native_ns": num(sub["native_ns"]) if sub else "",
        "runtime_s": f"{float(sub['native_ns']) / 1e9:.9f}" if sub and sub["native_ns"] else "",
        "baseline_s": f"{float(sub['baseline_ns']) / 1e9:.9f}" if sub and sub["baseline_ns"] else "",
        "suspect": num(sub["suspect"]) if sub else "",
        "preset": str(last["preset"]) if last else "",
        "datatype": str(last["datatype"]) if last else "",
        "language": str(last["language"]) if last else "",
        "baseline": str(sub["baseline"]) if sub else "",
        "compiler": str(on_task[-1]["compiler"] or "") if on_task else "",
        "execution": str(last.get("execution") or "") if last else "",
        "cpu": str(last.get("cpu") or "") if last else "",
        "best_score_speedup": f"{max(scores):.6f}" if scores else "",
        "n_calls": str(len(on_task)),
        "n_submit_calls": str(len([r for r in on_task if r.get("route") == "submit"])),
        "n_submissions": str(len([r for r in cell.submissions if r["benchmark"] == kernel])),
        "n_failed_submits": str(len([r for r in cell.attempts if r["benchmark"] == kernel])),
        "turns": str(max((int(r["round"]) for r in cell.calls), default=0)),
        "tokens": str(max((int(r["tokens"] or 0) for r in cell.calls), default=0)),
        "ts_first_ms": str(min(stamps)) if stamps else "",
        "ts_last_ms": str(max(stamps)) if stamps else "",
        "ts_first_iso": iso(min(stamps)) if stamps else "",
        "ts_last_iso": iso(max(stamps)) if stamps else "",
        "wall_clock_s": f"{(max(stamps) - min(stamps)) / 1000:.3f}" if stamps else "",
        "off_task_benchmarks": ";".join(off_task),
        "original_path": original_rel,
        "original_sha256": original_hash,
        "original_provenance": original_from,
        "original_origin": original_origin,
        "submitted_path": candidate_rel,
        "submitted_sha256": candidate_hash,
        "submitted_provenance": candidate_from,
    }


def summarise(job: Job, arm_dir: str, rows: list[dict[str, str]], adhoc: list[dict[str, Any]],
              kernels: list[str]) -> dict[str, str]:
    with_data = [r for r in rows if r["verdict"] != "no_data"]
    accepted = [r for r in rows if r["correct"] == "1"]
    seen = {r["kernel"] for r in with_data}
    missing = [k for k in kernels if k not in seen]
    return {
        "arm": job.arm,
        "arm_dir": arm_dir,
        "model": rows[0]["model"] if rows else job.model,
        "framing": job.framing,
        "job": job.job,
        "cells": str(len(rows)),
        "cells_with_data": str(len(with_data)),
        "cells_submitted": str(len([r for r in rows if r["submitted"] == "1"])),
        "cells_accepted": str(len(accepted)),
        "kernels_expected": str(len(kernels)),
        "kernels_with_data": str(len(seen)),
        "kernels_missing": ";".join(missing),
        "correct_rate_of_cells": f"{len(accepted) / len(rows):.4f}" if rows else "",
        "correct_rate_of_cells_with_data": f"{len(accepted) / len(with_data):.4f}" if with_data else "",
        "geomean_speedup_accepted": geomean([float(r["speedup"]) for r in accepted if r["speedup"]]),
        "geomean_speedup_best_score":
        geomean([float(r["best_score_speedup"]) for r in rows if r["best_score_speedup"]]),
        "total_tokens": str(sum(int(r["tokens"]) for r in rows)),
        "adhoc_rows": str(len(adhoc)),
        "off_task_rows": str(len([r for r in rows if r["off_task_benchmarks"]])),
    }


def write_csv(path: pathlib.Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    problems = load_problems(args.problems)
    kernels = sorted({p.kernel for p in problems})
    jobs = discover_jobs(args.runs)
    if not jobs:
        print(f"no git-scicomp jobs under {args.runs}", file=sys.stderr)
        return 1

    combined: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for job in sorted(jobs, key=lambda j: (j.model, j.framing)):
        arm_dir = job.arm[len(ARM_PREFIX):].replace("-", "_")
        # Rewritten from scratch so a rerun after a kernel is fixed cannot leave a stale file
        # behind that a reader would take for this run's output.
        shutil.rmtree(args.out / "artifacts" / arm_dir, ignore_errors=True)
        cells, adhoc = collect_cells(job)
        empty = Cell([], [], [], [])
        # One model served the whole job, so it is read once from any row that names it. Taking it
        # from the first cell reported the arm tag whenever that cell never reached the judge.
        named = sorted({
            str(row["optimizer"])
            for cell in cells.values()
            for row in cell.calls + cell.submissions + cell.attempts if row.get("optimizer")
        })
        model = named[0] if named else job.model
        rows = [
            cell_row(job, p, cells.get(p.problem_index, empty), args.benchmarks, args.out, arm_dir, model)
            for p in problems
        ]
        write_csv(args.out / "artifacts" / arm_dir / "results.csv", RESULT_FIELDS, rows)
        summaries.append(summarise(job, arm_dir, rows, adhoc, kernels))
        combined += rows

    write_csv(args.out / "data" / "git_experiment_all.csv", RESULT_FIELDS, combined)
    write_csv(args.out / "data" / "git_experiment_summary.csv", SUMMARY_FIELDS, summaries)

    for row in summaries:
        print(f"{row['arm_dir']:16s} cells={row['cells']:>3s} data={row['cells_with_data']:>3s} "
              f"submitted={row['cells_submitted']:>3s} accepted={row['cells_accepted']:>3s} "
              f"geomean={row['geomean_speedup_accepted'] or '-':>9s} missing={row['kernels_missing'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
