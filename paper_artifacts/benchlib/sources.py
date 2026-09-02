"""Save the FINAL submitted source for every cell of an experiment, beside the rows it produced.

A speed-up table says how fast an answer was; it does not say what the answer was. The judge stored
every submitted source as a blob beside its shard, so the code exists -- it is just stranded in a
3 GB tree nobody clones. This copies out one file per cell and writes an index that joins it back
to the table, so a reader who wants to know what "8.4x on tsvc_2_s311, Fortran, skills" actually
was can open it.

ONE FILE PER CELL, AND IT IS THE LAST SUBMISSION. The cell is
``(model, language, skills, kernel)`` pooled over every wave, exactly as in the tidy table, and the
file is the source of the submission with the latest judge stamp among that cell's trustworthy
rows -- the same rule ``benchlib.kernels`` uses for ``last_speedup``, so the code saved here is the
code the quoted number came from. A cell whose latest stamp is missing or tied has no defensible
last submission and is reported rather than guessed at.

THE RUN IS IDENTIFIED FROM THE SHARD, NOT FROM THE DIRECTORY NAME. ``sources.ts`` is the graded
row's own stamp, so ``(job, benchmark, ts)`` names the graded submission exactly; the ``run_id``
that comes back with it is then checked against the leg the row was attributed to. Several waves'
env files mis-inherited ``RUN_ROOT`` and their jobs sit under another wave's directory, so a
directory name is not evidence of anything.

Layout, mirroring the shape the git experiment already uses:

    artifacts/sources/<model>-<language>[-skills]/<kernel>__submitted.<ext>
    artifacts/sources/index.csv
"""
from __future__ import annotations

import collections
import csv
import hashlib
import pathlib
import sqlite3
import sys

from benchlib import constructs
from benchlib import kernels

#: What the saved file is called, by the language the agent DELIVERED. An arm's task language and
#: its delivered language can differ (the restricted prompt sanctions delivering python), so the
#: extension is taken from the source row rather than from the arm.
EXTENSION = {"c": ".c", "cpp": ".cpp", "fortran": ".f90", "python": ".py"}

INDEX_COLUMNS = [
    "model", "language", "skills", "benchmark", "wave", "job", "run_id", "ts_ms", "speedup", "sha256", "n_bytes", "path"
]


def leg_dir(model: str, language: str, skills: str) -> str:
    """``<model>-<language>[-skills]``: the arm directory, one per cell of the tidy table."""
    return f"{model}-{language}" + ("-skills" if skills not in ("", "0") else "")


def leg_of(run_id: str) -> tuple[str, str, bool] | None:
    """``(model, language, skills)`` read out of a ``run_id``, or None when it is not an arm row.

    The launcher writes ``run_id = <campaign>-<model>-<language>[-skills][-<batch>].n<N>.p<N>.w<N>``.
    A hand-run probe writes something else (``adhoc``), and crediting one of those to an arm is how
    a 1007x submission no arm ever made got into a wave's table.
    """
    parts = run_id.split(".", 1)[0].split("-")
    if len(parts) < 3:
        return None
    skills = "skills" in parts
    tail = parts[-1]
    batch = tail if tail != "skills" and tail.startswith(("r", "a", "b")) and len(tail) <= 2 else ""
    core = [p for p in parts[1:] if p not in ("skills", batch)]
    if len(core) < 2:
        return None
    return "-".join(core[:-1]), core[-1], skills


def winners(data: pathlib.Path,
            excluded: frozenset[str]) -> tuple[dict[tuple[str, str, str, str], dict[str, str]], list[str]]:
    """The last trustworthy submission of every cell, pooled over the experiment's waves.

    Returns the winning rows keyed by cell, and the cells that have submissions but no usable
    order. Reads the wave CSVs rather than the shards so that the rule cannot drift from the one
    the tidy table applies.
    """
    timed: dict[tuple[str, str, str, str], list[tuple[int, dict[str, str]]]] = collections.defaultdict(list)
    for wave in kernels.wave_dirs(data):
        with (wave / "submissions.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row["benchmark"] in excluded or not kernels.trustworthy(row):
                    continue
                row["wave"] = wave.name
                timed[(row["model"], row["language"], row["skills"], row["benchmark"])].append(
                    (kernels.stamp(row), row))
    best: dict[tuple[str, str, str, str], dict[str, str]] = {}
    unordered: list[str] = []
    for key, rows in timed.items():
        latest = max(ts for ts, _ in rows)
        tail = [row for ts, row in rows if ts == latest]
        if latest == 0 or len(tail) > 1:
            unordered.append("/".join(key))
            continue
        best[key] = tail[0]
    return best, unordered


def blob_of(run_root: pathlib.Path, job: str, benchmark: str, ts: int,
            leg: tuple[str, str, bool]) -> tuple[str, pathlib.Path, str] | None:
    """``(run_id, blob path, delivered language)`` of the source graded at ``ts``, or None.

    ``sources.ts`` is documented as the graded row's stamp, so this is a join and not a search.
    """
    for db in constructs.find_shards(run_root, job):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("select run_id, language, path from sources where benchmark = ? and ts = ?",
                               (benchmark, ts)).fetchall()
        except sqlite3.DatabaseError:
            continue
        finally:
            con.close()
        for run_id, language, rel in rows:
            if leg_of(str(run_id)) != leg:
                continue
            blob = constructs.blob_root(db) / rel
            if blob.is_file():
                return str(run_id), blob, str(language or "")
    return None


def export(run_root: pathlib.Path, data: pathlib.Path, out: pathlib.Path, excluded: frozenset[str]) -> int:
    """Write one final source per cell under ``out``, plus ``index.csv``. Returns files written."""
    best, unordered = winners(data, excluded)
    out.mkdir(parents=True, exist_ok=True)
    index: list[list[object]] = []
    missing: list[str] = []
    for (model, language, skills, benchmark), row in sorted(best.items()):
        leg = (model, language, skills not in ("", "0"))
        found = blob_of(run_root, row["job"], benchmark, kernels.stamp(row), leg)
        if found is None:
            missing.append(f"{model}/{language}/{skills}/{benchmark}")
            continue
        run_id, blob, delivered = found
        body = blob.read_bytes()
        arm = leg_dir(model, language, skills)
        path = pathlib.Path(arm) / f"{benchmark}__submitted{EXTENSION.get(delivered or language, '.txt')}"
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(body)
        index.append([
            model, language, skills, benchmark, row["wave"], row["job"], run_id,
            kernels.stamp(row), row["speedup"],
            hashlib.sha256(body).hexdigest(),
            len(body),
            path.as_posix()
        ])
    with (out / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        writer.writerows(sorted(index))
    print(f"  final sources: {len(index)} written of {len(best)} cells with a last submission -> {out}")
    if unordered:
        print(f"  no last submission (unstamped or tied) for {len(unordered)} cells: {', '.join(unordered[:6])}",
              file=sys.stderr)
    if missing:
        print(f"  NO STORED SOURCE for {len(missing)} cells: {', '.join(missing[:6])}", file=sys.stderr)
    return len(index)
