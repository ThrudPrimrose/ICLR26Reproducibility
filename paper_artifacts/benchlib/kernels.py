"""Pool every collected llr8 wave into ONE tidy table, one row per (model, language, skills, kernel).

This is the only place the campaign's aggregation rules live. The figures read the table it writes
and do no pooling of their own, so a rule can be changed, tested and reviewed in one file rather
than in each plot.

THE UNIT IS THE KERNEL, not the submission row. A per-row geomean is weighted by how often an agent
pressed submit: llr8w6's unskilled qwen38 C arm has 15 rows over 3 kernels, 11 of them one kernel
at 1.00x, and reads 2.08x per row against 6.89x per kernel.

THE VALUE IS THE LAST SUBMISSION. An agent's last submission is the one it stood behind; its best is
the luckiest attempt it ever made, which a resubmitting agent gets more of than a decisive one, so
"best" rewards volume. ``best_speedup`` stays in the table beside ``last_speedup`` as a diagnostic:
a cell whose two differ widely was carried by cherry-picking rather than by a repeatable win.

ORDER IS READ, NEVER GUESSED. "Last" is the maximum ``ts_ms``, the judge's own stamp on the graded
row, carried through by collect.py. Row order in a CSV is not an ordering. A kernel whose
trustworthy submissions carry no stamp, or whose latest stamp is a tie between two rows, has NO
defensible last submission: it is written with an empty ``last_speedup`` and an ``ordering`` of
``unstamped`` or ``tied``, counted, and reported. It is never filled in from row order.

THE ROW POOLS EVERY WAVE. A wave is not an experiment: waves 3, 4, 6, 7 and 12 to 15 are COMPLETION
waves that re-run only the kernels an earlier arm never submitted, and w8/w9 (and w10/w11) are two
HALVES of one 40-kernel draw that were split to fit the node budget. What the campaign measures is:
of the llr-focus40 kernels, which ones did this model / language / skills configuration ever solve,
given every attempt it got. So a kernel appears once per leg, with every wave that touched it named
in ``waves``, and a solve is a union over waves rather than a sum of per-wave counts.

TOKENS ARE SUMMED OVER WAVES, MAXIMISED WITHIN ONE. ``tokens`` on a judge call is the agent's
CUMULATIVE usage at that moment, so one wave's cost for a kernel is that wave's high-water mark;
two waves that both drew the kernel are two agents, and their costs add.

Driven by each experiment's ``aggregate_<name>.py``, which supplies the tag size and exclusions.
"""
from __future__ import annotations

import csv
import dataclasses
import pathlib
import sys

#: What the tag size and the exclusion set are is a property of the EXPERIMENT, not of the pooling
#: rule, so both arrive as arguments: llr8 and llr9 draw from tags of different size and drop
#: different kernels, and a default here would silently give one experiment the other's denominator.

FIELDS = ("model", "language", "skills", "benchmark", "waves", "submissions", "last_speedup", "best_speedup",
          "median_speedup", "ordering", "solved", "tokens")

#: What ``ordering`` says about a kernel's last submission: it is stamped and unique, every stamp is
#: missing, or the latest stamp is shared by two rows and no row is the last one.
STAMPED, UNSTAMPED, TIED, UNTIMED = "stamped", "unstamped", "tied", "untimed"


@dataclasses.dataclass
class Kernel:
    """One (model, language, skills, kernel) cell, accumulated across waves."""

    waves: list[str] = dataclasses.field(default_factory=list)
    timed: list[tuple[int, float]] = dataclasses.field(default_factory=list)
    unstamped: int = 0
    solved: bool = False
    tokens: int = 0


def read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def wave_dirs(data: pathlib.Path) -> list[pathlib.Path]:
    """Wave directories in wave order: the numbered ``w<N>`` waves first, then any named one.

    Numeric within the numbered ones, because ``w10`` sorts before ``w2`` as text and a wave printed
    out of order reads as a missing wave. A named directory (``v9``) is a wave that is not part of
    the ``w<N>`` series -- a separate campaign pooled into the same experiment -- and sorts last so
    the inherited history stays in run order ahead of it.
    """
    found = [p for p in data.iterdir() if p.is_dir() and (p / "submissions.csv").is_file()]
    numbered = [p for p in found if p.name.startswith("w") and p.name[1:].isdigit()]
    named = [p for p in found if p not in numbered]
    return sorted(numbered, key=lambda p: int(p.name[1:])) + sorted(named, key=lambda p: p.name)


def trustworthy(row: dict[str, str]) -> bool:
    """A submission the judge stands behind and that carries a timing.

    Suspect rows are dropped rather than averaged in: the judge marks a submission suspect when it
    cannot stand behind the timing, so keeping it quotes a number the harness itself disowned.
    """
    if row["suspect"] not in ("", "0"):
        return False
    return bool(row["speedup"]) and float(row["speedup"]) > 0


def stamp(row: dict[str, str]) -> int:
    """The judge's epoch-millisecond stamp, or 0 when the shard carried none."""
    value = row.get("ts_ms", "")
    return int(value) if value.isdigit() else 0


def last_of(timed: list[tuple[int, float]]) -> tuple[float | None, str]:
    """The speed-up of the latest submission, and what the ordering allowed.

    Two rows sharing the latest stamp are not one submission: which of them the agent stood behind
    is unrecoverable, so the kernel reports no last value rather than an arbitrary one.
    """
    if not timed:
        return None, UNTIMED
    latest = max(ts for ts, _ in timed)
    if latest == 0:
        return None, UNSTAMPED
    tail = [value for ts, value in timed if ts == latest]
    if len(tail) > 1:
        return None, TIED
    return tail[0], STAMPED


def collect(data: pathlib.Path, excluded: frozenset[str]) -> dict[tuple[str, str, str, str], Kernel]:
    """``(model, language, skills, kernel) -> Kernel`` over every wave under ``data``, less ``excluded``."""
    cells: dict[tuple[str, str, str, str], Kernel] = {}
    for wave in wave_dirs(data):
        per_arm_tokens: dict[tuple[str, str, str, str], int] = {}
        for row in read(wave / "calls.csv"):
            if row["benchmark"] in excluded:
                continue
            key = (row["model"], row["language"], row["skills"], row["benchmark"])
            cell = cells.setdefault(key, Kernel())
            if wave.name not in cell.waves:
                cell.waves.append(wave.name)
            cell.solved = cell.solved or (row["route"] == "submit" and row["status"] == "ok")
            spent = int(row["tokens"] or 0)
            per_arm_tokens[key] = max(per_arm_tokens.get(key, 0), spent)
        for key, spent in per_arm_tokens.items():
            cells[key].tokens += spent
        for row in read(wave / "submissions.csv"):
            if row["benchmark"] in excluded or not trustworthy(row):
                continue
            key = (row["model"], row["language"], row["skills"], row["benchmark"])
            cell = cells.setdefault(key, Kernel())
            if wave.name not in cell.waves:
                cell.waves.append(wave.name)
            ts = stamp(row)
            cell.unstamped += int(ts == 0)
            cell.timed.append((ts, float(row["speedup"])))
    return cells


def rows(cells: dict[tuple[str, str, str, str], Kernel]) -> list[dict[str, object]]:
    """The tidy table, sorted so the CSV is byte-stable across runs."""
    out: list[dict[str, object]] = []
    for (model, language, skills, benchmark), cell in sorted(cells.items()):
        values = sorted(value for _, value in cell.timed)
        last, ordering = last_of(cell.timed)
        out.append({
            "model": model,
            "language": language,
            "skills": skills,
            "benchmark": benchmark,
            "waves": ";".join(cell.waves),
            "submissions": len(values),
            "last_speedup": f"{last:.6f}" if last is not None else "",
            "best_speedup": f"{values[-1]:.6f}" if values else "",
            "median_speedup": f"{values[len(values) // 2]:.6f}" if values else "",
            "ordering": ordering,
            "solved": int(cell.solved),
            "tokens": cell.tokens,
        })
    return out


def write(path: pathlib.Path, table: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(table)


def run(data: pathlib.Path, out: pathlib.Path, excluded: frozenset[str], collector: str) -> int:
    """Pool ``data``'s wave directories into the tidy table at ``out``, dropping ``excluded``."""
    waves = wave_dirs(data)
    if not waves:
        raise SystemExit(f"no w<N> wave directories under {data}; run {collector} first")
    table = rows(collect(data, excluded))
    write(out, table)

    legs = {(r["model"], r["language"], r["skills"]) for r in table}
    unordered = [r for r in table if r["ordering"] in (UNSTAMPED, TIED)]
    print(f"  {len(waves)} waves -> {len(table)} kernel rows over {len(legs)} legs -> {out}")
    print(
        f"  timed kernels: {sum(1 for r in table if r['submissions'])}, solved: {sum(int(r['solved']) for r in table)}")
    if unordered:
        print(
            f"  NO LAST SUBMISSION for {len(unordered)} kernels (no usable order): "
            f"{', '.join(str(r['benchmark']) for r in unordered[:6])}",
            file=sys.stderr)
    return 0
