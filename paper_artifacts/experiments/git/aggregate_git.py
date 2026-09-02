"""Pool the git-scicomp cell table into ONE tidy kernel table, the shape the llr figures read.

``collect_git.py`` writes one row per CELL -- an (arm, kernel, attempt) triple. The llr8 and llr9
figures read one row per (model, language, leg, KERNEL). This file is the bridge, and it exists so
the framing A/B can be drawn by the same code, on the same axes, as the skills A/B: two experiments
drawn by two plotters diverge in ways a reader cannot see.

THE THREE POOLING RULES, restated because they are what a row means:

  THE UNIT IS THE KERNEL. An arm gets one vote per kernel, not one per attempt, so an arm that
  happens to submit the same flat kernel three times does not outvote one that solved three.

  THE THREE ATTEMPTS ARE INDEPENDENT AGENTS, not resubmissions of one. That is the one place this
  experiment differs from llr8/llr9, where "last" means the last thing ONE agent stood behind.
  Here there is no last in that sense, so ``last_speedup`` is the highest-numbered attempt that was
  graded correct -- a fixed rule that cannot cherry-pick, with ``best_speedup`` beside it so a
  reader can see when the two diverge (they diverge exactly when one attempt got lucky).

  A KERNEL WITH NO DATA IS STILL A ROW, with empty speed-ups and ``solved=0``. An arm that never
  reached a kernel must show as a kernel it did not solve, not vanish from the denominator.

THE LEGS. ``skills`` carries the framing, because that is the column the plotter groups on:
``0`` is the KERNEL framing (the hollow marker, the leg the pair started from) and ``1`` is the
REPOSITORY framing. The column keeps the llr name so the tidy table stays one schema.

Re-running over an unchanged cell table reproduces byte-identical output.

    python3 aggregate_git.py [--data data/git_experiment_all.csv] [--out data/kernels.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import statistics

#: The ten scientific-computing kernels the campaign served, and the denominator of every success
#: rate here. Fixed rather than read off the data: an arm that never reached a kernel must not
#: shrink its own denominator.
TAG_SIZE = 10

#: ``framing`` as the collector records it -> the ``skills`` leg the plotter groups on.
LEG_OF_FRAMING = {"kernel": "0", "repo": "1"}

#: The collector's full model ids -> the short keys the palette and the label table are held under,
#: so this experiment's hues match the ones llr8 and llr9 give the same models.
MODEL_OF = {"openai/gpt-oss-120b": "oss120b", "Qwen/Qwen3.8-27B-FP8": "qwen38"}

#: Every graded submission in this campaign was C. Recorded as a column rather than assumed,
#: because the plotter labels the row with it and a silent default would mislabel a Fortran arm.
LANGUAGE = "c"

FIELDS = ("model", "language", "skills", "benchmark", "waves", "submissions", "last_speedup", "best_speedup",
          "median_speedup", "ordering", "solved", "tokens")

ROOT = pathlib.Path(__file__).resolve().parent


def speedup_of(row: dict[str, str]) -> float | None:
    """The cell's speed-up, but ONLY when the judge graded it correct.

    A speed-up on an incorrect submission is the time of a program that computes the wrong thing.
    The collector records it because it is evidence; nothing that pools it may use it.
    """
    if row["correct"] != "1" or not row["speedup"]:
        return None
    value = float(row["speedup"])
    return value if value > 0.0 else None


def pool(cells: list[dict[str, str]]) -> dict[str, str]:
    """One kernel's three attempts -> the tidy row for it."""
    graded = sorted((int(c["attempt"]), s) for c in cells if (s := speedup_of(c)) is not None)
    values = [s for _, s in graded]
    submissions = sum(int(c["n_submissions"] or 0) for c in cells)
    # Tokens are summed over ALL three attempts, not only the graded ones: the question the token
    # figure asks is what the arm spent to attack this kernel, and two attempts that ran out of
    # turns cost what they cost.
    tokens = sum(int(c["tokens"] or 0) for c in cells)
    return {
        "waves": "git",
        "submissions": str(submissions),
        "last_speedup": f"{values[-1]:.6f}" if values else "",
        "best_speedup": f"{max(values):.6f}" if values else "",
        "median_speedup": f"{statistics.median(values):.6f}" if values else "",
        # There is one wave, so there is nothing for a wave-ordering rule to decide.
        "ordering": "single_wave",
        "solved": "1" if values else "0",
        "tokens": str(tokens),
    }


def rows(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    kernels: set[str] = set()
    legs: set[tuple[str, str]] = set()
    for cell in cells:
        model, leg = MODEL_OF[cell["model"]], LEG_OF_FRAMING[cell["framing"]]
        grouped[(model, leg, cell["kernel"])].append(cell)
        kernels.add(cell["kernel"])
        legs.add((model, leg))

    out: list[dict[str, str]] = []
    for model, leg in sorted(legs):
        for kernel in sorted(kernels):
            out.append({
                "model": model,
                "language": LANGUAGE,
                "skills": leg,
                "benchmark": kernel,
                **pool(grouped.get((model, leg, kernel), [])),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data" / "git_experiment_all.csv")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "kernels.csv")
    args = parser.parse_args()

    with args.data.open() as handle:
        cells = list(csv.DictReader(handle))
    table = rows(cells)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)

    for model, leg in sorted({(r["model"], r["skills"]) for r in table}):
        mine = [r for r in table if r["model"] == model and r["skills"] == leg]
        solved = sum(int(r["solved"]) for r in mine)
        framing = "kernel" if leg == "0" else "repo"
        print(f"  {model:9s} {framing:6s} solved {solved}/{TAG_SIZE}  "
              f"{sum(int(r['tokens']) for r in mine) / 1e6:5.1f}M tokens over {len(mine)} kernels")
    print(f"  wrote {args.out} ({len(table)} rows = {len(table) // TAG_SIZE} legs x {TAG_SIZE} kernels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
