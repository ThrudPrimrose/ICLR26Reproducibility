"""Three llr-focus40 figures across every collected wave, aggregated PER KERNEL.

The unit of measurement is the KERNEL, not the submission row. ``summary.csv``'s ``speedup_geomean``
is taken over submission rows, so it is weighted by how often an agent pressed submit: llr8w6's
unskilled qwen38 C arm has 15 rows over 3 kernels, 11 of them one kernel at 1.00x, and reads 2.08x
per row against 6.89x per kernel. Everything here takes each kernel's BEST timed submission and the
geometric mean of those, and quotes the geomean as the headline -- speed-up is a ratio, so the mean
that matches the product over the set is the geometric one, and the median of a mostly-flat
distribution reports "no effect" everywhere. The per-kernel MEDIAN variant is drawn as a tick beside
each marker, so an arm carried by one lucky resubmission shows as a wide gap between the two.

The ARM is the row, never a pool across waves. Waves 3, 4, 6 and 7 are COMPLETION waves that re-ran
only the kernels an earlier arm never submitted, so an arm's kernels are a subset computed from what
its predecessor failed at; summing them into a per-model total would report a denominator no arm
ever drew from.

Only PAIRED comparisons say anything about skills. The two legs of a completion wave drew subsets
computed from DIFFERENT predecessors and can share almost nothing -- llr8w6's qwen38 pair shares one
reached kernel and none with a timed submission on both sides. Where a pair shares timed kernels,
both bars are restricted to those. Where it shares none, both are drawn over each arm's own kernels
and the group is marked UNPAIRED, which is a statement that the two markers are not comparable
rather than a comparison.

THE FORM. Nineteen groups is too many for grouped vertical bars: the labels have to rotate, the two
legs of a pair end up further apart than two legs of neighbouring pairs, and the value labels have
to stand on end to fit. Every figure here is therefore a DUMBBELL row per group -- hollow marker for
the base leg, filled for the skills leg, joined in the model's colour -- which puts the pair's two
legs on one line and leaves the left column free for a label that reads horizontally.

Usage:  python3 plot_llr40.py
"""
from __future__ import annotations

import csv
import dataclasses
import math
import pathlib
import sys
from collections.abc import Callable

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

import artifact_style  # noqa: E402  -- ditto: these pull matplotlib in themselves
import collect  # noqa: E402
import collect_llr40  # noqa: E402

MODEL_COLOR = artifact_style.MODEL_COLOR

#: The two skills legs: base first, so the hollow marker is always the one the pair started from.
LEGS = ("0", "1")


@dataclasses.dataclass(frozen=True)
class Group:
    """One (wave, model, language) cell: the skills legs that exist, and their timed kernels."""

    wave: str
    model: str
    language: str
    arms: dict[str, dict[str, str]]
    kernels: dict[str, dict[str, list[float]]]

    def shared(self) -> list[str]:
        """Kernels BOTH legs produced a trustworthy timing for; empty when the pair is unpaired."""
        if len(self.kernels) < 2:
            return []
        return sorted(set(self.kernels["0"]) & set(self.kernels["1"]))

    def timed(self, skills: str) -> list[list[float]]:
        """This leg's per-kernel speed-ups, narrowed to the shared kernels when there are any."""
        table = self.kernels.get(skills, {})
        keep = self.shared()
        return [table[k] for k in keep] if keep else list(table.values())

    def label(self) -> str:
        language = "C" if self.language == "c" else "Fortran"
        return f"{artifact_style.MODEL_LABEL[self.model]} / {language}  w{self.wave[len('llr8w'):]}"


def style() -> None:
    """Kept as the entry point this script already called; the look itself lives in artifact_style."""
    artifact_style.apply()


def read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def timed_kernels(rows: list[dict[str, str]]) -> dict[str, dict[str, list[float]]]:
    """``arm -> kernel -> every trustworthy timed speed-up on it``.

    Suspect rows are dropped rather than averaged in: the judge marks a submission suspect when it
    cannot stand behind the timing, so keeping it quotes a number the harness itself disowned.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["suspect"] not in ("", "0") or not row["speedup"] or float(row["speedup"]) <= 0:
            continue
        out.setdefault(row["arm"], {}).setdefault(row["benchmark"], []).append(float(row["speedup"]))
    return {arm: {k: sorted(v) for k, v in sorted(kernels.items())} for arm, kernels in out.items()}


def groups() -> list[Group]:
    """Every collected wave's cells, ordered by model, then language, then wave.

    The order is fixed rather than by value so a hue never tracks rank and a group sits in the same
    place in all three figures, which is what makes them readable side by side.
    """
    root = pathlib.Path(__file__).resolve().parent
    found: dict[tuple[str, str, str], Group] = {}
    for wave in collect_llr40.waves():
        data = root / f"data-{wave}"
        kernels = timed_kernels(read(data / "submissions.csv"))
        for row in read(data / "summary.csv"):
            key = (row["model"], row["language"], wave)
            cell = found.setdefault(key, Group(wave, row["model"], row["language"], {}, {}))
            cell.arms[row["skills"]] = row
            cell.kernels[row["skills"]] = kernels.get(row["arm"], {})
    order = sorted(found, key=lambda k: (list(MODEL_COLOR).index(k[0]), k[1], int(k[2][len("llr8w"):])))
    return [found[k] for k in order]


def usable(value: float, floor: float) -> float | None:
    """A leg is drawn only where it has a number to draw; NaN and a value at the floor are not."""
    return None if math.isnan(value) or value <= floor else value


def row_figure(cells: list[Group], title: str, xlabel: str) -> tuple:
    tall = 1.15 + 0.29 * len(cells)
    fig, ax = plt.subplots(figsize=(7.0, tall))
    fig.subplots_adjust(left=0.315, right=0.845, top=1.0 - 0.72 / tall, bottom=0.62 / tall)
    artifact_style.axis_title(ax, title, pad=18.0)
    ax.set_xlabel(xlabel)
    return fig, ax


def draw_rows(ax, cells: list[Group], value: Callable[[Group, str], float | None], note: Callable[[Group], str],
              floor: float) -> None:
    for row, cell in enumerate(cells):
        legs = {skills: value(cell, skills) if skills in cell.arms else None for skills in LEGS}
        if legs["0"] is None and legs["1"] is None:
            artifact_style.right_label(ax, row, "no timed arm", artifact_style.MUTED, size=6.5)
            continue
        artifact_style.dumbbell(ax, row, legs["0"], legs["1"], MODEL_COLOR[cell.model])
        artifact_style.right_label(ax, row, note(cell))
    ax.set_xlim(left=floor)


def finish(fig, ax, cells: list[Group], out: pathlib.Path, unpaired: bool = False) -> None:
    if unpaired:
        for tick, cell in zip(ax.get_yticklabels(), cells, strict=True):
            if not cell.shared():
                tick.set_color(artifact_style.WARN)
    artifact_style.key(ax, [("base", artifact_style.marker(artifact_style.INK2, on=False)),
                            ("+ skills", artifact_style.marker(artifact_style.INK2, on=True))],
                       anchor=(1.0, 1.005))
    artifact_style.save(fig, out)


def speedup_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Per-kernel geomean speed-up, base leg against skills leg, one row per arm PAIR."""
    pairs = [c for c in cells if len(c.arms) == 2]
    lonely = [c for c in cells if len(c.arms) == 1]

    def value(cell: Group, skills: str) -> float | None:
        timed = cell.timed(skills)
        return usable(collect.geomean([max(v) for v in timed]) if timed else math.nan, 1.0)

    def middle(cell: Group, skills: str) -> float | None:
        timed = cell.timed(skills)
        return usable(collect.geomean([v[len(v) // 2] for v in timed]) if timed else math.nan, 1.0)

    def note(cell: Group) -> str:
        return "n=" + "/".join(str(len(cell.timed(s))) for s in LEGS)

    labels = [c.label() + ("  UNPAIRED" if not c.shared() else "") for c in pairs]
    fig, ax = row_figure(pairs, "Speed-up with and without the skills packet, aggregated per kernel",
                         "geomean of per-kernel best speed-up (x, log scale)")
    # Log: a speed-up is a ratio, and one n=2 group at 71x flattens every 3x-to-8x group into the
    # axis floor on a linear scale.
    ax.set_xscale("log")
    artifact_style.row_axis(ax, labels, gridaxis="none")
    draw_rows(ax, pairs, value, note, 1.0)
    for row, cell in enumerate(pairs):
        # The per-kernel MEDIAN geomean as a tick. A marker far from its tick is an arm whose
        # speed-up rests on a best-of-many resubmission rather than a repeatable win.
        for skills in LEGS:
            centre = middle(cell, skills)
            if centre is not None:
                ax.plot([centre, centre], [row - 0.26, row + 0.26], color=artifact_style.INK, linewidth=0.9, zorder=4)
    ax.set_xlim(1.0, ax.get_xlim()[1] * 1.3)
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x")
    finish(fig, ax, pairs, out, unpaired=True)
    print(f"  speedup: tick is the geomean over each kernel's MEDIAN timing, marker the geomean over its BEST; "
          f"{sum(1 for c in pairs if not c.shared())} of {len(pairs)} pairs share no timed kernel and are "
          f"marked UNPAIRED")
    print(f"  speedup: {len(lonely)} arms have no skills counterpart and are omitted: "
          f"{', '.join(c.label() for c in lonely)}")


def success_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Solved kernels over the set the arm DREW FROM, which is not 40 for a completion-wave arm."""

    def value(cell: Group, skills: str) -> float | None:
        arm = cell.arms[skills]
        return 100.0 * int(arm["solved"]) / int(arm["problems"])

    def note(cell: Group) -> str:
        return " ".join(f"{cell.arms[s]['solved']}/{cell.arms[s]['problems']}" for s in LEGS if s in cell.arms)

    fig, ax = row_figure(cells, "Successful completion, against the kernel set each arm drew from",
                         "successful completion rate (% of kernels drawn)")
    ax.set_xlim(0.0, 100.0)
    artifact_style.row_axis(ax, [c.label() for c in cells], gridaxis="none")
    for row in range(len(cells)):
        artifact_style.rounded_bar(ax, 0.0, 100.0, row, 0.62, artifact_style.RAISE, zorder=1.0)
    draw_rows(ax, cells, value, note, 0.0)
    ax.set_xlim(0.0, 100.0)
    finish(fig, ax, cells, out)
    print("  success: denominator is collect.problem_count(): 40 for a full arm, the computed gap list for a "
          "completion-wave arm, 20 for a kimi Fortran batch, never a hard-coded 40; the right column is "
          "solved / drawn for each leg")


def tokens_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Token cost of one kernel, first agent turn to last grade -- the term the packet moves."""

    def value(cell: Group, skills: str) -> float | None:
        return int(cell.arms[skills]["tokens_per_kernel"]) / 1e6

    def note(cell: Group) -> str:
        return "n=" + "/".join(cell.arms[s]["attempted"] for s in LEGS if s in cell.arms)

    fig, ax = row_figure(cells, "Token cost per kernel, with and without the skills packet",
                         "million tokens per kernel reached")
    artifact_style.row_axis(ax, [c.label() for c in cells])
    draw_rows(ax, cells, value, note, 0.0)
    ax.margins(x=0.1)
    finish(fig, ax, cells, out)
    print("  tokens: the agent's cumulative usage at its last judge call on a kernel, summed over the kernels the "
          "arm reached and divided by that count; n = kernels reached")


def main() -> int:
    style()
    cells = groups()
    out = pathlib.Path(__file__).resolve().parent / "figures"
    speedup_figure(cells, out / "llr40_speedup_per_kernel")
    success_figure(cells, out / "llr40_success_rate")
    tokens_figure(cells, out / "llr40_tokens_per_kernel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
