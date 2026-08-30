"""Three llr-focus40 figures across every collected wave, aggregated PER KERNEL.

The unit of measurement is the KERNEL, not the submission row. ``summary.csv``'s ``speedup_geomean``
is taken over submission rows, so it is weighted by how often an agent pressed submit: llr8w6's
unskilled qwen38 C arm has 15 rows over 3 kernels, 11 of them one kernel at 1.00x, and reads 2.08x
per row against 6.89x per kernel. Everything here takes each kernel's BEST timed submission and the
geometric mean of those, and quotes the geomean as the headline -- speed-up is a ratio, so the mean
that matches the product over the set is the geometric one, and the median of a mostly-flat
distribution reports "no effect" everywhere. The per-kernel MEDIAN variant is drawn as a tick beside
each marker, so an arm carried by one lucky resubmission shows as a wide gap between the two.

THE ROW IS THE EXPERIMENT, POOLED OVER ITS WAVES. A wave is not an experiment: waves 3, 4, 6, 7 and
12 are COMPLETION waves that re-run only the kernels an earlier arm never submitted, and w8/w9 (and
w10/w11) are two HALVES of one 40-kernel draw that were split to fit the node budget. Plotted one
row per wave, those halves read as four independent 20-kernel experiments and a one-kernel gap fill
reads as a whole arm at 100%. What the campaign actually measures is: of the llr-focus40 kernels,
which ones did this model / language / skills configuration ever solve, given every attempt it got.

So a row pools every wave of one cell and the numerator is the UNION of distinct kernels solved --
not a sum of per-wave rates, which would be meaningless over different subsets, and not a sum of
counts, which would double-count a kernel two waves both drew. The denominator is the tag, 40.

ONE WRINKLE IN THE DENOMINATOR. The tag's membership changed once: tsvc_2_s2233 was swapped out for
tsvc_2_s232 on 2026-08-30 because no arm could ever score it. A cell whose waves straddle that swap
drew from both memberships and so attempted 41 distinct kernels. The denominator stays 40 -- the set
size any single draw saw -- and no cell exceeds it, but a pooled row is over a set that shifted by
one member partway through, which is worth knowing before quoting it to four significant figures.

Skills legs are paired ON THE POOLED SETS. Two legs of a single completion wave often share almost
no kernels, which is why the per-wave figure had to mark pairs UNPAIRED; pooled, both legs cover
essentially the whole tag and the pairing is real.

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
    """One (model, language) cell POOLED over every wave: the skills legs, and their timed kernels."""

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
        return f"{artifact_style.MODEL_LABEL[self.model]} / {language}"


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


#: The llr-focus40 tag: the kernel set every wave of this campaign draws from. Not read off any one
#: arm, because a completion arm draws a SUBSET of it and would report its own gap list as the
#: experiment's denominator.
TAG_SIZE = 40

#: Dropped from every figure. `tsvc_2_s2233` took 296 judge calls across the campaign and graded ok
#: ZERO times, in every arm of every wave -- a kernel no arm can score measures the harness, not the
#: model (the pass/fail is size- and thread-dependent; see the open harness bug). It was swapped out
#: of the tag for tsvc_2_s232 partway through, which is the only reason a pooled cell could show 41
#: distinct kernels over a 40-kernel tag. Excluding it makes the pooled set exactly the tag, costs
#: no numerator anywhere (it was never solved), and stops its 296 unscoreable calls from inflating
#: the token cost of the kernels that ARE scoreable.
EXCLUDED = frozenset({"tsvc_2_s2233"})


def groups() -> list[Group]:
    """One pooled cell per (model, language), ordered by model then language.

    The order is fixed rather than by value so a hue never tracks rank and a group sits in the same
    place in all three figures, which is what makes them readable side by side.

    Pooling is over DISTINCT KERNELS, never over per-wave counts: a kernel that two completion waves
    both drew is one kernel, and adding their counts would report it twice. `solved` and `attempted`
    are therefore set unions taken from calls.csv, which is also where collect.py defines a solve
    (a `submit` route that graded `ok`), so the pooled numerator means the same thing as the
    per-wave one it replaces.
    """
    root = pathlib.Path(__file__).resolve().parent
    solved: dict[tuple[str, str, str], set[str]] = {}
    attempted: dict[tuple[str, str, str], set[str]] = {}
    tokens: dict[tuple[str, str, str], int] = {}
    kernels: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    waves: dict[tuple[str, str], set[str]] = {}

    for wave in collect_llr40.waves():
        data = root / f"data-{wave}"
        if not (data / "calls.csv").is_file():
            continue
        for row in read(data / "calls.csv"):
            if row["benchmark"] in EXCLUDED:
                continue
            leg = (row["model"], row["language"], row["skills"])
            attempted.setdefault(leg, set()).add(row["benchmark"])
            if row["route"] == "submit" and row["status"] == "ok":
                solved.setdefault(leg, set()).add(row["benchmark"])
        per_arm = timed_kernels(read(data / "submissions.csv"))
        for row in read(data / "summary.csv"):
            leg = (row["model"], row["language"], row["skills"])
            waves.setdefault((row["model"], row["language"]), set()).add(wave)
            tokens[leg] = tokens.get(leg, 0) + int(row["tokens"])
            for kernel, speedups in per_arm.get(row["arm"], {}).items():
                if kernel in EXCLUDED:
                    continue
                kernels.setdefault(leg, {}).setdefault(kernel, []).extend(speedups)

    found: dict[tuple[str, str], Group] = {}
    for model, language, skills in sorted(attempted):
        leg = (model, language, skills)
        reached = len(attempted[leg])
        hit = len(solved.get(leg, ()))
        cell = found.setdefault((model, language), Group("pooled", model, language, {}, {}))
        cell.arms[skills] = {
            "solved": str(hit),
            "problems": str(TAG_SIZE),
            "attempted": str(reached),
            "tokens_per_kernel": str(tokens.get(leg, 0) // reached if reached else 0),
            "waves": str(len(waves.get((model, language), ()))),
        }
        cell.kernels[skills] = {k: sorted(v) for k, v in sorted(kernels.get(leg, {}).items())}
    order = sorted(found, key=lambda k: (list(MODEL_COLOR).index(k[0]), k[1]))
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


#: Tick candidates for the speed-up axis, in the order a reader expects to see them on a log scale.
#: The set is filtered to whatever range the data occupies rather than fixed, because a fixed
#: 1..100 ladder sized for one 71x per-wave outlier squeezes a pooled 5x-to-12x campaign into a
#: third of the axis and makes six rows look identical.
SPEEDUP_TICKS = (1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)


def speedup_axis(values: list[float]) -> tuple[float, float, list[float]]:
    """``(low, high, ticks)`` framing ``values`` on a log axis, with a margin and readable ticks.

    1.0 is NOT pinned to the left edge. It is the meaningful reference -- no change -- but every
    value here is a multiple of it, so anchoring there spends most of the width on empty decades.
    The axis is labelled as a speed-up, which is what tells the reader where no-change sits.
    """
    low, high = min(values), max(values)
    span = (low / 1.35, high * 1.35)
    ticks = [t for t in SPEEDUP_TICKS if span[0] <= t <= span[1]]
    # A range narrow enough to contain fewer than three candidates gets the nearest ones either
    # side instead, so the axis never ships with one lonely tick.
    if len(ticks) < 3:
        ticks = sorted(
            {min(SPEEDUP_TICKS, key=lambda t: abs(t - low)), *ticks,
             min(SPEEDUP_TICKS, key=lambda t: abs(t - high))})
    return span[0], span[1], ticks


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
    drawn = [v for cell in pairs for s in LEGS for v in (value(cell, s), middle(cell, s)) if v is not None]
    low, high, ticks = speedup_axis(drawn)
    ax.set_xlim(low, high)
    ax.set_xticks(ticks)
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

    fig, ax = row_figure(cells, "Successful completion over the llr-focus40 tag, all waves pooled",
                         "kernels ever solved (% of the 40-kernel tag)")
    ax.set_xlim(0.0, 100.0)
    artifact_style.row_axis(ax, [c.label() for c in cells], gridaxis="none")
    for row in range(len(cells)):
        artifact_style.rounded_bar(ax, 0.0, 100.0, row, 0.62, artifact_style.RAISE, zorder=1.0)
    draw_rows(ax, cells, value, note, 0.0)
    ax.set_xlim(0.0, 100.0)
    finish(fig, ax, cells, out)
    print(f"  success: pooled over every wave of each cell -- the numerator is the UNION of distinct kernels "
          f"solved, never a sum of per-wave counts (which double-counts a kernel two completion waves both "
          f"drew), over the llr-focus40 tag of {TAG_SIZE}; the right column is solved/{TAG_SIZE} per leg")


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
