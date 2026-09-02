"""Three llr-focus40 figures from the pooled kernel table, one row per (model, language) cell.

The pooling rules are NOT here. ``kernels.py`` owns them and writes ``llr40_kernels.csv``;
this script reads that table and draws it. The three decisions the figures depend on are worth
restating because they are what the rows mean:

  THE UNIT IS THE KERNEL, not the submission row, so an arm that resubmits one flat kernel eleven
  times does not get eleven votes.
  THE VALUE IS THE LAST SUBMISSION, the one the agent stood behind. The BEST per kernel is printed
  beside it on stdout, never drawn: a cell whose two diverge was carried by cherry-picking.
  THE ROW POOLS EVERY WAVE of one cell, because a wave is not an experiment -- most llr8 waves are
  completion runs over the kernels an earlier arm never submitted.

WHY GEOMEAN. Speed-up is a ratio, so the mean that matches the product over the set is the
geometric one, and the median of a mostly-flat distribution reports "no effect" everywhere.

THE FORM. A DUMBBELL row per cell -- hollow marker for the base leg, filled for the skills leg,
joined in the model's colour. Grouped vertical bars would need rotated labels and would put the two
legs of a pair further apart than two legs of neighbouring pairs.

Driven by each experiment's ``plot_<name>_kernels.py``, which supplies its paths and tag size.
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import math
import pathlib
from collections.abc import Callable

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

from benchlib import style  # noqa: E402  -- ditto: it pulls matplotlib in itself

MODEL_COLOR = style.MODEL_COLOR

#: The two legs: the "0" one first, so the hollow marker is always the one the pair started from.
LEGS = ("0", "1")


@dataclasses.dataclass(frozen=True)
class Contrast:
    """What the two legs MEAN in the experiment being drawn.

    The FORM is the same wherever this module is used -- hollow marker, filled marker, joined in
    the model's colour -- but the contrast is not: llr8 and llr9 put a skills packet against its
    absence, the git experiment puts two framings of the same task against each other. Leaving the
    words hard-coded meant a second experiment had to copy the drawing to change a legend, and a
    copied figure drifts from the original in ways a reader cannot see.
    """

    #: Legend text, hollow leg then filled leg.
    key_labels: tuple[str, str] = ("base", "+ skills")
    #: The same two legs as stdout names, where a leading "+" would read as arithmetic.
    names: tuple[str, str] = ("base", "skills")
    #: Trailing clause for the speed-up and token titles; must read after "Speed-up ".
    clause: str = "with and without the skills packet"
    #: What the success figure's denominator is called when the count is explained on stdout.
    tag_label: str = "llr-focus40"


SKILLS = Contrast()


@dataclasses.dataclass(frozen=True)
class Cell:
    """One (model, language) cell: its two skills legs, each a kernel table from the tidy CSV."""

    model: str
    language: str
    legs: dict[str, dict[str, dict[str, str]]]

    def label(self) -> str:
        language = "C" if self.language == "c" else "Fortran"
        return f"{style.MODEL_LABEL[self.model]} / {language}"

    def shared(self) -> list[str]:
        """Kernels BOTH legs have a last submission for; empty when the pair is unpaired."""
        if len(self.legs) < 2:
            return []
        timed = [{k for k, row in leg.items() if row["last_speedup"]} for leg in self.legs.values()]
        return sorted(timed[0] & timed[1])

    def speedups(self, skills: str, column: str = "last_speedup") -> list[float]:
        """This leg's per-kernel speed-ups, narrowed to the shared kernels when there are any."""
        leg = self.legs.get(skills, {})
        keep = self.shared() or sorted(k for k, row in leg.items() if row[column])
        return [float(leg[k][column]) for k in keep if k in leg and leg[k][column]]


def read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def geomean(values: list[float]) -> float:
    """Geometric mean, or NaN for an empty set. Non-positive entries never reach here."""
    if not values:
        return math.nan
    return math.exp(sum(math.log(v) for v in values) / len(values))


def cells(rows: list[dict[str, str]]) -> list[Cell]:
    """One cell per (model, language), ordered by the palette's model order then by language.

    The order is fixed rather than by value so a hue never tracks rank and a cell sits in the same
    place in all three figures, which is what makes them readable side by side.
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, str]]]] = collections.defaultdict(dict)
    for row in rows:
        leg = grouped[(row["model"], row["language"])].setdefault(row["skills"], {})
        leg[row["benchmark"]] = row
    order = sorted(grouped, key=lambda k: (list(MODEL_COLOR).index(k[0]), k[1]))
    return [Cell(model, language, grouped[(model, language)]) for model, language in order]


def row_figure(rows: int, title: str, xlabel: str) -> tuple:
    tall = 1.15 + 0.29 * rows
    fig, ax = plt.subplots(figsize=(7.0, tall))
    fig.subplots_adjust(left=0.315, right=0.845, top=1.0 - 0.72 / tall, bottom=0.62 / tall)
    style.axis_title(ax, title, pad=18.0)
    ax.set_xlabel(xlabel)
    return fig, ax


def usable(value: float, floor: float) -> float | None:
    """A leg is drawn only where it has a number to draw; NaN and a value at the floor are not."""
    return None if math.isnan(value) or value <= floor else value


def draw_rows(ax, drawn: list[Cell], value: Callable[[Cell, str], float | None], note: Callable[[Cell], str],
              floor: float) -> None:
    for row, cell in enumerate(drawn):
        legs = {skills: value(cell, skills) if skills in cell.legs else None for skills in LEGS}
        if legs["0"] is None and legs["1"] is None:
            style.right_label(ax, row, "no timed arm", style.MUTED, size=6.5)
            continue
        style.dumbbell(ax, row, legs["0"], legs["1"], MODEL_COLOR[cell.model])
        style.right_label(ax, row, note(cell))
    ax.set_xlim(left=floor)


def finish(fig, ax, drawn: list[Cell], out: pathlib.Path, contrast: Contrast, unpaired: bool = False) -> None:
    if unpaired:
        for tick, cell in zip(ax.get_yticklabels(), drawn, strict=True):
            if not cell.shared():
                tick.set_color(style.WARN)
    style.key(ax, [(contrast.key_labels[0], style.marker(style.INK2, on=False)),
                   (contrast.key_labels[1], style.marker(style.INK2, on=True))],
              anchor=(1.0, 1.005))
    style.save(fig, out)


#: Tick candidates for the speed-up axis, in the order a reader expects to see them on a log scale.
#: The set is filtered to whatever range the data occupies rather than fixed, because a fixed
#: 1..100 ladder sized for one outlier squeezes a pooled 5x-to-12x campaign into a third of the axis.
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


def speedup_figure(drawn: list[Cell], out: pathlib.Path, contrast: Contrast) -> None:
    """Per-kernel geomean of the LAST submission, base leg against skills leg, one row per pair."""
    pairs = [c for c in drawn if len(c.legs) == 2]
    lonely = [c for c in drawn if len(c.legs) == 1]

    def value(cell: Cell, skills: str) -> float | None:
        return usable(geomean(cell.speedups(skills)), 1.0)

    def best(cell: Cell, skills: str) -> float | None:
        return usable(geomean(cell.speedups(skills, "best_speedup")), 1.0)

    def note(cell: Cell) -> str:
        return "n=" + "/".join(str(len(cell.speedups(s))) for s in LEGS)

    labels = [c.label() + ("  UNPAIRED" if not c.shared() else "") for c in pairs]
    fig, ax = row_figure(len(pairs), f"Speed-up {contrast.clause}, aggregated per kernel",
                         "geomean of each kernel's last submission (x, log scale)")
    # Log: a speed-up is a ratio, and one group at 70x flattens every 3x-to-8x group into the floor
    # on a linear scale.
    ax.set_xscale("log")
    style.row_axis(ax, labels, gridaxis="none")
    draw_rows(ax, pairs, value, note, 1.0)
    plotted = [v for cell in pairs for s in LEGS if (v := value(cell, s)) is not None]
    low, high, ticks = speedup_axis(plotted)
    ax.set_xlim(low, high)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x")
    finish(fig, ax, pairs, out, contrast, unpaired=True)
    # BEST is reported rather than drawn. What it is for: a cell whose last and best diverge was
    # carried by lucky resubmissions rather than by a repeatable win, so the ratio is the thing to
    # look at. Drawn as a second mark it was two identical ticks per row with nothing tying either
    # to its leg, and an undecodable mark is worse than a number.
    for cell in pairs:
        for skills, leg in zip(LEGS, contrast.names, strict=True):
            last, top = value(cell, skills), best(cell, skills)
            if last is not None and top is not None:
                print(f"  speedup: {cell.label()} {leg}: last {last:.2f}x, best {top:.2f}x "
                      f"(best/last {top / last:.3f})")
    print(f"  speedup: {sum(1 for c in pairs if not c.shared())} of {len(pairs)} pairs share no timed kernel "
          f"and are marked UNPAIRED")
    if lonely:
        print(f"  speedup: {len(lonely)} cells have no {contrast.names[1]} counterpart and are omitted: "
              f"{', '.join(c.label() for c in lonely)}")


def success_figure(drawn: list[Cell], out: pathlib.Path, tag_size: int, contrast: Contrast) -> None:
    """Solved kernels over the llr-focus40 tag, every wave of the cell pooled."""

    def value(cell: Cell, skills: str) -> float | None:
        solved = sum(int(row["solved"]) for row in cell.legs[skills].values())
        return 100.0 * solved / tag_size

    def note(cell: Cell) -> str:
        return " ".join(f"{sum(int(r['solved']) for r in cell.legs[s].values())}/{tag_size}" for s in LEGS
                        if s in cell.legs)

    fig, ax = row_figure(len(drawn), "Successful completion over the tag, all waves pooled",
                         f"kernels ever solved (% of the {tag_size}-kernel tag)")
    ax.set_xlim(0.0, 100.0)
    style.row_axis(ax, [c.label() for c in drawn], gridaxis="none")
    for row in range(len(drawn)):
        style.rounded_bar(ax, 0.0, 100.0, row, 0.62, style.RAISE, zorder=1.0)
    draw_rows(ax, drawn, value, note, 0.0)
    ax.set_xlim(0.0, 100.0)
    finish(fig, ax, drawn, out, contrast)
    print(f"  success: the numerator is the UNION of distinct kernels solved over every wave of the cell, never a "
          f"sum of per-wave counts (which double-counts a kernel two completion waves both drew), over the "
          f"{contrast.tag_label} tag of {tag_size}")


def tokens_figure(drawn: list[Cell], out: pathlib.Path, contrast: Contrast) -> None:
    """Token cost of one kernel, first agent turn to last grade -- the term the packet moves."""

    def value(cell: Cell, skills: str) -> float | None:
        leg = cell.legs[skills]
        return sum(int(row["tokens"]) for row in leg.values()) / len(leg) / 1e6 if leg else None

    def note(cell: Cell) -> str:
        return "n=" + "/".join(str(len(cell.legs[s])) for s in LEGS if s in cell.legs)

    fig, ax = row_figure(len(drawn), f"Token cost per kernel, {contrast.clause}",
                         "million tokens per kernel reached")
    style.row_axis(ax, [c.label() for c in drawn])
    draw_rows(ax, drawn, value, note, 0.0)
    ax.margins(x=0.1)
    finish(fig, ax, drawn, out, contrast)
    print("  tokens: the agent's cumulative usage at its last judge call on a kernel, summed over the waves that "
          "drew it and over the kernels the arm reached, divided by that count; n = kernels reached")


def main(data: pathlib.Path, out: pathlib.Path, prefix: str, tag_size: int,
         contrast: Contrast = SKILLS) -> int:
    """Draw the three figures for one experiment: ``<out>/<prefix>_<figure>.{png,pdf}``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=data, help="tidy kernel table")
    parser.add_argument("--out", type=pathlib.Path, default=out, help="figure directory")
    args = parser.parse_args()
    style.apply()
    drawn = cells(read(args.data))
    speedup_figure(drawn, args.out / f"{prefix}_speedup_per_kernel", contrast)
    success_figure(drawn, args.out / f"{prefix}_success_rate", tag_size, contrast)
    tokens_figure(drawn, args.out / f"{prefix}_tokens_per_kernel", contrast)
    return 0
