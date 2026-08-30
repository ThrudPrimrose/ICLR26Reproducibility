"""Three llr-focus40 figures across every collected wave, aggregated PER KERNEL.

The unit of measurement is the KERNEL, not the submission row. ``summary.csv``'s ``speedup_geomean``
is taken over submission rows, so it is weighted by how often an agent pressed submit: llr8w6's
unskilled qwen38 C arm has 15 rows over 3 kernels, 11 of them one kernel at 1.00x, and reads 2.08x
per row against 6.89x per kernel. Everything here takes each kernel's BEST timed submission and the
geometric mean of those, and quotes the geomean as the headline -- speed-up is a ratio, so the mean
that matches the product over the set is the geometric one, and the median of a mostly-flat
distribution reports "no effect" everywhere. The per-kernel MEDIAN variant is drawn as a tick on
each bar, so a bar carried by one lucky resubmission shows as a wide gap between the two.

The ARM is the row, never a pool across waves. Waves 3, 4, 6 and 7 are COMPLETION waves that re-ran
only the kernels an earlier arm never submitted, so an arm's kernels are a subset computed from what
its predecessor failed at; summing them into a per-model total would report a denominator no arm
ever drew from.

Only PAIRED comparisons say anything about skills. The two legs of a completion wave drew subsets
computed from DIFFERENT predecessors and can share almost nothing -- llr8w6's qwen38 pair shares one
reached kernel and none with a timed submission on both sides. Where a pair shares timed kernels,
both bars are restricted to those. Where it shares none, both bars are drawn over each arm's own
kernels and the group is marked UNPAIRED, which is a statement that the two bars are not comparable
rather than a comparison.

Colour identifies the model and nothing else; the skills condition is carried by fill and hatch, so
the comparison survives a reader who cannot separate the hues.

Usage:  python3 plot_llr40.py
"""
from __future__ import annotations

import csv
import dataclasses
import pathlib
import sys
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

import collect  # noqa: E402  -- ditto: these pull matplotlib in themselves
import collect_llr40  # noqa: E402

#: Model -> hue, from plot_llr8w2. Blue/orange/green separate under all three common colour vision
#: deficiencies, which a red/green pair does not.
MODEL_COLOR = {"qwen38": "#2a78d6", "oss120b": "#eb6834", "kimi27sglang": "#1b8a5a"}
INK, MUTED, GRID, WARN = "#1a1a1a", "#5c5c5c", "#d8d8d8", "#a01818"

#: The two skills legs, in the order they are drawn: off on the left of each group, on to the right.
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
        return f"{self.model}\n{language}  w{self.wave[len('llr8w'):]}"


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 300,
    })


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


def bar_pair(ax,
             index: int,
             cell: Group,
             value: Callable[[Group, str], float],
             note: Callable[[Group, str], str],
             base: float = 0.0) -> None:
    """The group's two bars: hollow and hatched for skills off, solid for skills on.

    ``value`` and ``note`` are passed in so the three figures share this geometry and differ only in
    what they measure, which is the only way the groups stay in one order across them. ``base`` is
    where a bar starts: 0 for a count, 1.0 for a RATIO, where the interesting quantity is the
    distance from "unchanged" and a bar rising from zero makes 1.05x look like most of a result.
    """
    width = 0.36
    colour = MODEL_COLOR.get(cell.model, MUTED)
    for offset, skills in ((-width / 2, LEGS[0]), (width / 2, LEGS[1])):
        if skills not in cell.arms:
            # A missing leg is a gap with a word in it, never a zero bar: zero is a measurement.
            ax.text(index + offset, base, "no arm", ha="center", va="bottom", fontsize=6.5, color=MUTED, style="italic")
            continue
        height = value(cell, skills)
        if height <= base:
            ax.text(index + offset, base, note(cell, skills), ha="center", va="bottom", fontsize=6.5, color=MUTED)
            continue
        ax.bar(index + offset,
               height - base,
               width,
               bottom=base,
               color=colour if skills == "1" else "white",
               edgecolor=colour,
               linewidth=1.3,
               hatch=None if skills == "1" else "///",
               zorder=3)
        # Upright: 19 groups on one axis put two of these within a few points of each other, and
        # horizontal labels overlap into an unreadable smear at exactly the pairs a reader compares.
        ax.text(index + offset,
                height,
                note(cell, skills),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
                rotation=90)


def frame(ax, labels: list[str], ylabel: str, title: str) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=22)
    ax.margins(y=0.30)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    fill = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="///", linewidth=1.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=INK, linewidth=1.3),
    ]
    hues = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=c) for c in MODEL_COLOR.values()]
    # On the title row rather than inside the axes: these bars are tall and irregular, and an
    # in-axes legend lands on whichever group happens to be the tallest.
    ax.add_artist(
        ax.legend(fill, ["skills off", "skills on"],
                  frameon=False,
                  loc="lower left",
                  bbox_to_anchor=(0.0, 1.0),
                  ncol=2,
                  fontsize=8))
    ax.legend(hues, list(MODEL_COLOR), frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8)


def save(fig, out: pathlib.Path, footer: str) -> None:
    fig.text(0.006, 0.006, footer, fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
    print(f"  {out}.pdf / .png")


def speedup_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Per-kernel geomean speed-up, skills off against on, one group per arm PAIR."""
    pairs = [c for c in cells if len(c.arms) == 2]
    lonely = [c for c in cells if len(c.arms) == 1]

    def value(cell: Group, skills: str) -> float:
        return collect.geomean([max(v) for v in cell.timed(skills)])

    def middle(cell: Group, skills: str) -> float:
        return collect.geomean([v[len(v) // 2] for v in cell.timed(skills)])

    def note(cell: Group, skills: str) -> str:
        return f"n={len(cell.timed(skills))}"

    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    # Log, and bars rising from 1.0: a speed-up is a ratio, and one n=2 group at 71x flattens every
    # 3x-to-8x group into the axis floor on a linear scale.
    ax.set_yscale("log")
    for index, cell in enumerate(pairs):
        bar_pair(ax, index, cell, value, note, base=1.0)
        # The per-kernel MEDIAN geomean as a tick inside its own bar. A bar far above its tick is an
        # arm whose speed-up rests on a best-of-many resubmission rather than a repeatable win.
        for offset, skills in ((-0.18, LEGS[0]), (0.18, LEGS[1])):
            if middle(cell, skills) > 1.0:
                ax.plot([index + offset - 0.16, index + offset + 0.16], [middle(cell, skills)] * 2,
                        color=INK,
                        linewidth=1.0,
                        zorder=4)
    frame(ax, [c.label() + ("\nUNPAIRED" if not c.shared() else "") for c in pairs],
          "geomean of per-kernel best speed-up ($\\times$, log scale)",
          "Speed-up with and without the skills packet, aggregated per kernel")
    ax.set_ylim(bottom=1.0)
    for tick, cell in zip(ax.get_xticklabels(), pairs, strict=True):
        if not cell.shared():
            tick.set_color(WARN)
    save(
        fig, out, "Bars: geomean over kernels of each kernel's BEST timed speed-up. Tick: the same geomean over "
        "each kernel's MEDIAN. n = kernels behind the bar. Paired groups are restricted to the kernels BOTH "
        "arms timed; an UNPAIRED group shares none, so its two bars measure different kernel sets and are not "
        f"a comparison, and a paired group at n<3 shared kernels is annotated but is not one either. "
        f"{len(lonely)} arms have no skills counterpart and are omitted: "
        f"{', '.join(c.label().replace(chr(10), ' ') for c in lonely)}.")


def success_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Solved kernels over the set the arm DREW FROM, which is not 40 for a completion-wave arm."""

    def value(cell: Group, skills: str) -> float:
        arm = cell.arms[skills]
        return 100.0 * int(arm["solved"]) / int(arm["problems"])

    def note(cell: Group, skills: str) -> str:
        arm = cell.arms[skills]
        return f"{arm['solved']}/{arm['problems']}"

    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    for index, cell in enumerate(cells):
        bar_pair(ax, index, cell, value, note)
    frame(ax, [c.label() for c in cells], "successful completion rate (% of kernels drawn)",
          "Successful completion, against the kernel set each arm drew from")
    save(
        fig, out, "Denominator is collect.problem_count(): 40 for a full arm, the computed gap list for a "
        "completion-wave arm, 20 for a kimi Fortran batch, never a hard-coded 40. The label above each bar "
        "is solved / drawn.")


def tokens_figure(cells: list[Group], out: pathlib.Path) -> None:
    """Token cost of one kernel, first agent turn to last grade -- the term the packet moves."""

    def value(cell: Group, skills: str) -> float:
        return int(cell.arms[skills]["tokens_per_kernel"]) / 1e6

    def note(cell: Group, skills: str) -> str:
        return f"n={cell.arms[skills]['attempted']}"

    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    for index, cell in enumerate(cells):
        bar_pair(ax, index, cell, value, note)
    frame(ax, [c.label() for c in cells], "million tokens per kernel reached",
          "Token cost per kernel, with and without the skills packet")
    save(
        fig, out, "Tokens are the agent's cumulative usage at its last judge call on a kernel, summed over the "
        "kernels the arm reached and divided by that count. n = kernels reached.")


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
