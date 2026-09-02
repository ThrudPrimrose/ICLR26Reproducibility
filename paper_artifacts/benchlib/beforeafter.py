"""What the skills packet did to each KERNEL, before against after, as connected pairs.

The dumbbell figure reports one geomean per cell. Underneath a 5.9x -> 7.8x sit nineteen kernels
that got faster and fourteen that got slower, and the two claims are not the same: a packet that
lifts every kernel a little and a packet that trades half the set against the other half both come
out at the same geomean. This figure draws the pairs, so the churn is visible.

THE PAIR IS SKILLS-OFF AGAINST SKILLS-ON, per kernel, which is the campaign's treatment and the
only pairing the data supports end to end. First-against-last submission would be the other
interesting reading and it is a different figure: it needs the trajectory, not the two arms.

TWO ROWS, ONE COLUMN PER CELL, because the question the two metrics only pose together is whether
paying more bought more. Reading down a column answers it for one model and language.

THE ENCODING. Colour is the model, as everywhere else in this set. Within a panel every line is the
same model, so the hue carries no model information there and is free to carry the sign: a kernel
the packet made WORSE -- slower, or dearer -- is drawn in grey. Grey against a hue separates by
lightness, so the direction survives a reader who cannot separate the hues, which is the same
argument the hatch idiom makes for the bars. The cell's geomean is drawn over the cloud as the
dumbbell this set already uses: hollow marker on the base leg, filled on the skills leg.

Each panel carries its own n, because the pairing depth is not 40 everywhere and a reader who
assumes it is will over-read the thin cells.

Driven by each experiment's ``plot_<name>_before_after.py``, which supplies its paths.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

from benchlib import style  # noqa: E402  -- ditto, it imports pyplot itself
from benchlib import dumbbell  # noqa: E402

#: The two x positions of a panel, and what they are called under it.
LEG_X = {"0": 0.0, "1": 1.0}
LEG_NAME = ("without", "with")

#: A change smaller than this is the same measurement twice, not a direction. 1% is the harness's
#: own dead band on the paired sign test in plot_canon_vs.
DEAD_BAND = 1.01


@dataclasses.dataclass(frozen=True)
class Metric:
    """One row of panels: what to read off a kernel, and which direction is an improvement."""

    name: str
    title: str
    axis: str
    log: bool
    higher_is_better: bool
    scale: float = 1.0

    def value(self, row: dict[str, str]) -> float | None:
        text = row[self.name]
        return float(text) / self.scale if text else None


SPEEDUP = Metric("last_speedup", "Speed-up per kernel", "last submission (x, log)", True, True)
# Log for the cost too: one 50M kernel against a 0.3M median puts every other line in the bottom
# tenth of a linear axis, and the summary drawn over them is a geomean either way.
TOKENS = Metric("tokens", "Token cost per kernel", "million tokens (log)", True, False, scale=1e6)


def paired(cell: dumbbell.Cell, metric: Metric) -> dict[str, tuple[float, float]]:
    """``kernel -> (without, with)`` over the kernels BOTH legs have a value for.

    A kernel only one leg produced is not a pair and is dropped in both directions; keeping it would
    let an arm that never reached a kernel look like an arm that reached it and did nothing.
    """
    if len(cell.legs) < 2:
        return {}
    off, on = cell.legs["0"], cell.legs["1"]
    out: dict[str, tuple[float, float]] = {}
    for kernel in sorted(set(off) & set(on)):
        before, after = metric.value(off[kernel]), metric.value(on[kernel])
        if before is not None and after is not None and before > 0 and after > 0:
            out[kernel] = (before, after)
    return out


def worse(before: float, after: float, metric: Metric) -> bool:
    """Did the packet make this kernel worse, outside the dead band?"""
    ratio = after / before if metric.higher_is_better else before / after
    return ratio < 1.0 / DEAD_BAND


def tally(pairs: dict[str, tuple[float, float]], metric: Metric) -> tuple[int, int, int]:
    """``(better, worse, unchanged)`` at the dead band."""
    down = sum(1 for before, after in pairs.values() if worse(before, after, metric))
    up = sum(1 for before, after in pairs.values() if worse(after, before, metric))
    return up, down, len(pairs) - up - down


def draw_panel(ax, cell: dumbbell.Cell, metric: Metric) -> tuple[int, int, int]:
    """One cell's pairs, the geomean dumbbell over them; returns the tally for the annotation."""
    pairs = paired(cell, metric)
    colour = style.MODEL_COLOR[cell.model]
    for before, after in pairs.values():
        ax.plot([LEG_X["0"], LEG_X["1"]], [before, after],
                color=style.MUTED if worse(before, after, metric) else colour,
                linewidth=0.6,
                alpha=0.45,
                solid_capstyle="round",
                zorder=2)
    if pairs:
        # The dumbbell idiom, drawn out by hand because the pair runs vertically here: the helper in
        # style puts the two legs on one ROW, which is the same treatment turned 90 degrees.
        centre = [dumbbell.geomean([v[i] for v in pairs.values()]) for i in (0, 1)]
        ax.plot(list(LEG_X.values()), centre, color=colour, linewidth=2.5, solid_capstyle="round", zorder=4)
        ax.plot([LEG_X["0"]], [centre[0]],
                marker="o",
                markersize=5.5,
                markerfacecolor=style.SURFACE,
                markeredgecolor=style.MUTED,
                markeredgewidth=2.0,
                linestyle="none",
                zorder=5)
        ax.plot([LEG_X["1"]], [centre[1]], marker="o", markersize=5.5, color=colour, linestyle="none", zorder=5)
    return tally(pairs, metric)


def panel_axis(ax, metric: Metric, first: bool) -> None:
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks(list(LEG_X.values()))
    ax.set_xticklabels(LEG_NAME)
    if metric.log:
        ax.set_yscale("log")
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="y", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(first)
    ax.spines["bottom"].set_color(style.RULE)
    ax.tick_params(axis="both", length=0.0)
    if not first:
        ax.tick_params(axis="y", labelleft=False)


def render(cells: list[dumbbell.Cell], out: pathlib.Path) -> pathlib.Path:
    """The whole figure: two metric rows over one column per cell."""
    metrics = (SPEEDUP, TOKENS)
    fig, axes = plt.subplots(len(metrics),
                             len(cells),
                             figsize=(7.0, 4.6),
                             sharey="row",
                             squeeze=False,
                             gridspec_kw={
                                 "hspace": 0.75,
                                 "wspace": 0.18
                             })
    fig.subplots_adjust(left=0.085, right=0.99, top=0.745, bottom=0.105)
    style.heading(fig, 0.06, 0.975, "What the skills packet did to each kernel")
    fig.text(0.06,
             0.925,
             "One line per kernel: its value without the packet, joined to its value with it. Grey marks a kernel\n"
             "the packet made worse. The heavy line is the cell's geomean, hollow without and filled with.",
             fontsize=7.0,
             color=style.INK2,
             va="top",
             linespacing=1.55)
    for column, cell in enumerate(cells):
        box = axes[0, column].get_position()
        axes[0, column].set_title("C" if cell.language == "c" else "Fortran",
                                  fontsize=8.0,
                                  color=style.INK,
                                  loc="center",
                                  pad=5.0)
        # The model name spans the two language columns it owns, so it is written once over the pair
        # rather than repeated and clipped in each panel.
        if cell.language == "c":
            other = axes[0, column + 1].get_position() if column + 1 < len(cells) else box
            fig.text((box.x0 + other.x1) / 2.0,
                     box.y1 + 0.038,
                     style.tracked(style.MODEL_LABEL[cell.model]),
                     fontsize=6.5,
                     color=style.MUTED,
                     ha="center",
                     va="bottom")
        for row, metric in enumerate(metrics):
            ax = axes[row, column]
            panel_axis(ax, metric, first=column == 0)
            if len(cell.legs) < 2:
                ax.text(0.5, 0.5, "no skills arm", transform=ax.transAxes, ha="center", color=style.MUTED)
                continue
            up, down, flat = draw_panel(ax, cell, metric)
            ax.annotate(f"n={up + down + flat}  {up} up  {down} dn",
                        xy=(0.5, -0.235),
                        xycoords="axes fraction",
                        family="monospace",
                        fontsize=5.6,
                        color=style.MUTED,
                        ha="center",
                        annotation_clip=False)
    for row, metric in enumerate(metrics):
        axes[row, 0].set_ylabel(metric.axis)
        # A pair that ends on the axis floor is a real reading; the margin keeps its marker on paper.
        low, high = axes[row, 0].get_ylim()
        axes[row, 0].set_ylim(low / 1.5, high * 1.5)
    style.save(fig, out)
    return out


def report(cells: list[dumbbell.Cell]) -> None:
    for metric in (SPEEDUP, TOKENS):
        for cell in cells:
            pairs = paired(cell, metric)
            if not pairs:
                continue
            up, down, flat = tally(pairs, metric)
            before = dumbbell.geomean([v[0] for v in pairs.values()])
            after = dumbbell.geomean([v[1] for v in pairs.values()])
            print(f"  {metric.name:13s} {cell.label():24s} n={len(pairs):3d} better={up:3d} worse={down:3d} "
                  f"same={flat:3d} geomean {before:7.3f} -> {after:7.3f} ({after / before:.3f}x)")


def main(data: pathlib.Path, out: pathlib.Path, prefix: str) -> int:
    """Draw the connected before/after scatter for one experiment: ``<out>/<prefix>_before_after``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=data, help="tidy kernel table")
    parser.add_argument("--out", type=pathlib.Path, default=out, help="figure directory")
    args = parser.parse_args()
    style.apply()
    cells = [c for c in dumbbell.cells(dumbbell.read(args.data)) if len(c.legs) == 2]
    if not cells:
        raise SystemExit(f"no cell in {args.data} has both skills legs; there is no before-and-after to draw")
    render(cells, args.out / f"{prefix}_before_after")
    report(cells)
    return 0
