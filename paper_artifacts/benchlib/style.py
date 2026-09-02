"""The one place the paper figures get their look: palette, fonts, rcParams and four idioms.

The palette and the idioms are lifted from the published HTML artifact of this campaign, so the
figures and the web page read as one piece of work. Light only -- these are print figures.

Four idioms, and what each is for:

  bar_row    a full-scale track in ``RAISE`` with the value filled over it. The track is what makes
             a row of bars readable without a second axis: every row shows the same scale, so the
             eye compares fills rather than measuring against tick marks.
  dumbbell   a paired A/B on one row. Hollow marker for the base leg, filled for the other, joined
             by a round-capped connector. Side-by-side bars spend twice the ink and put the two
             legs of a pair further apart than two legs of neighbouring pairs.
  hatch      the "off" leg of a skills pair: 45-degree stripes in the surface colour over the
             series fill, so the condition survives a reader who cannot separate the hues.
  eyebrow    an uppercase tracked section label, in place of a boxed frame or a heavier heading.

Everything else is subtraction: hairlines instead of a box, grid only along the measured axis, no
tick marks, warm off-white rather than pure white.
"""
from __future__ import annotations

import dataclasses
import functools
import pathlib
import typing

import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import matplotlib.font_manager
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
RAISE = "#f5f3ee"
INK = "#1c1b19"
INK2 = "#57544e"
MUTED = "#8b8780"
RULE = "#e3e0d9"
WARN = "#8a5a12"
WARN_BG = "#fbf3e2"

#: Series hue by short name. Four hues, fixed, so a model keeps its colour across every figure.
SERIES = {"qwen": "#3b6fd4", "oss": "#d4772a", "kimi": "#1f9c86", "q38": "#8e2f4f"}

#: Model id as it appears in the CSVs -> series name. Insertion order is the order on every axis.
#: qwen30b keeps its entry and its hue, and appears in no current figure: it ran only in llr8 wave 1,
#: which is excluded as void C data (see the README). The mapping is the record of the assignment,
#: not a claim that the model is in the paper.
MODEL_SERIES = {"qwen30b": "qwen", "oss120b": "oss", "kimi27sglang": "kimi", "qwen38": "q38"}
MODEL_COLOR = {model: SERIES[name] for model, name in MODEL_SERIES.items()}
MODEL_LABEL = {
    "qwen30b": "Qwen3-Coder-30B",
    "oss120b": "GPT-OSS-120B",
    "kimi27sglang": "Kimi K2.7",
    "qwen38": "Qwen3.8",
}

DISPLAY_STACK = ("Fraunces", "Georgia", "DejaVu Serif")
BODY_STACK = ("Inter", "Helvetica Neue", "Arial", "DejaVu Sans")
MONO_STACK = ("JetBrains Mono", "DejaVu Sans Mono")

#: 3 CSS pixels of corner rounding, in points, which is what the artifact's bars use.
CORNER_PT = 2.25

#: A thin space between letters is the only letter-spacing matplotlib has.
TRACK_SPACE = "\u2009"


@dataclasses.dataclass(frozen=True)
class Fonts:
    """The three families that actually resolved on this machine."""

    display: str
    body: str
    mono: str


@functools.lru_cache(maxsize=8, typed=True)
def resolve(stack: tuple[str, ...]) -> str:
    installed = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for name in stack:
        if name in installed:
            return name
    return stack[-1]


@functools.lru_cache(maxsize=1, typed=True)
def fonts() -> Fonts:
    return Fonts(resolve(DISPLAY_STACK), resolve(BODY_STACK), resolve(MONO_STACK))


def apply(dpi: int = 300) -> None:
    """Set every rcParam the look depends on. Idempotent; call once per process."""
    family = fonts()
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": [family.body, *BODY_STACK],
        "font.serif": [family.display, *DISPLAY_STACK],
        "font.monospace": [family.mono, *MONO_STACK],
        "font.size": 8.0,
        "text.color": INK2,
        "axes.labelsize": 8.0,
        "axes.labelcolor": INK2,
        "axes.titlesize": 10.5,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 9.0,
        "axes.titleweight": "bold",
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": RULE,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 5.0,
        "legend.frameon": False,
        "legend.fontsize": 7.0,
        "legend.labelcolor": INK2,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.4,
        "legend.borderpad": 0.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.8,
        "hatch.linewidth": 0.7,
        "hatch.color": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tracked(text: str) -> str:
    """Uppercase with the artifact's 0.13em letter-spacing, faked with thin spaces."""
    return TRACK_SPACE.join(text.upper())


def eyebrow(target: matplotlib.figure.Figure | matplotlib.axes.Axes, x: float, y: float, text: str) -> None:
    target.text(x, y, tracked(text), fontsize=6.5, color=MUTED, ha="left", va="bottom")


def axis_title(ax: matplotlib.axes.Axes, text: str, pad: float = 10.0) -> None:
    """The panel's heading, in the display face, left aligned over the plot area."""
    ax.set_title(text, family="serif", fontsize=10.5, weight="bold", color=INK, loc="left", pad=pad)


def heading(fig: matplotlib.figure.Figure, x: float, y: float, text: str) -> None:
    fig.text(x, y, text, fontsize=11.5, color=INK, family="serif", weight="bold", ha="left", va="top")


def value_label(ax: matplotlib.axes.Axes,
                x: float,
                y: float,
                text: str,
                colour: str = INK,
                ha: str = "right",
                size: float = 7.0) -> None:
    """A number, in mono so a column of them lines up on the digit."""
    ax.text(x, y, text, family="monospace", fontsize=size, color=colour, ha=ha, va="center")


def right_label(ax: matplotlib.axes.Axes,
                row: float,
                text: str,
                colour: str = INK,
                size: float = 7.0,
                offset: float = 7.0) -> None:
    """The row's number, in its own column just outside the axes, the way the artifact sets it."""
    ax.annotate(text,
                xy=(1.0, row),
                xycoords=("axes fraction", "data"),
                xytext=(offset, 0),
                textcoords="offset points",
                family="monospace",
                fontsize=size,
                color=colour,
                ha="left",
                va="center",
                annotation_clip=False)


def paint(patch: matplotlib.patches.Patch, colour: str, on: bool) -> None:
    """Fill one leg of a skills pair: solid for on, hatched for off.

    The stripes are the surface colour rather than a second hue, because a hatch drawn in ink turns
    the fill into a third colour and breaks the "colour is the model" rule the whole set relies on.
    """
    patch.set_facecolor(colour)
    patch.set_edgecolor(SURFACE)
    if not on:
        patch.set_hatch("///")


def data_aspect(ax: matplotlib.axes.Axes) -> tuple[float, float]:
    """Inches per data unit on each axis, from the axes box. Valid once the limits are set."""
    figure = ax.get_figure(root=True)
    assert figure is not None, "an axes always belongs to a figure by the time it is drawn into"
    box = ax.get_position()
    width, height = figure.get_size_inches()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    span_x = abs(x1 - x0) or 1.0
    span_y = abs(y1 - y0) or 1.0
    return box.width * width / span_x, box.height * height / span_y


def rounded_bar(ax: matplotlib.axes.Axes,
                start: float,
                stop: float,
                centre: float,
                height: float,
                colour: str,
                on: bool = True,
                zorder: float = 2.0) -> None:
    """One horizontal bar with the artifact's corner rounding, in data coordinates.

    Rounding is dropped on a log axis: a constant corner radius in data units is not a constant
    radius on screen there, and a corner that grows towards one end reads as a drawing error.
    """
    scale_x, scale_y = data_aspect(ax)
    radius = 0.0 if ax.get_xscale() != "linear" else (CORNER_PT / 72.0) / scale_x
    style = matplotlib.patches.BoxStyle("round", pad=0.0, rounding_size=radius)
    patch = matplotlib.patches.FancyBboxPatch((start, centre - height / 2.0),
                                              max(stop - start, 0.0),
                                              height,
                                              boxstyle=style,
                                              mutation_aspect=scale_x / scale_y,
                                              linewidth=0.0,
                                              zorder=zorder)
    paint(patch, colour, on)
    ax.add_patch(patch)


def bar_row(ax: matplotlib.axes.Axes,
            centre: float,
            value: float,
            colour: str,
            on: bool = True,
            height: float = 0.5,
            track: bool = True) -> None:
    """A value drawn as a fill over a full-scale track, the artifact's bar idiom."""
    left, right = ax.get_xlim()
    if track:
        rounded_bar(ax, left, right, centre, height, RAISE, on=True, zorder=1.0)
    rounded_bar(ax, left, value, centre, height, colour, on=on, zorder=2.0)


def dumbbell(ax: matplotlib.axes.Axes,
             centre: float,
             base: float | None,
             other: float | None,
             colour: str,
             size: float = 5.5) -> None:
    """Paired A/B on one row: hollow marker for ``base``, filled for ``other``.

    Either end may be absent; a lone leg keeps the fill that says which leg it is, so a row with
    one marker still reads as base-only or skills-only rather than as an unlabelled point.
    """
    if base is not None and other is not None:
        ax.plot([base, other], [centre, centre],
                color=colour,
                linewidth=2.5,
                solid_capstyle="round",
                zorder=2.0,
                clip_on=False)
    if base is not None:
        ax.plot([base], [centre],
                marker="o",
                markersize=size,
                markerfacecolor=SURFACE,
                markeredgecolor=MUTED,
                markeredgewidth=2.0,
                linestyle="none",
                zorder=3.0,
                clip_on=False)
    if other is not None:
        ax.plot([other], [centre],
                marker="o",
                markersize=size,
                markerfacecolor=colour,
                markeredgecolor=colour,
                linestyle="none",
                zorder=3.0,
                clip_on=False)


def row_axis(ax: matplotlib.axes.Axes,
             labels: list[str],
             positions: list[float] | None = None,
             gridaxis: typing.Literal["x", "y", "both", "none"] = "x") -> None:
    """Rows top to bottom, names on the left, hairline grid along the measured axis only.

    ``gridaxis="none"`` for a figure whose rows carry a track: the track already shows the scale,
    and a grid drawn over it turns one flat surface into a striped one.
    """
    rows = positions if positions is not None else [float(i) for i in range(len(labels))]
    ax.set_yticks(rows)
    ax.set_yticklabels(labels)
    ax.set_ylim(max(rows) + 0.62, min(rows) - 0.62)
    if gridaxis != "none":
        ax.grid(True, axis=gridaxis, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0.0)


def key(ax: matplotlib.axes.Axes,
        entries: list[tuple[str, matplotlib.artist.Artist]],
        anchor: tuple[float, float] = (1.0, 1.0)) -> None:
    """A frameless key on the title row, where it cannot land on the data."""
    ax.legend([artist for _, artist in entries], [name for name, _ in entries],
              loc="lower right",
              bbox_to_anchor=anchor,
              ncol=len(entries),
              frameon=False)


def swatch(colour: str, on: bool = True) -> matplotlib.patches.Patch:
    patch = matplotlib.patches.Rectangle((0.0, 0.0), 1.0, 1.0, linewidth=0.0)
    paint(patch, colour, on)
    return patch


def marker(colour: str, on: bool = True) -> matplotlib.lines.Line2D:
    return matplotlib.lines.Line2D([], [],
                                   marker="o",
                                   linestyle="none",
                                   markersize=5.5,
                                   markerfacecolor=colour if on else SURFACE,
                                   markeredgecolor=colour if on else MUTED,
                                   markeredgewidth=0.0 if on else 2.0)


def save(fig: matplotlib.figure.Figure, out: pathlib.Path, formats: tuple[str, ...] = ("pdf", "png")) -> pathlib.Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  {out}.{' / .'.join(formats)}")
    return out
