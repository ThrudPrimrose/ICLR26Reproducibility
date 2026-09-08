"""Median speedup over sequential C, one bar per framework, from one canon sweep.

Median is what the bars show, and on its own it would mislead: ``cc_autopar`` and ``numba`` both sit
at exactly 1.00x median while their geometric means are near 1.9x, because each helps a lot on a few
kernels and not at all on more than half. The geomean is therefore drawn as a second mark rather
than left to a caption.

The x axis is logarithmic because the columns span 1x to ~67x; on a linear axis every CPU bar
collapses into the axis line next to the GPU one.

Usage:  python3 plot_canon_speedup.py [--data data/canon_llr40.csv] [--out figures]
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import style  # noqa: E402  -- the artifact is run from a clone, not installed

#: The baseline every speedup is taken against: unmodified C at -O3, one thread.
BASELINE = "cc"

#: Columns on the figure, in axis order, with the label each carries. dace_cpu / dace_gpu (the
#: non-canonicalized DaCe columns) are collected but not drawn here: this figure answers what
#: canonicalization is worth against the compilers, not what DaCe is worth against itself.
DRAW = (
    ("cc", "C -O3 (baseline)"),
    ("cc_autopar", "C -O3 + autopar"),
    ("numba", "Numba"),
    ("dace_cpu_canonicalize", "DaCe canon CPU"),
    ("dace_gpu_canonicalize", "DaCe canon GPU"),
)

MEDIAN_HUE = "#3b6fd4"
GEOMEAN_HUE = "#d4772a"


def read(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """``column -> {kernel: median_ms}``, keeping only validated rows with a positive time.

    A row that did not validate is not a slow result, it is not a result: including it would credit a
    framework for producing the wrong answer quickly.
    """
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["validated"].strip().lower() not in ("true", "1", "yes"):
                continue
            try:
                ms = float(row["median_ms"])
            except ValueError:
                continue
            if ms > 0:
                out[row["column"]][row["kernel"]] = ms
    return out


def speedups(times: dict[str, dict[str, float]], column: str) -> list[float]:
    """Per-kernel baseline/column ratios, over the kernels BOTH measured."""
    base = times.get(BASELINE, {})
    cur = times.get(column, {})
    return [base[k] / cur[k] for k in sorted(base) if k in cur]


def draw(data: pathlib.Path, out_dir: pathlib.Path) -> int:
    import matplotlib.lines
    import matplotlib.patheffects
    import matplotlib.pyplot as plt

    times = read(data)
    if BASELINE not in times:
        print(f"{data} has no '{BASELINE}' column to divide by", file=sys.stderr)
        return 1

    rows = []
    for column, label in DRAW:
        sp = speedups(times, column)
        if not sp:
            continue
        geo = math.exp(statistics.fmean(math.log(s) for s in sp))
        rows.append((label, statistics.median(sp), geo, len(sp)))
    if not rows:
        print(f"{data} holds none of the drawn columns", file=sys.stderr)
        return 1

    style.apply()
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.9))
    ypos = list(range(len(rows)))[::-1]

    # Scale and limits first: rounded_bar reads the axis to size its corners and to find the left
    # edge a bar grows from, and on a log axis that edge cannot be zero.
    ax.set_xscale("log")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.8, max(max(r[1], r[2]) for r in rows) * 2.6)
    left = ax.get_xlim()[0]

    for y, (_label, median, geo, _n) in zip(ypos, rows, strict=True):
        style.rounded_bar(ax, left, median, y, 0.5, MEDIAN_HUE)
        # A tick spanning the bar's height rather than a dot on it. The geomean can fall either side
        # of the median and, where it falls just inside, a dot lands on top of the value label; a
        # tick occupies a channel the label never uses.
        ax.vlines(geo, y - 0.31, y + 0.31, color=GEOMEAN_HUE, linewidth=2.0, zorder=6)
        # Always past the BAR's end, so it labels the bar. Placed past whichever mark sits further
        # right it would read as the geomean's label, and the two are different statistics.
        # Haloed rather than nudged: where the geomean lands just past the bar's end the tick would
        # otherwise cross the digits, and moving the label to clear it would detach it from the bar.
        ax.text(median * 1.09, y, f"{median:.2f}x", va="center", ha="left", zorder=7,
                fontsize=8, color=style.INK2, family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6, foreground=style.SURFACE)])

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{label}  (n={n})" for label, _m, _g, n in rows], fontsize=9, color=style.INK)
    ax.set_xlabel("speedup over sequential C  (log scale, higher is better)", fontsize=8.5, color=style.INK2)
    ax.axvline(1.0, color=style.RULE, linewidth=1.0, zorder=0)
    ax.grid(axis="x", color=style.RULE, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(style.RULE)
    ax.tick_params(axis="both", length=0, colors=style.INK2, labelsize=8)

    # Below the axis label, not inside the axes and not above them: the longest bar reaches the right
    # edge, and the title already occupies the strip above.
    geomean_key = matplotlib.lines.Line2D([], [], color=GEOMEAN_HUE, linewidth=2.0, marker="none")
    ax.legend(handles=[style.swatch(MEDIAN_HUE), geomean_key],
              labels=["median speedup", "geometric mean"],
              loc="upper right", bbox_to_anchor=(1.0, -0.13), ncols=2,
              frameon=False, fontsize=8, handletextpad=0.6, columnspacing=1.6)
    style.axis_title(ax, "Canonicalization against the compilers, llr-focus40")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = style.save(fig, out_dir / "canon_speedup")
    print(f"{written}")
    for label, median, geo, n in rows:
        print(f"  {label:<24} median {median:>7.2f}x   geomean {geo:>7.2f}x   n={n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=here / "data" / "canon_llr40.csv")
    ap.add_argument("--out", type=pathlib.Path, default=here / "figures")
    args = ap.parse_args(argv)
    return draw(args.data, args.out)


if __name__ == "__main__":
    sys.exit(main())
