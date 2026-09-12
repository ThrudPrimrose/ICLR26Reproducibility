"""Median speedup over the track baseline, one bar per framework, from one canon sweep.

Median is what the bars show, and on its own it would mislead: ``cc_autopar`` and ``numba`` both sit
at exactly 1.00x median while their geometric means are near 1.9x, because each helps a lot on a few
kernels and not at all on more than half. The geomean is therefore drawn as a second mark rather
than left to a caption.

Both statistics are taken over KERNELS and both carry an interval (SC15 Rules 5 and 7): the median a
percentile bootstrap, the geomean the log-t interval. The table beside the figure holds the two
median times every ratio is a quotient of (Rule 4). A statistic is not an entity, so its marks take
neutral ink rather than a hue some model or packet wears in another figure.

The x axis is logarithmic because the columns span 1x to ~67x; on a linear axis every CPU bar
collapses into the axis line next to the GPU one.

Usage:  python3 plot_canon_speedup.py [--data data/canon_llr40.csv] [--out figures] [--table data/canon_speedup.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import statistics
import sys

import numpy as np
import pandas as pd

from hpcagent_bench.stats import rules, style, summary

BASELINE = "numba"

#: Columns on the figure, in axis order, with the label each carries. dace_cpu / dace_gpu (the
#: non-canonicalized DaCe columns) are collected but not drawn here: this figure answers what
#: canonicalization is worth against the compilers, not what DaCe is worth against itself.
#: How each column is named in prose, for the axis label. Keyed off the same names DRAW uses.
LABEL: dict[str, str] = {"cc": "sequential C", "numba": "Numba", "cc_autopar": "C -O3 + autopar"}

DRAW = (
    ("cc", "C -O3, one thread"),
    ("cc_autopar", "C -O3 + autopar"),
    ("numba", "Numba (baseline)"),
    ("dace_cpu_canonicalize", "DaCe canon CPU"),
    ("dace_gpu_canonicalize", "DaCe canon GPU"),
)

#: The bar is the median; the tick and the bracket above it are the geomean and its interval.
MEDIAN_INK = style.FAINT
GEOMEAN_INK = style.INK


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


def summarize(times: dict[str, dict[str, float]]) -> pd.DataFrame:
    """One row per drawn column over the kernels it and the baseline BOTH measured: the median and the
    geomean of the per-kernel ratios, each with its interval, and the two median times behind them."""
    base = times.get(BASELINE, {})
    rows = []
    for column, label in DRAW:
        current = times.get(column, {})
        kernels = [k for k in sorted(base) if k in current]
        if not kernels:
            continue
        ratios = [base[k] / current[k] for k in kernels]
        median = summary.bootstrap_ci(ratios, np.median, "median", method="percentile")
        geomean = summary.geomean_ci(ratios)
        rows.append({
            "column": column,
            "label": label,
            "n": len(ratios),
            "median": median.point,
            "median_low": median.low,
            "median_high": median.high,
            "geomean": geomean.point,
            "geomean_low": geomean.low,
            "geomean_high": geomean.high,
            "baseline_median_ms": statistics.median(base[k] for k in kernels),
            "column_median_ms": statistics.median(current[k] for k in kernels),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    rules.require_costs(frame, "median", ["baseline_median_ms", "column_median_ms"])
    rules.require_interval(frame, "median", "median_low", "median_high")
    return rules.require_interval(frame, "geomean", "geomean_low", "geomean_high")


def draw(data: pathlib.Path, out_dir: pathlib.Path, table: pathlib.Path) -> int:
    import matplotlib.lines
    import matplotlib.patches
    import matplotlib.patheffects
    import matplotlib.pyplot as plt

    times = read(data)
    if BASELINE not in times:
        print(f"{data} has no '{BASELINE}' column to divide by", file=sys.stderr)
        return 1
    rows = summarize(times)
    if rows.empty:
        print(f"{data} holds none of the drawn columns", file=sys.stderr)
        return 1
    table.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(table, index=False)

    style.apply()
    fig, ax = plt.subplots(figsize=(10.0, 0.8 * len(rows) + 2.6))
    ypos = list(range(len(rows)))[::-1]
    # Scale and limits first: a bar grows from the left edge, and on a log axis that edge is not zero.
    ax.set_xscale("log")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.8, float(max(rows.median_high.max(), rows.geomean_high.max())) * 2.6)
    left = ax.get_xlim()[0]

    for y, row in zip(ypos, rows.itertuples(index=False), strict=True):
        ax.barh(y, row.median - left, left=left, height=0.5, color=MEDIAN_INK, zorder=2)
        ax.hlines(y, row.median_low, row.median_high, color=style.INK, linewidth=1.2, zorder=5)
        # A tick spanning the bar's height rather than a dot on it: the geomean can fall either side of
        # the median, and a dot just inside the bar lands on the value label.
        ax.vlines(row.geomean, y - 0.31, y + 0.31, color=GEOMEAN_INK, linewidth=2.0, zorder=6)
        ax.hlines(y + 0.36, row.geomean_low, row.geomean_high, color=GEOMEAN_INK, linewidth=1.0, zorder=6)
        # Past the BAR's end, haloed rather than nudged, so it labels the bar and not the geomean.
        ax.text(row.median * 1.09,
                y,
                f"{row.median:.2f}x",
                va="center",
                ha="left",
                zorder=7,
                fontsize=style.ANNOTATION_PT,
                color=style.INK,
                family="monospace",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6, foreground="white")])

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{row.label}  (n={row.n})" for row in rows.itertuples(index=False)], color=style.INK)
    # Named from BASELINE, never written out: the label and the divisor drifting apart is exactly how a
    # figure comes to say "over sequential C" while dividing by something else.
    ax.set_xlabel(f"Speedup over {LABEL.get(BASELINE, BASELINE)} (Log Scale, Higher Is Better)")
    ax.axvline(1.0, color=style.REFERENCE, linewidth=1.0, zorder=1)
    style.value_axis(ax, "x", minor=False, major=False)
    style.despine(ax)
    ax.tick_params(axis="y", length=0)
    handles = [
        matplotlib.patches.Patch(facecolor=MEDIAN_INK, label="Median Speedup, 95% Bootstrap Interval"),
        matplotlib.lines.Line2D([], [], color=GEOMEAN_INK, linewidth=2.0, label="Geometric Mean, 95% Log-t Interval"),
    ]
    style.legend_below(fig, handles, ncol=1)
    ax.set_title("Canonicalization against the Compilers, llr-focus40", loc="left", color=style.INK)
    written = style.save(fig, out_dir / "canon_speedup")
    print(f"{written}")
    for row in rows.itertuples(index=False):
        print(f"  {row.label:<24} median {row.median:>7.2f}x   geomean {row.geomean:>7.2f}x   n={row.n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=here / "data" / "canon_llr40.csv")
    ap.add_argument("--out", type=pathlib.Path, default=here / "figures")
    ap.add_argument("--table", type=pathlib.Path, default=here / "data" / "canon_speedup.csv")
    args = ap.parse_args(argv)
    return draw(args.data, args.out, args.table)


if __name__ == "__main__":
    sys.exit(main())
