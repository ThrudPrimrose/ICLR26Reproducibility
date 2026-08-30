"""TSVC kernel speed-ups from the MPR framework sweep, on a SIGNED relative axis.

Three arms against the numpy reference the sweep times per kernel: dace canon, dace main and llvm
autopar. The sweep writes one CSV per framework -- ``<framework>.rank<N>.csv`` when a job shards a
node four ways, ``<framework>.csv`` when it does not -- and both shapes are read and concatenated.

THE AXIS. A ratio axis is unreadable for a two-sided result: 2x faster is 2.0 and 2x slower is 0.5,
so the win occupies the whole axis and the equal-sized loss is squeezed into the strip below 1. This
plots the SIGNED relative speed-up instead, ``s - 1`` for a speed-up and ``1 - 1/s`` for a slowdown,
which is 0 at no change and puts a 2x win at +1 and a 2x loss at -1 -- equidistant, and odd about
zero, so ``f(1/s) == -f(s)`` exactly.

THE ESTIMATOR. Speed-ups are ratios, so the centre is the GEOMETRIC mean and it and its interval are
computed in LOG space: the mean of ``log s``, a Student-t 95% interval on that mean, both mapped back
through ``exp`` and then through the signed transform. The interval is asymmetric on this axis, which
is correct and is why it is drawn from explicit ends rather than a half-width. The median is drawn as
a secondary tick; the geomean is the headline.

A row that crashed, that the harness did not validate, or that carries no timing is NOT a data point.
Those rows are counted and reported on the figure, never plotted as 1.0 -- a miscompile scored as
"no change" is the one failure mode that would flatter every arm equally.

Usage:  python3 plot_tsvc_speedup.py <sweep-directory>
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import math
import pathlib
import random
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection
import scipy.stats  # noqa: E402  -- ditto, it pulls matplotlib in itself

#: Framework -> the name a reader knows it by. Insertion order is the order on the axis.
ARMS = {"dace_cpu_canonicalize": "dace canon", "dace_cpu": "dace main", "cc_llvm_autopar": "llvm autopar"}

#: The speed-up DENOMINATOR: the interpreted reference every arm is trying to beat.
BASELINE = "numpy"

#: Kernels this figure is about. tsvc_2_5* sources are already under this prefix.
TSVC_PREFIX = "tsvc_2"

#: Arm -> hue, the same blue/orange/green as plot.py and plot_llr8w2.py: separable under all three
#: common colour vision deficiencies, which a red/green pair is not.
ARM_COLOR = {"dace_cpu_canonicalize": "#2a78d6", "dace_cpu": "#eb6834", "cc_llvm_autopar": "#1b8a5a"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

CONFIDENCE = 0.95


@dataclasses.dataclass(frozen=True)
class Arm:
    """One framework's usable TSVC timings, and a tally of what was thrown away and why."""

    framework: str
    times: dict[str, float]
    rejected: collections.Counter


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


def signed(speedup: float) -> float:
    """Ratio -> signed relative speed-up: 2.0 to +1.0, 1.0 to 0.0, 0.5 to -1.0, 0.25 to -3.0.

    Odd about 1.0 by construction -- for s >= 1 the win is ``s - 1``, and a slowdown of the same
    factor is that win negated, ``-(1/s - 1)`` -- so a win and its exact inverse land at mirrored
    distances from zero and neither side of the axis flatters an arm.
    """
    return speedup - 1.0 if speedup >= 1.0 else 1.0 - 1.0 / speedup


def shard_paths(root: pathlib.Path, framework: str) -> list[pathlib.Path]:
    """The framework's CSVs: the unsharded file, the per-rank shards, or both, in a stable order."""
    single = root / f"{framework}.csv"
    return ([single] if single.is_file() else []) + sorted(root.glob(f"{framework}.rank*.csv"))


def read_arm(root: pathlib.Path, framework: str) -> Arm:
    """Concatenate the framework's shards into ``kernel -> ms``, tallying every rejected TSVC row.

    Only TSVC rows are counted as rejects. A non-TSVC row is out of SCOPE, not excluded, and mixing
    the two would report the corpus size as an exclusion count.
    """
    times: dict[str, float] = {}
    rejected: collections.Counter = collections.Counter()
    for path in shard_paths(root, framework):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                kernel = row["kernel"]
                if not kernel.startswith(TSVC_PREFIX):
                    continue
                if row["status"] != "ok":
                    rejected[f"status={row['status']}"] += 1
                elif row["validated"] != "True":
                    rejected["not validated"] += 1
                elif not row["median_ms"] or float(row["median_ms"]) <= 0:
                    rejected["no timing"] += 1
                else:
                    # A kernel can appear twice across shards only if a rank was re-run; the fastest
                    # of the two is the one the sweep itself would report, so keep the minimum.
                    ms = float(row["median_ms"])
                    times[kernel] = min(times.get(kernel, ms), ms)
    return Arm(framework, times, rejected)


def speedups(arm: Arm, baseline: dict[str, float]) -> dict[str, float]:
    """``kernel -> baseline_ms / arm_ms`` over the kernels BOTH the arm and the reference timed."""
    return {k: baseline[k] / ms for k, ms in sorted(arm.times.items()) if k in baseline}


def geomean_interval(values: list[float]) -> tuple[float, float, float]:
    """Geometric mean of a ratio set and its 95% t-interval, computed in LOG space.

    The interval ends come back as ratios, not as a half-width: the transform out of log space is
    not linear, so a symmetric +/- would be wrong on both this axis and the ratio one. A single
    observation has no spread to estimate, so its interval is the point itself.
    """
    logs = [math.log(v) for v in values]
    centre = math.exp(statistics.fmean(logs))
    if len(logs) < 2:
        return centre, centre, centre
    half = scipy.stats.t.ppf(0.5 + CONFIDENCE / 2.0, len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs))
    return centre, math.exp(statistics.fmean(logs) - half), math.exp(statistics.fmean(logs) + half)


def render(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Draw the figure for one sweep directory; returns the stem both files were written to."""
    reference = read_arm(root, BASELINE)
    if not reference.times:
        raise SystemExit(f"no {BASELINE} baseline in {root}: looked for {BASELINE}.csv and {BASELINE}.rank*.csv. "
                         f"Every speed-up here is a ratio against it, so there is nothing to plot without it.")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # Seeded, so the same CSVs draw the same cloud: an unseeded jitter makes two renders of one
    # dataset look like two measurements.
    jitter = random.Random(0)
    notes: list[str] = []
    for index, (framework, name) in enumerate(ARMS.items()):
        arm = read_arm(root, framework)
        values = speedups(arm, reference.times)
        colour = ARM_COLOR[framework]
        missing = len(arm.times) - len(values)
        tally = ", ".join(f"{n} {why}" for why, n in sorted(arm.rejected.items()))
        if not values:
            ax.text(index, 0.0, "no data", ha="center", va="bottom", fontsize=8, color=MUTED, style="italic")
            notes.append(f"{name}: no usable rows" + (f" ({tally})" if tally else ""))
            continue
        ax.scatter([index - 0.16 + jitter.uniform(-0.11, 0.11) for _ in values], [signed(v) for v in values.values()],
                   s=13,
                   color=colour,
                   alpha=0.45,
                   linewidth=0,
                   zorder=2)
        centre, low, high = geomean_interval(list(values.values()))
        ax.errorbar(index + 0.22,
                    signed(centre),
                    yerr=[[signed(centre) - signed(low)], [signed(high) - signed(centre)]],
                    fmt="o",
                    markersize=6,
                    color=colour,
                    ecolor=INK,
                    elinewidth=1.2,
                    capsize=5,
                    zorder=4)
        middle = statistics.median(list(values.values()))
        ax.plot([index + 0.14, index + 0.30], [signed(middle)] * 2, color=INK, linewidth=1.0, zorder=3)
        ax.annotate(f"n={len(values)}", (index + 0.22, signed(high)),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7.5,
                    color=INK)
        excluded = [tally] if tally else []
        if missing:
            excluded.append(f"{missing} not timed by {BASELINE}")
        notes.append(f"{name}: geomean {centre:.2f}x, median {middle:.2f}x, "
                     f"excluded {'; '.join(excluded) if excluded else 'none'}")

    ax.axhline(0.0, color=INK, linewidth=1.0, zorder=1)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(list(ARMS.values()))
    ax.set_ylabel(
        "signed relative speed-up vs numpy\n$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower")
    ax.set_title(f"TSVC kernels, signed speed-up against the {BASELINE} reference", loc="left", pad=30)
    # The category axis spans the arms, not the arms that produced points: an arm whose CSV never
    # appeared was still part of the comparison, and letting the data set the limits dropped its
    # tick label off the end of the axis entirely.
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.margins(y=0.10)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=MUTED, markersize=4, alpha=0.5),
        plt.Line2D([], [], marker="o", linestyle="none", color=MUTED, markersize=6),
        plt.Line2D([], [], color=INK, linewidth=1.0),
    ]
    ax.legend(handles, ["one kernel", f"geomean, {CONFIDENCE:.0%} t-interval in log space", "median"],
              frameon=False,
              loc="lower left",
              bbox_to_anchor=(0.0, 1.0),
              ncol=3,
              fontsize=7.5)
    # One line per arm: the three notes on one line ran off the page at every figure width that
    # still fits a paper column.
    fig.text(0.008, 0.008, "\n".join(notes + [f"source: {root.name}"]), fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)
    for note in notes:
        print(f"  {note}")
    print(f"  {out}.pdf / .png")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=pathlib.Path, help="directory of <framework>[.rank<N>].csv files")
    args = parser.parse_args()
    style()
    # The source directory is in the file name: two sweeps of the same three arms are two
    # measurements, and one silently overwriting the other is how a stale figure reaches a paper.
    figures = pathlib.Path(__file__).resolve().parent / "figures"
    render(args.sweep, figures / f"tsvc_signed_speedup_{args.sweep.resolve().name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
