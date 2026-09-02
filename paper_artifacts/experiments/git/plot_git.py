"""Kernel framing against repository framing, per model: what the wider task costs and buys.

The A/B is the FRAMING, not the model: both legs of a pair are the same model on the same ten
scientific kernels, one shown the kernel alone and one shown the repository it lives in. So the
figure is the same dumbbell as the llr8 and llr9 ones, with the hollow marker for the kernel
framing and the filled one for the repository framing.

THREE PANELS, because the framing moves three different things and one of them is not speed. A
repository-framed agent submits more often and spends more tokens per kernel; whether it submits
anything FASTER is the question, and putting the acceptance count beside the speed-up is what
stops a faster geomean over two accepted cells being read as a better agent.

GEOMEAN, never median: a speed-up is a ratio, and the mean that matches the product over the set is
the geometric one.

Usage:  python3 plot_git.py [--data data/git_experiment_summary.csv] [--out figures]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchlib import dumbbell  # noqa: E402  -- the artifact is run from a clone, not installed
from benchlib import style  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

#: ``arm_dir`` prefix -> the model key the palette and the label table are held under, so this
#: experiment's hues match the ones the llr8 and llr9 figures give the same models.
MODEL_OF = {"oss120b": "oss120b", "qwen38": "qwen38"}

#: ``(column, panel title, axis label, scale)``. Accepted cells before speed-up, because a geomean
#: over two cells and one over seven are not the same claim.
PANELS = (
    ("cells_accepted", "Accepted submissions", "cells graded correct (of 30)", "linear"),
    ("geomean_speedup_accepted", "Speed-up of what was accepted", "geomean (x, log scale)", "log"),
    ("total_tokens", "Token cost", "million tokens over the arm", "linear"),
)

#: What each panel's raw column has to be multiplied by to reach the unit its axis is labelled in.
SCALE = {"total_tokens": 1e-6}


def read(path: pathlib.Path) -> dict[str, dict[str, dict[str, str]]]:
    """``model -> framing -> row``, from the per-arm summary the collector writes."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            model = MODEL_OF[row["arm_dir"].split("_", 1)[0]]
            out.setdefault(model, {})[row["framing"]] = row
    return out


def draw(arms: dict[str, dict[str, dict[str, str]]], out: pathlib.Path) -> pathlib.Path:
    models = [m for m in style.MODEL_COLOR if m in arms]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.4, 1.05 + 0.42 * len(models)))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.66, bottom=0.30, wspace=0.42)
    style.heading(fig, 0.012, 0.95, "Kernel framing against repository framing")
    for ax, (column, title, xlabel, scale) in zip(axes, PANELS, strict=True):
        style.axis_title(ax, title, pad=12.0)
        ax.set_xlabel(xlabel)
        style.row_axis(ax, [style.MODEL_LABEL[m] for m in models])
        drawn: list[float] = []
        for row, model in enumerate(models):
            legs = [
                float(arms[model][framing][column]) * SCALE.get(column, 1.0) if framing in arms[model] else None
                for framing in ("kernel", "repo")
            ]
            drawn += [v for v in legs if v is not None]
            style.dumbbell(ax, row, legs[0], legs[1], style.MODEL_COLOR[model])
        ax.margins(x=0.22)
        if scale == "log":
            # A log axis left to itself puts one decade tick on a 2x-to-8x range and labels the
            # minors `2 x 10^0`, which collide at this width. The speed-up ladder the llr figures
            # use frames the range that is actually occupied and labels it as plain multiples.
            ax.set_xscale("log")
            low, high, ticks = dumbbell.speedup_axis(drawn)
            ax.set_xlim(low, high)
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        if column == "cells_accepted":
            ax.get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style.key(axes[-1], [("kernel", style.marker(style.INK2, on=False)), ("repo", style.marker(style.INK2, on=True))],
              anchor=(1.0, 1.02))
    return style.save(fig, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "data" / "git_experiment_summary.csv")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "figures")
    args = parser.parse_args()
    style.apply()
    arms = read(args.data)
    draw(arms, args.out / "git_framing")
    for model, legs in arms.items():
        for framing, row in sorted(legs.items()):
            print(f"  {style.MODEL_LABEL[model]:16s} {framing:7s} accepted={row['cells_accepted']:>2s}/30 "
                  f"geomean {float(row['geomean_speedup_accepted']):6.3f}x  "
                  f"{int(row['total_tokens']) / 1e6:5.1f}M tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
