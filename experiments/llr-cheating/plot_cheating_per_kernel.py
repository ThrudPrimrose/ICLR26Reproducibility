# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up of three models in C, on the CPU and GPU loop-level tracks.

Model is the COLOUR and device the SHAPE, so one kernel column carries both tracks of every model;
the summary column gives each series' geomean with its 95% interval and value. Drawn by
:func:`hpcagent_bench.stats.figures.per_kernel.figure_one` at the ICLR text width.

Usage::

    python3 experiments/llr-cheating/plot_cheating_per_kernel.py \\
        --cpu experiments/llr-cpu/data/llr-cpu.csv \\
        --gpu experiments/llr-gpu/data/llr-gpu.csv \\
        --out experiments/llr-cheating/figures/cheating-per-kernel.pdf
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette, style
from hpcagent_bench.stats.figures import per_kernel

#: The C control arm on each track: no packet, so the figure reads model and device only.
ARMS: dict[str, str] = {"CPU": "cpf-llr-focus40-{model}-c", "GPU": "gpu-llr-focus40-{model}-c-openmp"}

MODELS: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang")

#: Device is the shape. A filled circle and an open triangle stay apart in print and for a
#: colour-blind reader, which two filled shapes of the same size do not.
DEVICE_MARK: dict[str, tuple[str, bool]] = {"CPU": ("o", True), "GPU": ("^", False)}
DEVICE_NAME: dict[str, str] = {"CPU": "CPU", "GPU": "GPU (OpenMP Offload)"}


def series_of(frames: dict[str, pd.DataFrame]) -> list[per_kernel.Series]:
    """One series per (model, device) with an answer, each kernel at its latest run's answer."""
    out: list[per_kernel.Series] = []
    for model in MODELS:
        for device, frame in frames.items():
            cells = per_kernel.answer_cells(frame[frame.arm == ARMS[device].format(model=model)])
            if cells:
                marker, filled = DEVICE_MARK[device]
                label = f"{experiment_tags.model_name(model)} {device}"
                out.append(per_kernel.Series(label, tuple(cells), palette.model_color(model), marker, filled))
    return out


def legend() -> list[matplotlib.lines.Line2D]:
    """The two channels once each: a swatch per model, a shape per device."""
    models = [
        matplotlib.lines.Line2D([], [], marker="o", linestyle="none", color=palette.model_color(model),
                                label=experiment_tags.model_name(model))
        for model in MODELS
    ]  # fmt: skip
    devices = [
        matplotlib.lines.Line2D([], [], marker=marker, linestyle="none", color=style.INK,
                                markerfacecolor=style.INK if filled else "none", label=DEVICE_NAME[device])
        for device, (marker, filled) in DEVICE_MARK.items()
    ]  # fmt: skip
    return models + devices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=pathlib.Path, required=True)
    ap.add_argument("--gpu", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    frames = {"CPU": pd.read_csv(args.cpu, low_memory=False), "GPU": pd.read_csv(args.gpu, low_memory=False)}
    series = series_of(frames)
    if not series:
        raise SystemExit("no arms matched")
    style.apply()
    metric = per_kernel.speedup_series_metric(series, "Speed-Up\n(higher is better)")
    kernels = per_kernel.ordered_kernels(metric.cells)
    fig = per_kernel.figure_one(metric, kernels, "ci", True, "", width_in=style.ICLR_TEXT_WIDTH_IN, legend=legend())
    print(f"figure -> {per_kernel.save(fig, args.out)}")


if __name__ == "__main__":
    main()
