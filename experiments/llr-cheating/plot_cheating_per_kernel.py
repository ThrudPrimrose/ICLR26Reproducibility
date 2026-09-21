# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up of three models in C, on the CPU and GPU loop-level tracks.

Device is the SHAPE and model is the COLOUR, so one column carries both tracks for one model. The
median column carries one sub-column per device, each mark labelled with its value.

Usage::

    python3 experiments/llr-cheating/plot_cheating_per_kernel.py \\
        --cpu experiments/llr-cpu/data/llr-cpu.csv \\
        --gpu experiments/llr-gpu/data/llr-gpu.csv \\
        --out experiments/llr-cheating/figures/cheating-per-kernel.pdf
"""

import argparse
import pathlib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette, style
from hpcagent_bench.stats.figures import per_kernel

#: The C control arm on each track: no packet, so the figure reads model and device only.
CPU_ARM: str = "cpf-llr-focus40-{model}-c"
GPU_ARM: str = "gpu-llr-focus40-{model}-c-openmp"

MODELS: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang")

#: Device is the shape. A filled circle and an open triangle stay apart in print and for a
#: colour-blind reader, which two filled shapes of the same size do not.
DEVICE_SHAPE: dict[str, str] = {"CPU": "o", "GPU": "^"}

#: Reduced height: this figure carries one measure, not the stacked pair the efficacy row does.
HEIGHT_IN: float = 1.8

#: A kernel name longer than this folds onto a second line.
NAME_WRAP: int = 14

#: X offset of the GPU median sub-column from the CPU one, and of a value label from its mark.
DEVICE_STEP: float = 3.0
LABEL_PAD: float = 0.7

#: Minimum vertical gap between two value labels of one sub-column, in decades.
LABEL_GAP_DECADES: float = 0.13


def answers(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The graded submission rows of one arm, latest attempt per kernel."""
    rows = frame[(frame.arm == arm) & (frame.record == "submission")]
    rows = rows[rows.speedup.notna() & (rows.speedup > 0)]
    if rows.empty:
        return rows
    return rows.sort_values("ts_ms").drop_duplicates("benchmark", keep="last")


def cells(frame: pd.DataFrame, arm: str) -> list[per_kernel.KernelCell]:
    rows = answers(frame, arm)
    out: list[per_kernel.KernelCell] = []
    for kernel, group in rows.groupby("benchmark"):
        out.append(
            per_kernel.KernelCell(str(kernel), tuple(float(v) for v in group.speedup))
        )
    return out


def kernel_order(
    series: dict[tuple[str, str], list[per_kernel.KernelCell]],
) -> list[str]:
    """Kernels ordered by their median speed-up across the series, fastest first."""
    medians: dict[str, list[float]] = {}
    for group in series.values():
        for cell in group:
            medians.setdefault(cell.kernel, []).append(cell.median())
    return sorted(medians, key=lambda k: -float(np.median(medians[k])))


def kernel_label(kernel: str) -> str:
    """The manifest's short display name, folded onto at most two lines."""
    name = experiment_tags.kernel_short_display_name(kernel)
    return "\n".join(textwrap.wrap(name, NAME_WRAP, max_lines=2, placeholder=".."))


def spread_labels(values: list[float]) -> list[float]:
    """Label heights for marks at ``values``: each at its mark, pushed up just enough to keep
    LABEL_GAP_DECADES from the label below it."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    floor = -np.inf
    for i in order:
        height = max(np.log10(values[i]), floor + LABEL_GAP_DECADES)
        placed[i] = float(10.0**height)
        floor = height
    return placed


def draw(
    series: dict[tuple[str, str], list[per_kernel.KernelCell]], out: pathlib.Path
) -> None:
    style.apply()
    order = kernel_order(series)
    x_of = {k: i for i, k in enumerate(order)}
    summary_x = len(order) + per_kernel.SUMMARY_GAP
    device_x = {"CPU": summary_x, "GPU": summary_x + DEVICE_STEP}
    figure, ax = plt.subplots(figsize=(style.ACM_TEXT_WIDTH_IN, HEIGHT_IN))
    summaries: dict[str, list[tuple[float, str]]] = {"CPU": [], "GPU": []}
    for (model, device), group in series.items():
        colour = palette.model_color(model)
        shape = DEVICE_SHAPE[device]
        ax.plot(
            [x_of[c.kernel] for c in group], [c.median() for c in group], marker=shape, markersize=3.0,
            linestyle="none", color=colour, markerfacecolor=colour if device == "CPU" else "none",
            markeredgewidth=0.8, alpha=0.85, zorder=3,
        )  # fmt: skip
        median = float(np.median([c.median() for c in group]))
        summaries[device].append((median, colour))
        ax.plot(
            [device_x[device]], [median], marker=shape, markersize=5.0, linestyle="none", color=colour,
            markerfacecolor=colour if device == "CPU" else "none", markeredgewidth=1.2, zorder=4,
        )  # fmt: skip
    for device, marks in summaries.items():
        for (median, colour), height in zip(
            marks, spread_labels([m for m, _ in marks]), strict=True
        ):
            ax.text(device_x[device] + LABEL_PAD, height, f"{median:.1f}x", fontsize=5.0, color=colour,
                    ha="left", va="center")  # fmt: skip
    ax.axvline(
        len(order) - 0.5 + per_kernel.SUMMARY_GAP / 2,
        color=style.MUTED,
        linestyle="--",
        linewidth=0.5,
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color=style.INK, linewidth=0.6, zorder=1)
    ax.set_ylabel("Speed-Up\n(higher is better)", fontsize=8.0, color=style.INK)
    ax.set_xticks([*range(len(order)), summary_x + DEVICE_STEP / 2])
    ax.set_xticklabels([*(kernel_label(k) for k in order), "Median"], rotation=90, fontsize=4.2,
                       color=style.INK, linespacing=0.95)  # fmt: skip
    ax.set_xlim(-0.8, summary_x + DEVICE_STEP + 2.0)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=palette.model_color(m), markersize=4.0,
                   label=experiment_tags.model_name(m))
        for m in MODELS
    ]  # fmt: skip
    handles += [
        plt.Line2D([], [], marker=DEVICE_SHAPE["CPU"], linestyle="none", color=style.INK, markersize=4.0, label="CPU"),
        plt.Line2D([], [], marker=DEVICE_SHAPE["GPU"], linestyle="none", color=style.INK, markersize=4.0,
                   markerfacecolor="none", label="GPU (OpenMP Offload)"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=5, fontsize=6.0,
              frameon=False, handletextpad=0.4, columnspacing=1.1)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight", dpi=300)
    figure.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(figure)
    print(f"figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=pathlib.Path, required=True)
    ap.add_argument("--gpu", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cpu, gpu = (
        pd.read_csv(args.cpu, low_memory=False),
        pd.read_csv(args.gpu, low_memory=False),
    )
    series: dict[tuple[str, str], list[per_kernel.KernelCell]] = {}
    for model in MODELS:
        for device, frame, template in (("CPU", cpu, CPU_ARM), ("GPU", gpu, GPU_ARM)):
            group = cells(frame, template.format(model=model))
            if group:
                series[(model, device)] = group
    if not series:
        raise SystemExit("no arms matched")
    draw(series, args.out)


if __name__ == "__main__":
    main()
