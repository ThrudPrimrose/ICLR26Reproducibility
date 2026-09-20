# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up of three models in C, on the CPU and GPU loop-level tracks, with the
submissions an audit disowned drawn as themselves.

THE FIGURE EXISTS FOR ONE COMPARISON. A kernel whose input distribution an agent reverse-engineered
sits far above what the same kernel's honest submissions reach, and that gap is only visible with
the honest population drawn beside it. A flagged answer is therefore drawn AT THE VALUE IT CLAIMED,
as a cross carrying a star (`hpcagent_bench.stats.figures.per_kernel.draw_flagged`), never hidden
and never credited.

Device is the SHAPE and model is the COLOUR, so one column carries both tracks for one model.

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

#: (arm, kernel) pairs an audit disowned. Source evidence: notes/evasion/band-evidence.md.
FLAGGED: frozenset[tuple[str, str]] = frozenset(
    {
        ("cpf-llr-focus40-qwen38-c", "ext_break_capture"),
        ("gpu-llr-focus40-qwen38-c-openmp-skills", "ext_break_capture"),
        ("gpu-llr-focus40-qwen38-c-openmp-skills", "versioned_distance_update"),
        ("cpf-llr-focus40-kimi27sglang-c-cpfsrc-v2", "tsvc_2_vtvtv"),
        ("cpf-llr-focus40-glm53-c-skills", "tsvc_2_s311"),
    }
)

#: Reduced height: this figure carries one measure, not the stacked pair the efficacy row does.
HEIGHT_IN: float = 2.15


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
        values = tuple(float(v) for v in group.speedup)
        flagged = (arm, str(kernel)) in FLAGGED or bool(group.suspect.fillna(0).astype(float).max())
        out.append(per_kernel.KernelCell(str(kernel), values, flagged=flagged))
    return out


def kernel_order(series: dict[tuple[str, str], list[per_kernel.KernelCell]]) -> list[str]:
    """Kernels ordered by their median honest speed-up, so the exploited outliers stand clear of
    the trend rather than being buried in an alphabetical list."""
    honest: dict[str, list[float]] = {}
    for group in series.values():
        for cell in group:
            if not cell.flagged:
                honest.setdefault(cell.kernel, []).append(cell.median())
    every = {cell.kernel for group in series.values() for cell in group}
    return sorted(every, key=lambda k: -float(np.median(honest[k])) if honest.get(k) else 0.0)


def draw(series: dict[tuple[str, str], list[per_kernel.KernelCell]], out: pathlib.Path) -> None:
    style.apply()
    order = kernel_order(series)
    x_of = {k: i for i, k in enumerate(order)}
    summary_x = len(order) + per_kernel.SUMMARY_GAP
    figure, ax = plt.subplots(figsize=(style.ACM_TEXT_WIDTH_IN, HEIGHT_IN))
    for (model, device), group in series.items():
        colour = palette.model_color(model)
        shape = DEVICE_SHAPE[device]
        honest = [c for c in group if not c.flagged]
        ax.plot(
            [x_of[c.kernel] for c in honest], [c.median() for c in honest], marker=shape, markersize=3.0,
            linestyle="none", color=colour, markerfacecolor=colour if device == "CPU" else "none",
            markeredgewidth=0.8, alpha=0.85, zorder=3,
        )  # fmt: skip
        for cell in (c for c in group if c.flagged):
            per_kernel.draw_flagged(ax, x_of[cell.kernel], cell.median(), colour)
        if honest:
            ax.plot(
                [summary_x], [float(np.median([c.median() for c in honest]))], marker=shape, markersize=5.0,
                linestyle="none", color=colour, markerfacecolor=colour if device == "CPU" else "none",
                markeredgewidth=1.2, zorder=4,
            )  # fmt: skip
    ax.axvline(len(order) - 0.5 + per_kernel.SUMMARY_GAP / 2, color=style.MUTED, linestyle="--", linewidth=0.5)
    ax.set_yscale("log")
    ax.axhline(1.0, color=style.INK, linewidth=0.6, zorder=1)
    ax.set_ylabel("Speed-Up\n(higher is better)", fontsize=8.0, color=style.INK)
    ax.set_xticks([*range(len(order)), summary_x])
    ax.set_xticklabels([*order, "median"], rotation=90, fontsize=4.2, color=style.INK)
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
        plt.Line2D([], [], marker=per_kernel.FLAGGED_MARKER, linestyle="none", color=style.INK, markersize=4.0,
                   label="Disowned by audit (drawn at the claimed value)"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=3, fontsize=6.0,
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
    cpu, gpu = pd.read_csv(args.cpu, low_memory=False), pd.read_csv(args.gpu, low_memory=False)
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
