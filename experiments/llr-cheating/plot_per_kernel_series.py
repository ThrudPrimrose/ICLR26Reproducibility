# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stacked per-kernel figure with SEVERAL populations on one kernel axis.

The single-series form answers "is this kernel hard". It cannot answer "is this kernel hard, or is
this model bad at it", which needs the models drawn over the same kernels. Each series gets a small
symmetric x offset inside its kernel column (`per_kernel.dodge_offsets`) so the per-kernel
confidence intervals stay readable where they would otherwise overprint.

Colour names the model, shape names the language, and a hollow shape names the GPU track.

Usage::

    python3 experiments/llr-cheating/plot_per_kernel_series.py \\
        --observations experiments/llr-cpu/data/llr-cpu.csv \\
        --observations experiments/llr-gpu/data/llr-gpu.csv \\
        --series 'Qwen3.8-27B C=cpf-llr-focus40-qwen38-c' \\
        --series 'Qwen3.8-27B HIP=gpu-llr-focus40-qwen38-hip' \\
        --out experiments/llr-cheating/figures/per-kernel-series.pdf
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from hpcagent_bench.stats import palette, style
from hpcagent_bench.stats.figures import per_kernel

#: Language is the shape. These stay apart at 3.6pt in print and under every CVD simulation in the
#: palette guide; two filled circles of different size do not.
LANGUAGE_SHAPE: dict[str, str] = {"C": "o", "Fortran": "s", "HIP": "^", "Triton": "D", "OMP": "v"}

#: A GPU series draws hollow, so the device is readable without spending a second colour channel.
GPU_TOKENS: frozenset[str] = frozenset({"HIP", "Triton", "OMP"})

DEFAULT_SERIES: tuple[str, ...] = (
    "Qwen3.8-27B C=cpf-llr-focus40-qwen38-c",
    "GPT-OSS-120B C=cpf-llr-focus40-oss120b-c",
    "Kimi-K2.7-Code C=cpf-llr-focus40-kimi27sglang-c",
    "Qwen3.8-27B HIP=gpu-llr-focus40-qwen38-hip",
    "GPT-OSS-120B HIP=gpu-llr-focus40-oss120b-hip",
    "Kimi-K2.7-Code HIP=gpu-llr-focus40-kimi27sglang-hip",
)

#: (arm, kernel) pairs an audit disowned; see notes/evasion/band-evidence.md.
FLAGGED: frozenset[tuple[str, str]] = frozenset(
    {
        ("gpu-llr-focus40-qwen38-c-openmp-skills", "versioned_distance_update"),
        ("cpf-llr-focus40-kimi27sglang-c-cpfsrc-v2", "tsvc_2_vtvtv"),
        ("cpf-llr-focus40-glm53-c-skills", "tsvc_2_s311"),
    }
)


def model_of(label: str) -> str:
    return label.rsplit(" ", 1)[0]


def language_of(label: str) -> str:
    return label.rsplit(" ", 1)[-1]


def model_key(name: str) -> str:
    for key, shown in (("qwen38", "Qwen3.8-27B"), ("oss120b", "GPT-OSS-120B"), ("kimi27sglang", "Kimi-K2.7-Code")):
        if shown == name:
            return key
    return name


def graded(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows = frame[(frame.arm == arm) & (frame.record == "submission")]
    rows = rows[rows.speedup.notna() & (rows.speedup > 0)]
    return rows.sort_values("ts_ms")


def speed_cells(frame: pd.DataFrame, arm: str) -> list[per_kernel.KernelCell]:
    """One cell per kernel holding EVERY episode's speed-up, which is what gives the cell an
    interval; a kernel with one episode still draws its point."""
    out: list[per_kernel.KernelCell] = []
    for kernel, group in graded(frame, arm).groupby("benchmark"):
        values = tuple(float(v) for v in group.speedup)
        flagged = (arm, str(kernel)) in FLAGGED or bool(group.suspect.fillna(0).astype(float).max())
        out.append(per_kernel.KernelCell(str(kernel), values, flagged=flagged))
    return out


def token_cells(frame: pd.DataFrame, arm: str, roster: set[str]) -> list[per_kernel.KernelCell]:
    """Tokens over the SAME kernels the speed-up panel draws.

    Without the roster the token panel picks up every row the arm has under any record, which on
    these CSVs is a superset of the graded roster and stretches the shared x axis to kernels the
    top panel has nothing to put in.
    """
    column = "tokens_billed" if "tokens_billed" in frame.columns else "tokens"
    rows = frame[(frame.arm == arm) & frame[column].notna() & (frame[column] > 0)]
    rows = rows[rows.benchmark.astype(str).isin(roster)]
    out: list[per_kernel.KernelCell] = []
    for kernel, group in rows.groupby("benchmark"):
        out.append(per_kernel.KernelCell(str(kernel), tuple(float(v) for v in group[column])))
    return out


def build(frames: list[pd.DataFrame], specs: list[str]) -> tuple[list[per_kernel.Series], list[per_kernel.Series]]:
    speed: list[per_kernel.Series] = []
    tokens: list[per_kernel.Series] = []
    for spec in specs:
        label, _, arm = spec.partition("=")
        language = language_of(label)
        colour = palette.model_color(model_key(model_of(label)))
        marker = LANGUAGE_SHAPE.get(language, "o")
        filled = language not in GPU_TOKENS
        for frame in frames:
            cells = speed_cells(frame, arm)
            if not cells:
                continue
            roster = {c.kernel for c in cells}
            speed.append(per_kernel.Series(label, tuple(cells), colour, marker, filled))
            tokens.append(per_kernel.Series(label, tuple(token_cells(frame, arm, roster)), colour, marker, filled))
            break
        else:
            print(f"  no rows for {arm}")
    return speed, tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", type=pathlib.Path, action="append", required=True)
    ap.add_argument("--series", action="append", default=None, help="'Label=arm', repeatable")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    frames = [pd.read_csv(p, low_memory=False) for p in args.observations]
    speed, tokens = build(frames, list(args.series or DEFAULT_SERIES))
    if not speed:
        raise SystemExit("no series matched")
    kernels = per_kernel.shared_kernel_order(
        [c for s in speed for c in s.cells], [c for s in tokens for c in s.cells]
    )  # fmt: skip
    style.apply()
    figure = per_kernel.figure_stacked(
        per_kernel.speedup_series_metric(speed, "Speed-Up"),
        per_kernel.token_series_metric(tokens, "Tokens per Episode"),
        kernels,
        "ci",
        True,
        args.title,
    )
    handles = [
        matplotlib.lines.Line2D(
            [], [], marker=s.marker, linestyle="none", color=s.color, markersize=4.0,
            markerfacecolor=s.color if s.filled else "none", markeredgewidth=1.0, label=s.label,
        )  # fmt: skip
        for s in speed
    ]
    # The kernel names are rotated and long, so the legend has to clear them by the height of the
    # longest one rather than by a fixed nudge; anchoring it to the figure's bottom put it through
    # the middle of the labels.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lowest = min(
        (t.get_window_extent(renderer).y0 for t in figure.axes[-1].get_xticklabels()),
        default=0.0,
    )
    figure.legend(
        handles=handles, loc="upper center", ncol=min(len(handles), 3), fontsize=6.0, frameon=False,
        bbox_to_anchor=(0.5, lowest / figure.bbox.height - 0.02), bbox_transform=figure.transFigure,
        handletextpad=0.4, columnspacing=1.4,
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight", dpi=300)
    figure.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"figure -> {args.out}  ({len(speed)} series, {len(kernels)} kernels)")


if __name__ == "__main__":
    main()
