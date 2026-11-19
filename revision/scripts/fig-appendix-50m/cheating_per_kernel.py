"""Appendix C per-kernel speedup figure from the one-limit reruns: per model (colour) and device (shape), the
control setups rev-llr40-<model>-{c,hip,triton-device}. Rows from hpcagent_bench.stats.figures.kernel_speedups
(kernel_rows, kernel_metrics, summary_table), drawn by per_kernel.figure_panels at the ICLR text width, as v1.

    python cheating_per_kernel.py --tag-file llr40.txt --height-pt 145.984 --out cheating_per_kernel.pdf obs-*.db
"""

import argparse
import dataclasses
import pathlib

import pandas as pd
from matplotlib.lines import Line2D

from hpcagent_bench import study_tags, tags
from hpcagent_bench.results import studies
from hpcagent_bench.stats import cost, population, palette, style
from hpcagent_bench.stats.figures import kernel_speedups, per_kernel, setup_names

EXPERIMENT = "rev-llr40"
MODELS = ("qwen38", "kimi27sglang", "oss120b")
#: device -> (language suffix, v1 shape, filled, legend name)
DEVICES = {
    "CPU": ("c", "o", True, "CPU (C)"),
    "Triton": ("triton-device", "^", False, "GPU (Triton)"),
    "HIP": ("hip", "s", False, "GPU (HIP)"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("observations", type=pathlib.Path, nargs="+")
    ap.add_argument("--tag-file", type=pathlib.Path, required=True)
    ap.add_argument("--height-pt", type=float, required=True)
    ap.add_argument("--panel-height-in", type=float, default=1.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    kernels = tags.split_names(args.tag_file.read_text())
    frame = pd.concat([studies.read_observations(p) for p in args.observations], ignore_index=True)
    frame = cost.priced(population.select_setups(frame, EXPERIMENT), cost.resolve(cost.DEFAULT_COST_MODEL))
    rows: list[kernel_speedups.Row] = []
    for step, (device, (language, marker, filled, _)) in enumerate(DEVICES.items()):
        found = kernel_speedups.kernel_rows(
            pd.DataFrame(), frame, kernels, [], setup_names.setup_pattern(EXPERIMENT, language), [""],
            include_incomplete=True,
        )
        for row in found:
            model = row.framework.removeprefix(f"{EXPERIMENT}-").removesuffix(f"-{language}")
            rows.append(dataclasses.replace(
                row, label=f"{study_tags.model_name(model)} {device}", color=palette.model_shade(model, step),
                marker=marker, filled=filled,
            ))
    order = {f"{EXPERIMENT}-{m}-{DEVICES[d][0]}": (i, j) for i, m in enumerate(MODELS) for j, d in enumerate(DEVICES)}
    rows.sort(key=lambda row: order[row.framework])
    missing = sorted(set(order) - {row.framework for row in rows})
    if missing:
        raise SystemExit(f"no rows for {missing}")
    stem = args.out.with_suffix("")
    kernel_speedups.write_tables(rows, stem)
    style.apply()
    baseline = population.rows_denominator(frame.loc[frame.row_kind == "submission"])
    speed = kernel_speedups.kernel_metrics(rows, kernels, baseline)[0]  # the speedup panel only, as v1
    speed = dataclasses.replace(speed, ylabel="Speedup\n(higher is better)")
    legend = [
        Line2D([], [], marker="o", linestyle="none", color=palette.model_color(m), label=study_tags.model_name(m))
        for m in MODELS
    ] + [
        Line2D([], [], marker=marker, linestyle="none", color=style.INK, markerfacecolor=style.INK if filled else "none",
               label=name)
        for _, marker, filled, name in DEVICES.values()
    ]
    fig = per_kernel.figure_panels(
        [speed], sorted(kernels), per_kernel.Style.CI, True, "", width_in=style.ICLR_TEXT_WIDTH_IN, legend=legend,
        panel_height_in=args.panel_height_in, summary_values=False,
    )
    per_kernel.save(fig, args.out, print_size=True)
    width, height = (72 * v for v in fig.get_size_inches())
    print(f"baseline {baseline}; page {width:.3f} x {height:.3f} pt (v1 height {args.height_pt})")
    print(kernel_speedups.summary_table(rows).to_string(index=False))


if __name__ == "__main__":
    main()
