"""Figure 5: the cost ratio rho_C of three comparisons under each token weighting w = (fresh input, cached input,
output), with 95% sign-flip intervals; above 1x the treated setup is cheaper. Same comparisons, axes and page size
as the submitted figure (1.9 x 1.644 in). Values come from main's per-kernel reduction
(``paired_setups.tokens_by_setup_kernel``) priced by main's ``cost.priced`` under each shipped card, and main's
paired test (``significance.paired``); colours, shapes and type from ``hpcagent_bench.stats.palette`` / ``style``.

    python figure5.py --wt WORKTREE --db DB_DIR --out OUT_DIR/figures/figure5_cost_weighting.pdf
"""

import argparse
import math
import pathlib
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from hpcagent_bench.stats import cost, palette, significance, style

import analysis

#: The submitted page, inches (396 / 72 wide figure rows aside: 136.8 x 118.37 pt).
PAGE_IN = (136.8 / 72, 118.37221875 / 72)
CARDS = (("effective", "1,0,1"), ("billed", "1,.1,1"), ("total", "1,1,1"))


class Comparison(NamedTuple):
    label: str
    model: str
    marker: str
    db: str
    treated: str
    control: str


COMPARISONS = (
    Comparison("OpenHands", "qwen38", palette.harness_marker("openhands"), "harness20",
               "harness20-qwen38-openhands", "harness20-qwen38-claude"),
    Comparison("CPF", "qwen38", palette.packet_marker("cpf-cpu-src"), "llr-focus40",
               "cpf-llr-focus40-qwen38-c-cpfsrc-v2", "cpf-llr-focus40-qwen38-c"),
    Comparison("HIP Skills", "oss120b", palette.packet_marker("lang-skills"), "llr-focus40",
               "gpu-llr-focus40-oss120b-hip-skills", "gpu-llr-focus40-oss120b-hip"),
)


def ratios(ps: object, db: pathlib.Path) -> pd.DataFrame:
    """rho_C and its 95% sign-flip interval, per comparison and card."""
    rows = []
    loaded: dict[str, pd.DataFrame] = {}
    for comparison in COMPARISONS:
        if comparison.db not in loaded:
            loaded[comparison.db] = ps.load_observations([db / f"{comparison.db}.db"], cost.resolve("effective"))
        frame = loaded[comparison.db]
        frame = frame[frame.setup.isin([comparison.treated, comparison.control])]
        for card, weights in CARDS:
            tokens = ps.tokens_by_setup_kernel(cost.priced(frame, cost.resolve(card)))
            kernels = sorted(k for (s, k) in tokens if s == comparison.treated and tokens.get((comparison.control, k), 0) > 0)
            logs = [math.log(tokens[(comparison.control, k)] / tokens[(comparison.treated, k)]) for k in kernels]
            test = significance.paired(logs)
            rows.append({
                "comparison": comparison.label, "treated": comparison.treated, "control": comparison.control,
                "card": card, "weights": weights, "n": len(logs), "rho_c": math.exp(test.estimate),
                "ci_low": math.exp(test.low), "ci_high": math.exp(test.high), "p": test.pvalue, "test": test.method,
            })
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame, out: pathlib.Path) -> None:
    style.apply()
    fig = plt.figure(figsize=PAGE_IN)
    left, width = 0.23, 0.74
    ax = fig.add_axes((left, 0.43, width, 0.55))
    offsets = (-0.18, 0.0, 0.18)
    for offset, comparison in zip(offsets, COMPARISONS, strict=True):
        rows = table[table.comparison == comparison.label]
        x = [index + offset for index in range(len(CARDS))]
        colour = palette.model_color(comparison.model)
        ax.errorbar(
            x, rows.rho_c, yerr=[rows.rho_c - rows.ci_low, rows.ci_high - rows.rho_c], fmt="none",
            ecolor=colour, elinewidth=0.9, capsize=0,
        )
        ax.scatter(x, rows.rho_c, marker=palette.marker_style(comparison.marker), s=22, color=colour, zorder=3,
                   linewidths=0)
    ax.axhline(1.0, color=style.REFERENCE, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: style.ratio_tick_label(value)))
    low, high = table.ci_low.min(), table.ci_high.max()
    ax.set_ylim(min(0.4, low / 1.1), max(4.5, high * 1.1))
    ax.set_xticks(range(len(CARDS)), [weights for _, weights in CARDS])
    ax.set_xlim(-0.5, len(CARDS) - 0.5)
    ax.set_ylabel("Cost Ratio", fontsize=style.PRINT_LABEL_PT, labelpad=1)
    ax.set_xlabel("Weights $w$ (in, cached, out)", fontsize=style.PRINT_LABEL_PT, labelpad=1)
    # centred on the page, not on the axes, so the label never runs past the right edge
    ax.xaxis.set_label_coords((0.5 - left) / width, -0.2)
    ax.tick_params(labelsize=style.PRINT_TICK_PT, length=2, pad=1)
    ax.grid(axis="y", color=style.MINOR_RULE, linewidth=style.MINOR_GRID_WIDTH)
    style.despine(ax, keep=("left", "bottom"))
    models = [Line2D([], [], marker="o", linestyle="none", color=palette.model_color(m), markersize=4,
                     label=name) for m, name in (("qwen38", "Qwen3.8-27B"), ("oss120b", "OSS-120B"))]
    shapes = [Line2D([], [], marker=palette.marker_style(c.marker), linestyle="none", color=style.MUTED, markersize=4,
                     label=c.label) for c in COMPARISONS]
    fig.legend(handles=[*models, *shapes], loc="lower center", bbox_to_anchor=(0.55, -0.01), ncol=2,
               fontsize=style.PRINT_LEGEND_PT, frameon=False, handletextpad=0.2, columnspacing=0.7,
               labelspacing=0.15, borderaxespad=0.0)
    violations = style.print_type_violations(fig)
    if violations:
        raise SystemExit(f"type below the print minimum: {violations}")
    # The page is the submitted one: no tight crop (style.apply sets savefig.bbox = tight).
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=style.SAVE_DPI * 2)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wt", type=pathlib.Path, required=True)
    ap.add_argument("--db", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    ps = analysis.load_module(args.wt / "statistics" / "paired_setups.py")
    table = ratios(ps, args.db)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    draw(table, args.out)
    spread = table.groupby("comparison").rho_c.agg(lambda r: r.max() / r.min())
    print(table.round(3).to_string(index=False))
    print("weighting factor (max / min rho_C) per comparison:", spread.round(2).to_dict())


if __name__ == "__main__":
    main()
