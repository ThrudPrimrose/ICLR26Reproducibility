"""Per-kernel speed-up of every agent over the graded baseline, in log2 space.

One point per (agent, kernel): the median of that cell's accepted submissions, drawn as
``log2(speedup)`` so 0 is the baseline, +1 is twice as fast and -1 is twice as slow. A ratio is
multiplicative, so log space is the only one where a 2x win and a 2x loss are the same distance
from the line -- on a linear speed-up axis a slowdown is squeezed into [0, 1) and a speed-up gets
the whole half-plane, which reads as "nothing ever regresses".

The baseline is whichever the judge graded that submission against and it is NOT the same for every
campaign -- v9 and v10 graded against the C single-core lowering, v11 against numba. Mixing them on
one axis would compare two different denominators, so the campaigns are drawn separately and the
subtitle names the denominator being used.

Cells differ in how many accepted submissions they hold: 75 of v11's 165 have exactly one. A cell
with one observation has no interval, and inventing one from a single point is how a figure claims
a precision it does not have -- those are drawn as HOLLOW markers, the rest filled with a 95%
bootstrap interval on the cell median. The legend says which is which.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from hpcagent_bench import palette, plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402 -- pyplot must follow plotstyle.apply()

#: Bootstrap resamples per cell, and the seed that makes the published figure reproducible.
BOOTSTRAP: int = 2000
SEED: int = 0

#: Agents this figure knows, in the order the shared registry ranks them. The COLOURS come from
#: :mod:`hpcagent_bench.palette`, so an agent wears the same hue here as in every harness figure --
#: a reader carries colour between figures whether or not we intend them to.
AGENTS: tuple[str, ...] = ("oss120b", "qwen38", "kimi27sglang")
AGENT_COLORS: dict[str, str] = palette.colors("model", AGENTS)
INK, MUTED, RULE = plotstyle.INK, plotstyle.MUTED, plotstyle.RULE


def agent_of(arm: str) -> str:
    """The model an arm ran, which is the series identity -- ``v11w2-qwen38-c-skills`` -> qwen38."""
    for name in AGENTS:
        if f"-{name}-" in arm or arm.endswith(f"-{name}"):
            return name
    return "other"


def campaign_of(arm: str) -> str:
    """Which campaign an arm belongs to. Campaigns differ in their graded baseline."""
    if arm.startswith(("v11w2", "llr40v11")):
        return "v11"
    if arm.startswith("llr40v10"):
        return "v10"
    if arm.startswith("llr40v9"):
        return "v9"
    return "other"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile bootstrap interval on the MEDIAN of ``values`` (already in log2)."""
    draws = rng.choice(values, size=(BOOTSTRAP, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def cells(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (agent, language, kernel): median log2 speed-up, its interval, and n."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (agent, language, kernel), group in frame.groupby(["agent", "language", "benchmark"], sort=True):
        log2 = np.log2(group["speedup"].to_numpy(dtype=float))
        point = float(np.median(log2))
        low, high = bootstrap_ci(log2, rng) if log2.size > 1 else (point, point)
        rows.append({"agent": agent, "language": language, "benchmark": kernel,
                     "n": int(log2.size), "log2_speedup": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def draw(cell_frame: pd.DataFrame, baseline: str, campaign: str, out: pathlib.Path) -> pathlib.Path:
    """One panel per language, kernels down the y axis so their names stay readable."""
    languages = [lang for lang in ("c", "fortran", "cpp") if lang in set(cell_frame["language"])]
    # Kernels ordered by overall median, so the reader walks a gradient instead of an alphabet.
    order = (cell_frame.groupby("benchmark")["log2_speedup"].median().sort_values(ascending=False).index.tolist())
    height = max(4.0, 0.24 * len(order) + 1.6)
    fig, axes = plt.subplots(1, len(languages), figsize=(5.6 * len(languages), height),
                             squeeze=False, sharey=True)
    positions = {kernel: i for i, kernel in enumerate(order)}
    # Agents get a small vertical offset each so three points on one kernel never overplot.
    agents = [a for a in AGENTS if a in set(cell_frame["agent"])]
    offsets = np.linspace(-0.26, 0.26, len(agents)) if len(agents) > 1 else [0.0]

    for column, language in enumerate(languages):
        ax = axes[0][column]
        ax.axvline(0.0, color=INK, linewidth=1.1, zorder=2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=RULE, linewidth=0.6)
        panel = cell_frame[cell_frame["language"] == language]
        for agent, offset in zip(agents, offsets, strict=True):
            part = panel[panel["agent"] == agent]
            if part.empty:
                continue
            y = np.array([positions[k] for k in part["benchmark"]], dtype=float) + offset
            colour = AGENT_COLORS[agent]
            many = part["n"].to_numpy() > 1
            ax.hlines(y[many], part["ci_low"].to_numpy()[many], part["ci_high"].to_numpy()[many],
                      color=colour, linewidth=1.6, alpha=0.55, zorder=3)
            ax.scatter(part["log2_speedup"].to_numpy()[many], y[many], s=26, color=colour,
                       edgecolor="white", linewidth=0.6, zorder=4, label=agent if column == 0 else None)
            ax.scatter(part["log2_speedup"].to_numpy()[~many], y[~many], s=26, facecolor="none",
                       edgecolor=colour, linewidth=1.1, zorder=4)
        ax.set_title(f"{language}", pad=8)
        ax.set_xlabel("log2 speed-up over the graded baseline")
        plotstyle.despine(ax)

    axes[0][0].set_yticks(range(len(order)))
    axes[0][0].set_yticklabels(order, fontsize=7)
    axes[0][0].set_ylim(-0.8, len(order) - 0.2)
    axes[0][0].invert_yaxis()

    handles, labels = axes[0][0].get_legend_handles_labels()
    # The hollow key is added only when a hollow marker is actually drawn: a legend entry for a
    # style the figure does not use tells the reader to go looking for something that is not there.
    if bool((cell_frame["n"] == 1).any()):
        hollow = plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                            markeredgecolor=MUTED, markersize=6, label="single submission (no interval)")
        handles, labels = handles + [hollow], labels + [hollow.get_label()]
    axes[0][-1].legend(handles=handles, labels=labels, loc="lower right", frameon=False, fontsize=8)
    top = plotstyle.title(
        fig, f"{campaign}: per-kernel speed-up by agent",
        f"log2 of the median accepted submission, skills and non-skills arms pooled per model; "
        f"0 = the {baseline} baseline. Bars are 95% bootstrap intervals on the cell median. "
        f"Only submissions at or above the baseline are accepted, so no point can fall below 0.")
    fig.tight_layout(rect=(0, 0, 1, top))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=pathlib.Path, default=pathlib.Path("data/submissions_index.csv"))
    parser.add_argument("--campaign", default="v11", choices=("v9", "v10", "v11"))
    parser.add_argument("--baseline", default="numba", help="name of the graded denominator, for the subtitle")
    parser.add_argument("--languages", default="c,fortran", help="comma-separated; CPU languages by default")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/per_kernel_speedup_by_agent.pdf"))
    parser.add_argument("--table", type=pathlib.Path, default=pathlib.Path("data/per_kernel_agent_speedup.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.submissions, low_memory=False)
    frame = frame[frame["speedup"] > 0]
    frame["agent"] = frame["arm"].map(agent_of)
    frame["campaign"] = frame["arm"].map(campaign_of)
    frame = frame[(frame["campaign"] == args.campaign)
                  & frame["language"].isin(args.languages.split(","))
                  & (frame["agent"] != "other")]
    if frame.empty:
        raise SystemExit(f"no rows for campaign {args.campaign} in {args.submissions}")

    cell_frame = cells(frame)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    cell_frame.to_csv(args.table, index=False)
    written = draw(cell_frame, args.baseline, args.campaign, args.out)
    singles = int((cell_frame["n"] == 1).sum())
    print(f"{len(frame)} submissions -> {len(cell_frame)} cells "
          f"({singles} with a single submission, drawn hollow)")
    print(f"table  -> {args.table}")
    print(f"figure -> {written} (+ .png)")


if __name__ == "__main__":
    main()
