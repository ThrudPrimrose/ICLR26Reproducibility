"""Render the llr8w2 (C vs Fortran, skills on/off) figures from data-llr8w2/*.csv.

Three questions, three panels, each in the form its answer needs:

  cost      tokens per SOLVED problem against the arm's geomean speed-up. A scatter, with an arrow
            joining each skills pair, because the question is directional: did paying more buy
            more? Totals would not be comparable -- the arms solved different numbers of problems.
  effect    the PAIRED skills ratio per (model, language), on kernels both arms timed. The
            arm-level geomeans in summary.csv are taken over different kernel sets, so their
            difference mixes the skills effect with which problems each arm happened to reach;
            this panel is the only one of the two that isolates the effect. Sign-test counts are
            annotated because a ratio from 14 kernels needs its n visible.
  adoption  construct rates, skills off against on. Bars, since the question is magnitude within a
            pair, and the constructs that stayed at zero are kept in the axis rather than dropped:
            "no arm ever emitted collapse" is the finding, and a chart that omits the row hides it.

Colour identifies the MODEL; the skills condition is carried by fill (hatched vs solid) and by
marker (hollow vs filled), so the comparison survives a reader who cannot separate the hues.
Palette, fonts and idioms come from artifact_style.

Usage:  python3 plot_llr8w2.py [--data data-llr8w2] [--out figures-llr8w2]
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the backend selection

import artifact_style  # noqa: E402  -- ditto, it imports pyplot itself

MODEL_COLOR = artifact_style.MODEL_COLOR

#: Constructs worth plotting, in the order a reader would ask about them: parallelism first, then
#: vectorisation, then the loop restructuring that turns out to be absent everywhere.
CONSTRUCTS = ("omp_parallel", "omp_simd", "omp_reduction", "omp_schedule", "restrict", "omp_collapse", "blocking",
              "do_concurrent", "unroll", "ivdep")


def read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def label(arm: str) -> str:
    """``llr8w2-qwen38-fortran-skills`` -> ``qwen38 fortran +skills``."""
    rest = arm.split("-", 1)[1]
    return rest.replace("-skills", " +skills").replace("-", " ")


def cost_panel(ax, summary: list[dict[str, str]]) -> None:
    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in summary:
        pairs.setdefault((row["model"], row["language"]), {})[row["skills"]] = row
    for (model, language), sides in pairs.items():
        colour = MODEL_COLOR.get(model, "#666666")
        for skills, row in sides.items():
            x = float(row["tokens_per_solved"]) / 1e6
            y = float(row["speedup_geomean"])
            on = skills == "1"
            ax.scatter(x,
                       y,
                       s=95,
                       color=colour if on else artifact_style.SURFACE,
                       edgecolor=colour,
                       linewidth=2.0,
                       marker="o" if language == "c" else "s",
                       zorder=3)
            # Two unskilled arms land within 0.05x of each other at the bottom of this axis, so a
            # fixed offset overlaps them into an unreadable smear. Skilled arms label above;
            # unskilled ones label below and are staggered by model, which separates every pair
            # without hand-placing any single label.
            stagger = -11 - 8 * list(MODEL_COLOR).index(model)
            ax.annotate(f"{model} {language}{' +sk' if on else ''}", (x, y),
                        textcoords="offset points",
                        xytext=(8, 6) if on else (8, stagger),
                        fontsize=6.5,
                        color=artifact_style.INK2)
        if "0" in sides and "1" in sides:
            a, b = sides["0"], sides["1"]
            ax.annotate("",
                        xy=(float(b["tokens_per_solved"]) / 1e6, float(b["speedup_geomean"])),
                        xytext=(float(a["tokens_per_solved"]) / 1e6, float(a["speedup_geomean"])),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": colour,
                            "alpha": 0.6,
                            "linewidth": 1.6
                        })
    ax.margins(x=0.22, y=0.22)
    ax.set_xlabel("million tokens per solved kernel")
    ax.set_ylabel("geomean speed-up (arm level)")
    artifact_style.axis_title(ax, "Cost against return")
    ax.grid(True)
    ax.set_axisbelow(True)


def effect_panel(ax, matched: list[dict[str, str]]) -> None:
    labels, ratios, colours, notes = [], [], [], []
    for row in matched:
        labels.append(f"{artifact_style.MODEL_LABEL[row['model']]} / {row['language']}")
        ratios.append(float(row["paired_ratio"]))
        colours.append(MODEL_COLOR.get(row["model"], "#666666"))
        notes.append(f"n={row['paired_n']}  {row['sign_wins']}W/{row['sign_losses']}L  p={row['sign_p']}")
    ax.set_xlim(0.0, max(ratios + [1.0]) * 1.25)
    artifact_style.row_axis(ax, labels, gridaxis="none")
    # Three rows in a panel this tall would otherwise draw bars an inch thick.
    ax.set_ylim(len(labels) + 0.8, -1.0)
    for row, (ratio, colour) in enumerate(zip(ratios, colours, strict=True)):
        artifact_style.bar_row(ax, row, ratio, colour, height=0.62)
        artifact_style.right_label(ax, row, f"{ratio:.2f}x")
    ax.axvline(1.0, color=artifact_style.INK2, linewidth=1.0, zorder=3)
    for row, note in enumerate(notes):
        ax.annotate(note, (0.012, row + 0.38),
                    xycoords=("axes fraction", "data"),
                    fontsize=6.0,
                    color=artifact_style.MUTED,
                    family="monospace",
                    ha="left",
                    va="center")
    ax.set_xlabel("paired speed-up ratio (skills / no skills)")
    artifact_style.axis_title(ax, "Skills effect, paired on kernels both arms timed")


def adoption_panel(ax, constructs: list[dict[str, str]]) -> None:
    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in constructs:
        pairs.setdefault((row["model"], row["language"]), {})[row["skills"]] = row
    complete = [k for k, v in pairs.items() if "0" in v and "1" in v]
    width = 0.8 / (len(complete) * 2)
    for slot, key in enumerate(complete):
        model, language = key
        colour = MODEL_COLOR.get(model, "#666666")
        for j, skills in enumerate(("0", "1")):
            row = pairs[key][skills]
            offset = (slot * 2 + j) * width - 0.4 + width / 2
            bars = ax.bar([i + offset for i in range(len(CONSTRUCTS))], [float(row[c]) for c in CONSTRUCTS],
                          width=width,
                          label=f"{artifact_style.MODEL_LABEL[model]} {language}"
                          f"{' + skills' if skills == '1' else ''}",
                          linewidth=0.0)
            for bar in bars:
                artifact_style.paint(bar, colour, skills == "1")
    ax.set_xticks(range(len(CONSTRUCTS)))
    ax.set_xticklabels([c.replace("omp_", "") for c in CONSTRUCTS], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("fraction of accepted answers using it")
    ax.set_ylim(0.0, 1.38)
    artifact_style.axis_title(ax, "What the agents actually wrote")
    ax.legend(fontsize=6.0, ncol=1, frameon=False, loc="upper right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def render(data: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Render the overview figure from ``data/*.csv`` into ``out``; returns the file written."""
    out.mkdir(parents=True, exist_ok=True)
    summary = read(data / "summary.csv")
    matched = read(data / "matched.csv")
    constructs = read(data / "constructs.csv")

    artifact_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    cost_panel(axes[0], summary)
    effect_panel(axes[1], matched)
    adoption_panel(axes[2], constructs)
    fig.tight_layout(pad=1.6, w_pad=3.0)
    return artifact_style.save(fig, out / "overview")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data-llr8w2"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures-llr8w2"))
    args = parser.parse_args()
    render(args.data, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
