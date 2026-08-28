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

Colour identifies the MODEL; the skills condition is carried by fill (hollow vs solid) and line
style, so the comparison survives a reader who cannot separate the hues.

Usage:  python3 plot_llr8w2.py [--data data-llr8w2] [--out figures-llr8w2]
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the backend selection

#: Model -> hue. Same two-hue scheme as plot.py, extended for the wave-2 models. Chosen for CVD
#: separation rather than prettiness: blue/orange/green are distinguishable under all three common
#: deficiencies, which a red/green pair is not.
MODEL_COLOR = {"qwen38": "#2a78d6", "oss120b": "#eb6834", "kimi27sglang": "#1b8a5a"}

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
                       s=110,
                       color=colour if on else "none",
                       edgecolor=colour,
                       linewidth=1.8,
                       marker="o" if language == "c" else "s",
                       zorder=3)
            # Two unskilled arms land within 0.05x of each other at the bottom of this axis, so a
            # fixed offset overlaps them into an unreadable smear. Skilled arms label above;
            # unskilled ones label below and are staggered by model, which separates every pair
            # without hand-placing any single label.
            stagger = -12 - 10 * list(MODEL_COLOR).index(model)
            ax.annotate(f"{model} {language}{' +sk' if on else ''}", (x, y),
                        textcoords="offset points",
                        xytext=(8, 6) if on else (8, stagger),
                        fontsize=7.5,
                        color="#333333")
        if "0" in sides and "1" in sides:
            a, b = sides["0"], sides["1"]
            ax.annotate("",
                        xy=(float(b["tokens_per_solved"]) / 1e6, float(b["speedup_geomean"])),
                        xytext=(float(a["tokens_per_solved"]) / 1e6, float(a["speedup_geomean"])),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": colour,
                            "alpha": 0.55,
                            "linewidth": 1.4
                        })
    ax.margins(x=0.22)
    ax.set_xlabel("million tokens per solved kernel")
    ax.set_ylabel("geomean speed-up (arm level)")
    ax.set_title("Cost against return\narrow: skills off $\\rightarrow$ on", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)


def effect_panel(ax, matched: list[dict[str, str]]) -> None:
    labels, ratios, colours, notes = [], [], [], []
    for row in matched:
        labels.append(f"{row['model']}\n{row['language']}")
        ratios.append(float(row["paired_ratio"]))
        colours.append(MODEL_COLOR.get(row["model"], "#666666"))
        notes.append(f"n={row['paired_n']}  {row['sign_wins']}W/{row['sign_losses']}L  p={row['sign_p']}")
    pos = range(len(labels))
    ax.bar(pos, ratios, color=colours, alpha=0.85, width=0.55)
    ax.axhline(1.0, color="#333333", linewidth=1.0)
    for i, (r, note) in enumerate(zip(ratios, notes)):
        ax.annotate(note, (i, max(r, 1.0)), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("paired speed-up ratio (skills / no skills)")
    ax.set_title("Skills effect, paired on kernels\nboth arms timed", fontsize=10)
    ax.set_ylim(0, max(ratios + [1.0]) * 1.35)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


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
            ax.bar([i + offset for i in range(len(CONSTRUCTS))], [float(row[c]) for c in CONSTRUCTS],
                   width=width,
                   color=colour if skills == "1" else "none",
                   edgecolor=colour,
                   linewidth=1.0,
                   label=f"{model} {language}{' +skills' if skills == '1' else ''}")
    ax.set_xticks(range(len(CONSTRUCTS)))
    ax.set_xticklabels([c.replace("omp_", "") for c in CONSTRUCTS], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("fraction of accepted answers using it")
    ax.set_title("What the agents actually wrote\nhollow: skills off, solid: skills on", fontsize=10)
    ax.legend(fontsize=6.5, ncol=1, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data-llr8w2"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures-llr8w2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = read(args.data / "summary.csv")
    matched = read(args.data / "matched.csv")
    constructs = read(args.data / "constructs.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    cost_panel(axes[0], summary)
    effect_panel(axes[1], matched)
    adoption_panel(axes[2], constructs)
    fig.tight_layout()
    target = args.out / "llr8w2_overview.png"
    fig.savefig(target, dpi=170)
    print(f"  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
