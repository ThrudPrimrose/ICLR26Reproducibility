"""Render the three paper figures from data/*.csv.

Form follows the job each metric does:

  success  magnitude compared WITHIN a skills pair -> grouped bars, delta annotated.
  speedup  a distribution, not a level. The median is 1.00 for six of seven arms while the tail
           reaches 12.5x, so a bar of medians would report "no effect" everywhere and hide the
           entire result. An ECDF on a log axis shows the whole distribution and needs no binning.
  tokens   cost. Totals are not comparable across models that solved different numbers of
           problems, so this plots tokens per SOLVED problem.

Colour identifies the model and nothing else; the skills condition is carried by fill and line
style, so a reader who cannot separate the hues still reads the comparison. Palette validated with
the dataviz validator: worst adjacent CVD dE 24.7 (protan), normal-vision 33.6, all checks pass.

Usage:  python3 plot.py [--data data] [--out figures]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

MODEL_COLOR = {"qwen30b": "#2a78d6", "oss120b": "#eb6834"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
LABEL = {"qwen30b": "Qwen3-Coder-30B", "oss120b": "gpt-oss-120b"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.dpi": 300,
    })


def read_summary(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def cells(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    """(model, language) groups in a stable order, so colour never tracks rank."""
    seen: list[tuple[str, str]] = []
    for row in rows:
        key = (row["model"], row["language"])
        if key not in seen:
            seen.append(key)
    return sorted(seen, key=lambda k: (list(MODEL_COLOR).index(k[0]), k[1]))


def paired_bars(rows: list[dict[str, str]], field: str, scale: float, ylabel: str, title: str, fmt: str,
                out: pathlib.Path) -> None:
    groups = cells(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width, positions = 0.34, range(len(groups))
    for index, (model, language) in enumerate(groups):
        colour = MODEL_COLOR[model]
        for offset, skills in ((-width / 2, "0"), (width / 2, "1")):
            match = [r for r in rows if r["model"] == model and r["language"] == language and r["skills"] == skills]
            if not match:
                # An arm that never finished is left as a gap, never as a zero bar.
                ax.text(index + offset,
                        0.01,
                        "arm\nincomplete",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color=MUTED,
                        style="italic")
                continue
            value = float(match[0][field]) * scale
            ax.bar(index + offset,
                   value,
                   width,
                   color=colour if skills == "1" else "white",
                   edgecolor=colour,
                   linewidth=1.4,
                   hatch=None if skills == "1" else "///",
                   zorder=3)
            ax.text(index + offset, value, fmt.format(value), ha="center", va="bottom", fontsize=8, color=INK)
        pair = {r["skills"]: r for r in rows if r["model"] == model and r["language"] == language}
        if "0" in pair and "1" in pair:
            delta = (float(pair["1"][field]) - float(pair["0"][field])) * scale
            top = max(float(pair[s][field]) * scale for s in ("0", "1"))
            # Above the pair, never below the axis: at the axis it collides with the tick labels.
            ax.annotate(f"{delta:+.1f}" if abs(delta) >= 1 else f"{delta:+.2f}",
                        xy=(index, top),
                        xytext=(0, 16),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=MUTED,
                        fontweight="bold")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{LABEL[m]}\n{lang.upper() if lang == 'c' else lang.capitalize()}" for m, lang in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=18)
    ax.margins(y=0.16)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="///", linewidth=1.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=INK, linewidth=1.4)
    ]
    ax.legend(handles, ["skills off", "skills on"], frameon=False, loc="upper right", fontsize=8)
    fig.text(0.012,
             0.015,
             "bold value above each pair is the skills-on minus skills-off delta",
             fontsize=7,
             color=MUTED)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}.png / .svg")


def speedup_ecdf(data_dir: pathlib.Path, out: pathlib.Path) -> None:
    """Faceted by language. Colour is the model and dash is the skills condition; language gets a
    panel rather than a third visual channel, because with both models and both languages on one
    axis the two solid blue lines (and the two solid orange) are indistinguishable."""
    with (data_dir / "submissions.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["speedup"] and r["suspect"] == "0"]
    languages = ["c", "fortran"]
    fig, axes = plt.subplots(1, len(languages), figsize=(7.6, 3.3), sharey=True)
    for ax, language in zip(axes, languages, strict=True):
        subset = [r for r in rows if r["language"] == language]
        for model in MODEL_COLOR:
            for skills in ("0", "1"):
                values = sorted(float(r["speedup"]) for r in subset if r["model"] == model and r["skills"] == skills)
                if not values:
                    continue
                fraction = [(i + 1) / len(values) for i in range(len(values))]
                ax.step(values,
                        fraction,
                        where="post",
                        linewidth=2.0,
                        color=MODEL_COLOR[model],
                        linestyle="-" if skills == "0" else "--",
                        alpha=0.95,
                        label=f"{LABEL[model]} - skills {'on' if skills == '1' else 'off'} (n={len(values)})")
        ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("speedup over serial baseline")
        ax.set_title(language.upper() if language == "c" else language.capitalize(), loc="left", pad=6)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_ylabel("fraction of correct submissions")
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Speedup distribution of correct submissions", x=0.012, ha="left", fontsize=10)
    fig.text(0.012, -0.02, "a step at x=1 is a correct answer that did not go faster", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}.png / .svg")


def matched(data_dir: pathlib.Path, out: pathlib.Path) -> None:
    """The fair comparison: each pair scored only on kernels BOTH arms reached.

    Carries n and the McNemar exact p on the plot, because at these sample sizes every delta is
    inside the noise and a bare bar pair would invite the reader to believe otherwise.
    """
    with (data_dir / "matched.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.34
    for index, row in enumerate(rows):
        colour = MODEL_COLOR[row["model"]]
        for offset, key, skills in ((-width / 2, "rate_off", False), (width / 2, "rate_skills", True)):
            value = float(row[key]) * 100
            ax.bar(index + offset,
                   value,
                   width,
                   color=colour if skills else "white",
                   edgecolor=colour,
                   linewidth=1.4,
                   hatch=None if skills else "///",
                   zorder=3)
            ax.text(index + offset, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        top = max(float(row["rate_off"]), float(row["rate_skills"])) * 100
        ax.annotate(f"{float(row['delta_pp']):+.1f} pp\np={float(row['mcnemar_p']):.2f}",
                    xy=(index, top),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=MUTED,
                    fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([
        f"{LABEL[r['model']]}\n{r['language'].upper() if r['language'] == 'c' else r['language'].capitalize()}"
        f"\nn={r['common']}" for r in rows
    ])
    ax.set_ylabel("solved on the common subset (%)")
    ax.set_title("Skills effect on matched problems (both arms reached)", loc="left", pad=22)
    ax.margins(y=0.26)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="///", linewidth=1.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor=INK, linewidth=1.4)
    ]
    ax.legend(handles, ["skills off", "skills on"], frameon=False, loc="upper right", fontsize=8)
    fig.text(0.012, 0.015, "McNemar exact test on paired outcomes; no pair reaches p<0.05", fontsize=7, color=MUTED)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}.png / .svg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()

    matched(args.data, args.out / "matched")
    rows = read_summary(args.data / "summary.csv")
    paired_bars(rows, "success_rate", 100.0, "solved / 242 problems (%)", "Success rate at a fixed budget (yield)",
                "{:.1f}", args.out / "success")
    paired_bars(rows, "success_rate_attempted", 100.0, "solved / attempted (%)",
                "Success rate on problems actually reached (capability)", "{:.1f}", args.out / "success_attempted")
    paired_bars(rows, "attempted", 1.0, "problems reached (of 242)",
                "Coverage: how much of the set each arm got through", "{:.0f}", args.out / "coverage")
    paired_bars(rows, "tokens_per_solved", 1e-6, "million tokens per solved problem", "Token cost per solved problem",
                "{:.1f}", args.out / "tokens")
    speedup_ecdf(args.data, args.out / "speedup_ecdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
