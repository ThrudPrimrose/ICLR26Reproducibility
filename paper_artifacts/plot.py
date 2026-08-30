"""Render the paper figures from data/*.csv.

Form follows the job each metric does:

  success  a bounded fraction, so it is a row of bars over a FULL-SCALE track: the empty part of
           the track is the rest of the problem set, which is the comparison the number invites.
  speedup  a distribution, not a level. The median is 1.00 for six of seven arms in the first
           campaign while the tail reaches 12.5x, so a bar of medians would report "no effect"
           everywhere and hide the entire result. An ECDF on a log axis shows the whole
           distribution and needs no binning. One SMALL PANEL PER ARM PAIR rather than one panel
           per language: with four curves on an axis the two legs of a pair sit further apart than
           two legs of neighbouring pairs, and the legend that then becomes necessary has to go
           somewhere, which is how the old version ended up with a legend block in dead space.
  tokens   cost, and unbounded, so it is a DUMBBELL: hollow marker for the base leg, filled for the
           skills leg, one row per pair. Totals are not comparable across models that solved
           different numbers of problems, so this plots tokens per SOLVED problem.
  cost     the mechanism behind the two disagreeing success denominators, in two figures: tokens
           per KERNEL is what the skills packet moves, coverage is what that buys.

Colour identifies the model and nothing else; the skills condition is carried by fill (hatched for
off, solid for on) and by marker (hollow for off, filled for on), so a reader who cannot separate
the hues still reads the comparison. Palette, fonts and idioms all come from artifact_style.

Usage:  python3 plot.py [--data data] [--out figures]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import matplotlib
import matplotlib.lines
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- must follow the Agg backend selection

import artifact_style  # noqa: E402  -- ditto, it imports pyplot itself

MODEL_COLOR = artifact_style.MODEL_COLOR
LABEL = artifact_style.MODEL_LABEL

#: Row pitch of a paired row figure: the two legs sit 1.0 apart, the next pair starts here.
PAIR_PITCH = 2.45


def style() -> None:
    """Kept as the entry point every script already calls; the look itself lives in artifact_style."""
    artifact_style.apply()


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


def language_name(language: str) -> str:
    return language.upper() if language == "c" else language.capitalize()


def group_name(model: str, language: str) -> str:
    return f"{LABEL[model]} / {language_name(language)}"


def pair_rows(names: list[str]) -> tuple[list[str], list[float]]:
    """Row labels and y positions: the two legs of a pair sit closer than two neighbouring pairs."""
    labels: list[str] = []
    positions: list[float] = []
    for index, name in enumerate(names):
        labels += [name, "+ skills"]
        positions += [index * PAIR_PITCH, index * PAIR_PITCH + 1.0]
    return labels, positions


def leg(rows: list[dict[str, str]], model: str, language: str, skills: str) -> dict[str, str] | None:
    match = [r for r in rows if r["model"] == model and r["language"] == language and r["skills"] == skills]
    return match[0] if match else None


def missing(ax, row: int) -> None:
    """An arm that never finished is a gap with a word in it, never a zero bar."""
    ax.annotate("arm incomplete",
                xy=(0.0, row),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=6.5,
                color=artifact_style.MUTED,
                style="italic",
                ha="left",
                va="center")


def row_figure(names: list[str], title: str, xlabel: str, left: float = 0.235, right: float = 0.855) -> tuple:
    labels, positions = pair_rows(names)
    tall = 0.92 + 0.235 * (max(positions) + 1.24)
    fig, ax = plt.subplots(figsize=(6.6, tall))
    fig.subplots_adjust(left=left, right=right, top=1.0 - 0.48 / tall, bottom=0.60 / tall)
    artifact_style.axis_title(ax, title)
    ax.set_xlabel(xlabel)
    return fig, ax, labels, positions


def paired_bars(rows: list[dict[str, str]], field: str, scale: float, top: float, xlabel: str, title: str, fmt: str,
                out: pathlib.Path) -> None:
    """A bounded fraction, one row per arm, each drawn over a full-scale track.

    No series key: the rows name themselves, so the hatch is a second reading of the label rather
    than the only one, and the corner a key would need stays empty.
    """
    groups = cells(rows)
    fig, ax, labels, positions = row_figure([group_name(m, lang) for m, lang in groups], title, xlabel)
    ax.set_xlim(0.0, top)
    artifact_style.row_axis(ax, labels, positions, gridaxis="none")
    for index, (model, language) in enumerate(groups):
        for offset, skills in ((0, "0"), (1, "1")):
            row = positions[2 * index + offset]
            arm = leg(rows, model, language, skills)
            if arm is None:
                missing(ax, row)
                continue
            value = float(arm[field]) * scale
            artifact_style.bar_row(ax, row, value, MODEL_COLOR[model], on=skills == "1", height=0.72)
            artifact_style.right_label(ax, row, fmt.format(value))
    artifact_style.save(fig, out)


def paired_dumbbell(rows: list[dict[str, str]], field: str, scale: float, xlabel: str, title: str, fmt: str,
                    out: pathlib.Path) -> None:
    """An unbounded magnitude: one row per pair, hollow base leg to filled skills leg."""
    groups = cells(rows)
    labels = [group_name(m, lang) for m, lang in groups]
    tall = 1.15 + 0.36 * len(labels)
    fig, ax = plt.subplots(figsize=(6.6, tall))
    fig.subplots_adjust(left=0.235, right=0.845, top=1.0 - 0.72 / tall, bottom=0.66 / tall)
    artifact_style.axis_title(ax, title, pad=18.0)
    ax.set_xlabel(xlabel)
    artifact_style.row_axis(ax, labels)
    for row, (model, language) in enumerate(groups):
        legs = {s: leg(rows, model, language, s) for s in ("0", "1")}
        values = {s: float(a[field]) * scale if a is not None else None for s, a in legs.items()}
        if values["0"] is None and values["1"] is None:
            missing(ax, row)
            continue
        artifact_style.dumbbell(ax, row, values["0"], values["1"], MODEL_COLOR[model])
        if values["0"] is None or values["1"] is None:
            artifact_style.right_label(ax, row, "one leg only", artifact_style.MUTED)
            continue
        artifact_style.right_label(ax, row, fmt.format(values["1"] - values["0"]))
    ax.margins(x=0.12)
    artifact_style.key(ax, [("base", artifact_style.marker(artifact_style.INK2, on=False)),
                            ("+ skills", artifact_style.marker(artifact_style.INK2, on=True))],
                       anchor=(1.0, 1.005))
    artifact_style.save(fig, out)


def curve_panels(groups: list[tuple[str, str]], title: str, xlabel: str, ylabel: str) -> tuple:
    """One small panel per arm pair: two curves each, so no figure ever needs a series legend.

    The x label is set once under the row and the condition key sits on the figure's top rule, so
    nothing competes with the panels for the space the curves leave.
    """
    wide = 1.66 * len(groups) + 0.9
    fig, axes = plt.subplots(1, len(groups), figsize=(wide, 2.6), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.66 / wide, right=0.99, top=0.80, bottom=0.215, wspace=0.16)
    for ax, (model, language) in zip(axes[0], groups, strict=True):
        ax.set_title(group_name(model, language), fontsize=8.0, weight="bold", pad=6.0)
    axes[0][0].set_ylabel(ylabel)
    fig.text(0.5, 0.015, xlabel, ha="center", va="bottom", fontsize=8.0, color=artifact_style.INK2)
    artifact_style.eyebrow(fig, 0.008, 0.935, title)
    return fig, axes[0]


def log_speedup_axis(ax, ticks: list[float], limits: tuple[float, float]) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)


def curve(ax, xs: list[float], ys: list[float], colour: str, on: bool) -> None:
    ax.plot(xs,
            ys,
            drawstyle="steps-post",
            linewidth=1.9 if on else 1.5,
            color=colour,
            linestyle="-" if on else (0, (3.0, 2.0)),
            zorder=3)


def leg_key(fig) -> None:
    entries = [("skills off", matplotlib.lines.Line2D([], [], color=artifact_style.MUTED, linestyle=(0, (3.0, 2.0)))),
               ("skills on", matplotlib.lines.Line2D([], [], color=artifact_style.MUTED))]
    fig.legend([artist for _, artist in entries], [name for name, _ in entries],
               loc="upper right",
               bbox_to_anchor=(0.99, 1.0),
               ncol=2,
               frameon=False)


def speedup_ecdf(data_dir: pathlib.Path, out: pathlib.Path) -> None:
    """The speed-up distribution of every correct, non-suspect submission, one panel per arm pair.

    The old form put every arm of a language on one axis. Four curves then need a legend, the legend
    needs a corner, and the corner it lands in is the one the curves left empty -- which is how a
    figure ends up mostly white. Small panels label themselves.
    """
    with (data_dir / "submissions.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["speedup"] and r["suspect"] == "0"]
    groups = cells(rows)
    top = max(float(r["speedup"]) for r in rows)
    fig, axes = curve_panels(groups, "speed-up distribution of correct submissions", "speed-up over serial baseline",
                             "fraction of correct submissions")
    for ax, (model, language) in zip(axes, groups, strict=True):
        notes: list[tuple[str, str]] = []
        for skills in ("0", "1"):
            values = sorted(
                float(r["speedup"]) for r in rows
                if r["model"] == model and r["language"] == language and r["skills"] == skills)
            if not values:
                continue
            curve(ax, values, [(i + 1) / len(values) for i in range(len(values))], MODEL_COLOR[model], skills == "1")
            notes.append((f"{'skills' if skills == '1' else 'base'} n={len(values)}", MODEL_COLOR[model]))
        for index, (note, colour) in enumerate(notes):
            ax.text(0.96,
                    0.06 + 0.11 * (len(notes) - 1 - index),
                    note,
                    transform=ax.transAxes,
                    family="monospace",
                    fontsize=6.5,
                    color=colour,
                    ha="right",
                    va="bottom")
        log_speedup_axis(ax, [1, 2, 5, 10, 25, 50, 100], (0.93, top * 1.25))
        ax.set_ylim(0.0, 1.02)
    leg_key(fig)
    artifact_style.save(fig, out)


def matched(data_dir: pathlib.Path, out: pathlib.Path) -> None:
    """The fair comparison: each pair scored only on kernels BOTH arms reached.

    Carries n and the McNemar exact p, because at these sample sizes every delta is inside the
    noise and a bare pair of bars would invite the reader to believe otherwise.
    """
    with (data_dir / "matched.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    names = [f"{group_name(r['model'], r['language'])}\nn={r['common']}" for r in rows]
    fig, ax, labels, positions = row_figure(names,
                                            "Skills effect on matched problems (both arms reached)",
                                            "solved on the common subset (%)",
                                            right=0.80)
    ax.set_xlim(0.0, 100.0)
    artifact_style.row_axis(ax, labels, positions, gridaxis="none")
    for index, row in enumerate(rows):
        for offset, field in ((0, "rate_off"), (1, "rate_skills")):
            value = float(row[field]) * 100.0
            artifact_style.bar_row(ax,
                                   positions[2 * index + offset],
                                   value,
                                   MODEL_COLOR[row["model"]],
                                   on=offset == 1,
                                   height=0.72)
            artifact_style.right_label(ax, positions[2 * index + offset], f"{value:5.1f}")
        artifact_style.right_label(ax,
                                   positions[2 * index] + 0.5,
                                   f"{float(row['delta_pp']):+.1f}pp  p={float(row['mcnemar_p']):.2f}",
                                   artifact_style.MUTED,
                                   size=6.5,
                                   offset=44.0)
    artifact_style.save(fig, out)
    reached = sum(float(r["mcnemar_p"]) < 0.05 for r in rows)
    print(f"  matched: McNemar exact test on paired outcomes; {reached} of {len(rows)} pairs reach p<0.05")


def fastp(data_dir: pathlib.Path, out: pathlib.Path) -> None:
    """fast_p over the matched subset: fraction of problems solved AND at least p times faster.

    At p=1 this is the success rate, so one curve carries correctness and performance together.
    Computed from the per-problem best correct submission, on the kernels both arms of a pair
    reached, so the two sides face identical problems.
    """
    with (data_dir / "submissions.csv").open() as handle:
        subs = [r for r in csv.DictReader(handle) if r["speedup"] and r["suspect"] == "0"]
    with (data_dir / "calls.csv").open() as handle:
        reached: dict[str, set[str]] = {}
        for row in csv.DictReader(handle):
            reached.setdefault(row["arm"], set()).add(row["benchmark"])
    best: dict[str, dict[str, float]] = {}
    for r in subs:
        table = best.setdefault(r["arm"], {})
        table[r["benchmark"]] = max(table.get(r["benchmark"], 0.0), float(r["speedup"]))

    with (data_dir / "matched.csv").open() as handle:
        pairs = list(csv.DictReader(handle))
    groups = [(r["model"], r["language"]) for r in pairs]
    grid = [1.0 + 0.02 * i for i in range(0, 350)]
    fig, axes = curve_panels(groups, "fast_p on matched problems", "speed-up threshold p",
                             "solved and >= p times faster")
    for ax, row in zip(axes, pairs, strict=True):
        common = reached[row["arm_off"]] & reached[row["arm_skills"]]
        for side, arm in (("off", row["arm_off"]), ("skills", row["arm_skills"])):
            values = [sum(1 for b in common if best.get(arm, {}).get(b, 0.0) >= p) / len(common) for p in grid]
            curve(ax, grid, values, MODEL_COLOR[row["model"]], side == "skills")
        ax.text(0.96,
                0.06,
                f"n={len(common)}",
                transform=ax.transAxes,
                family="monospace",
                fontsize=6.5,
                color=MODEL_COLOR[row["model"]],
                ha="right",
                va="bottom")
        log_speedup_axis(ax, [1, 2, 4, 8], (1.0, 8.0))
        ax.set_ylim(0.0, 1.0)
    leg_key(fig)
    artifact_style.save(fig, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()

    matched(args.data, args.out / "matched")
    fastp(args.data, args.out / "fastp")
    rows = read_summary(args.data / "summary.csv")
    total = max(int(r["problems"]) for r in rows)
    paired_bars(rows, "success_rate", 100.0, 100.0, f"solved / {total} problems (%)",
                "Success rate at a fixed budget (yield)", "{:5.1f}", args.out / "success")
    paired_bars(rows, "success_rate_attempted", 100.0, 100.0, "solved / attempted (%)",
                "Success rate on problems actually reached (capability)", "{:5.1f}", args.out / "success_attempted")
    paired_bars(rows, "attempted", 1.0, float(total), f"problems reached (of {total})",
                "Coverage: how much of the set each arm got through", "{:4.0f}", args.out / "coverage")
    paired_dumbbell(rows, "tokens_per_solved", 1e-6, "million tokens per solved problem",
                    "Token cost per solved problem", "{:+.2f}", args.out / "tokens")
    paired_dumbbell(rows, "tokens_per_kernel", 1e-6, "million tokens per kernel reached",
                    "What one kernel costs: the price the skills packet moves", "{:+.2f}", args.out / "cost_per_kernel")
    speedup_ecdf(args.data, args.out / "speedup_ecdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
