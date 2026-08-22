#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired per-kernel view of one llr6 skills A/B, plus the packet's size history.

Pooling every submission hides the comparison that matters. Both legs of an arm submit many
times per kernel and lose different agents to startup failures, so a pooled distribution is
partly a statement about which kernels each leg happened to reach. The paired view takes each
leg's BEST submission per kernel and compares only kernels both legs graded -- the same kernel,
the same budget, one variable.

Usage: python3 plot_llr6_paired.py [--data data-llr6] [--out figures-llr6]
"""
import argparse
import csv
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Chars of skill text a C agent carries, per packet version. Measured from skill_history/, which
# snapshots what the agents actually read rather than what the tree says today.
PACKET_CHARS = {"v4": 6308, "v5": 6111, "v6": 7742, "v7": 13034}


def best_per_kernel(rows, arm):
    out: dict[str, float] = {}
    for row in rows:
        if row["arm"] != arm:
            continue
        speedup = float(row["speedup"] or 0)
        if speedup > 0:
            out[row["benchmark"]] = max(out.get(row["benchmark"], 0.0), speedup)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data-llr6"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures-llr6"))
    parser.add_argument("--arm", default="llr6-qwen30b-c")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((args.data / "submissions.csv").open()))
    base = best_per_kernel(rows, args.arm)
    skills = best_per_kernel(rows, f"{args.arm}-skills")
    shared = sorted(set(base) & set(skills))
    if not shared:
        raise SystemExit(f"no kernels graded by both legs of {args.arm}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))

    xs = [base[k] for k in shared]
    ys = [skills[k] for k in shared]
    lo, hi = 0.8, max(xs + ys) * 1.3
    left.plot([lo, hi], [lo, hi], color="0.6", linewidth=1, zorder=1)
    better = [i for i in range(len(shared)) if ys[i] > xs[i] * 1.02]
    worse = [i for i in range(len(shared)) if xs[i] > ys[i] * 1.02]
    left.scatter([xs[i] for i in better], [ys[i] for i in better], s=46, color="#2c7fb8",
                 label=f"skills faster ({len(better)})", zorder=3)
    left.scatter([xs[i] for i in worse], [ys[i] for i in worse], s=46, color="#d95f0e",
                 label=f"skills slower ({len(worse)})", zorder=3)
    tied = len(shared) - len(better) - len(worse)
    if tied:
        left.scatter([xs[i] for i in range(len(shared)) if i not in better + worse],
                     [ys[i] for i in range(len(shared)) if i not in better + worse],
                     s=40, color="0.5", label=f"within 2% ({tied})", zorder=2)
    # Name the collapses: these are the kernels where the control found a restructuring worth
    # many x and the skills leg settled for a directive worth ~1.
    for index in sorted(worse, key=lambda i: ys[i] / xs[i])[:4]:
        left.annotate(shared[index], (xs[index], ys[index]), fontsize=8,
                      xytext=(4, -9), textcoords="offset points", color="#8c3a00")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlim(lo, hi)
    left.set_ylim(lo, hi)
    left.set_xlabel("best speedup, skills OFF")
    left.set_ylabel("best speedup, skills ON")
    left.set_title(f"{args.arm}: {len(shared)} kernels both legs graded")
    left.legend(loc="upper left", fontsize=9)
    left.grid(alpha=0.3)

    versions = sorted(PACKET_CHARS)
    bars = right.bar(versions, [PACKET_CHARS[v] / 1000 for v in versions], color="#7fb8d8")
    bars[-1].set_color("#d95f0e")
    right.set_ylabel("C packet, thousands of characters")
    right.set_title("The packet has grown, not shrunk")
    right.grid(alpha=0.3, axis="y")
    for version, bar in zip(versions, bars):
        right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                   f"{PACKET_CHARS[version] / 1000:.1f}k", ha="center", fontsize=9)

    fig.suptitle("Skills A/B, paired per kernel -- and what the packet costs to say it", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"paired_skills.{suffix}", dpi=140)
    print(f"  {args.out}/paired_skills.png / .svg")
    print(f"  paired median   off={statistics.median(xs):.3f}  on={statistics.median(ys):.3f}")
    print(f"  per kernel      skills faster={len(better)}  slower={len(worse)}  tied={tied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
