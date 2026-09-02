"""The three per-kernel git figures: speed-up, success rate, token cost -- the llr9 trio, per framing.

Same drawing as ``plot_llr8_kernels.py`` and ``plot_llr9_kernels.py``, over the same tidy table
shape, with one substitution: the two legs are the KERNEL and REPOSITORY framings rather than the
absence and presence of a skills packet. Sharing the plotter is the point -- a reader who has read
the llr figures can read these without relearning the form.

The companion ``plot_git.py`` draws the ARM-level view (three panels, one row per model). This one
drops to the kernel and is where an arm that won on one lucky kernel stops looking like an arm that
won on the set.

Usage:  python3 plot_git_kernels.py [--data data/kernels.csv] [--out figures]
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import aggregate_git  # noqa: E402  -- sibling: the experiment's tag size lives with its rules
from benchlib import dumbbell  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

#: What the two dumbbell legs mean here. The hollow marker is the kernel framing because that is
#: the leg the pair started from -- the repository framing is the treatment being tested.
FRAMING = dumbbell.Contrast(
    key_labels=("kernel", "repo"),
    names=("kernel", "repo"),
    clause="under kernel and repository framing",
    tag_label="git-scicomp",
)

if __name__ == "__main__":
    sys.exit(
        dumbbell.main(ROOT / "data" / "kernels.csv", ROOT / "figures", "git", aggregate_git.TAG_SIZE, FRAMING))
