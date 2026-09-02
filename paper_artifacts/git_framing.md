# Does framing a kernel as a repository change what an agent does?

The git-scicomp A/B holds the model and the work fixed and moves only the FRAMING. One leg is
served a kernel by name, the way every other experiment in this artifact serves one. The other is
served the same kernel as a git repository with an `ISSUE.md`, a history, and the surrounding
files. Two models, ten scientific-computing kernels, three independent agents per cell: 120 cells,
`experiments/git/`.

The headline in the arm-level table is that repository framing more than triples Qwen3.8's
speed-up, 2.33x to 8.30x. **That number is one kernel.** This document is why it should not be
read as a framing effect, and what the experiment does support.

## The paired view

`experiments/git/data/kernels.csv` pools the three attempts into one row per kernel, and the
per-kernel figure narrows each pair to the kernels BOTH framings solved. Restricted that way, the
framing does essentially nothing:

| model | kernel | kernel framing | repo framing |
|---|---|---|---|
| GPT-OSS-120B | `heat_3d` | 2.45x | 2.40x |
| GPT-OSS-120B | `jacobi_2d` | 1.89x | 2.05x |
| Qwen3.8 | `heat_3d` | 2.50x | 2.52x |
| Qwen3.8 | `jacobi_2d` | 2.17x | 2.17x |

Geomean over the shared set: 2.15x -> 2.22x for GPT-OSS-120B, 2.33x -> 2.34x for Qwen3.8. On
`jacobi_2d` the two legs of the Qwen pair agree to three significant figures.

The 8.30x comes from `laplacian_stencil_3d`, solved by the repository leg alone, at **104.25x**
(572.8 ms of NumPy against 5.29 ms of gcc-compiled C). Nothing marks that cell as suspect -- it is
a graded submission against the seed committed in the task repo, and a fused C stencil beating a
temporaries-heavy NumPy one by two orders of magnitude is ordinary. It is a real number. It is just
not a framing result: it is the arithmetic of putting one 104x kernel into a geomean over three.

## What the framing does move

Not speed. Two other things, and they point the same way:

**More submissions get through.** Cells that reached a graded submission: 8 under kernel framing,
14 under repository framing. Both models submit more when they can see the repository. The
acceptance count barely follows (4 -> 5 of 120), so the extra submissions are mostly extra failures.

**It costs more, for one model.** Qwen3.8 spends 23.0M tokens under kernel framing and 41.8M under
repository framing -- 1.8x for one extra solved kernel. GPT-OSS-120B goes the other way, 35.2M down
to 28.2M. The token cost of framing is not a constant; it is a per-model interaction.

## The honest limit

Every arm sits near the floor: 2, 2, 2 and 3 kernels solved of ten. Nine correct cells out of 120.
At that density a single kernel moves an arm's geomean by a factor of three, which is exactly what
happened, and no per-kernel comparison here has the power to separate a 5% framing effect from
noise. The paired figure is the right way to look at the data BECAUSE it is so sparse: it is the
only view in which the two legs are being asked the same question.

What the experiment does establish is a null and a cost: on the kernels both framings solved, the
repository framing changed the speed-up by less than 3%, while changing the token bill by up to
1.8x. If the repository framing is worth having, this experiment does not show it in the speed-up.

## Reproducing

```bash
python3 experiments/git/collect_git.py      # cluster only: reads the judge databases read-only
python3 experiments/git/aggregate_git.py    # cell table  -> data/kernels.csv
python3 experiments/git/plot_git.py         # arm-level, three panels
python3 experiments/git/plot_git_kernels.py # per-kernel, the llr9 trio
```

`collect_git.py` and `aggregate_git.py` are both byte-reproducible over unchanged inputs. The
per-kernel figures are drawn by `benchlib/dumbbell.py`, the same code that draws llr8 and llr9,
with the two legs relabelled -- so the form a reader learned on those figures carries over.
