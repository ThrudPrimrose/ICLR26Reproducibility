# Does framing a kernel as a repository change what an agent does?

The git-scicomp A/B holds the model and the work fixed and moves only the FRAMING. One leg is
served a kernel by name, the way every other experiment in this artifact serves one. The other is
served the same kernel as a git repository with an `ISSUE.md`, a history, and the surrounding
files. Two models, ten scientific-computing kernels, three independent agents per cell: 120 cells,
`experiments/git/`.

The arm-level table reports repository framing ahead in both models -- GPT-OSS-120B 3.38x to
4.04x, Qwen3.8 6.68x to 8.63x. **Neither number is a framing effect on speed.** This document is
why, and what the experiment does support instead.

## The arm-level comparison is not a comparison

The geomean of each arm is taken over the cells that arm got ACCEPTED, and the two legs do not
accept the same cells:

| model | framing | accepted cells | kernels solved |
|---|---|---|---|
| GPT-OSS-120B | kernel | 9 / 30 | 3 / 10 |
| GPT-OSS-120B | repo | 24 / 30 | 9 / 10 |
| Qwen3.8 | kernel | 16 / 30 | 9 / 10 |
| Qwen3.8 | repo | 15 / 30 | 7 / 10 |

GPT-OSS-120B's repository leg solved six kernels its kernel leg never solved at all (`addusxx_g`,
`dfa`, `edge_laplacian`, `kmp`, `lda_xc_potential`, `warpx_boris_push`). Those six enter the
repository geomean and cannot enter the other, because the other has nothing to put there. The
arm-level gap is which kernels each leg reached, not how fast it made them.

## The paired view

Restricted to the kernels BOTH framings solved, the framing does essentially nothing:

| model | shared kernels | kernel framing | repo framing | ratio |
|---|---|---|---|---|
| GPT-OSS-120B | 3 | 3.49x | 3.50x | 1.00x |
| Qwen3.8 | 7 | 9.67x | 11.06x | 1.14x |

GPT-OSS-120B is a null to three significant figures. Qwen3.8's 1.14x is one kernel:
`edge_laplacian` goes 23.67x to 117.47x, and `lda_xc_potential` moves the other way, 34.55x to
14.11x. The rest agree closely -- `fdtd_2d` 1.13x against 1.13x, `warpx_boris_push` 1.93x against
1.95x, `addusxx_g` 93.44x against 91.60x.

## What the framing does move

**Whether the agent finishes at all.** This is the large, consistent effect, and it is about
SUBMISSION rather than optimization. Cell outcomes:

| model | framing | ok | no submission | incorrect | no data |
|---|---|---|---|---|---|
| GPT-OSS-120B | kernel | 9 | 20 | 0 | 1 |
| GPT-OSS-120B | repo | 24 | 3 | 3 | 0 |
| Qwen3.8 | kernel | 16 | 10 | 0 | 4 |
| Qwen3.8 | repo | 15 | 5 | 2 | 8 |

Two thirds of GPT-OSS-120B's kernel-framing cells ended holding work they never submitted; under
repository framing that falls to one tenth. The repository leg is not writing faster code, it is
reaching the judge.

**Token cost, per model.** GPT-OSS-120B spends 141.0M tokens under kernel framing and 58.3M under
repository framing -- less than half, for two and a half times the accepted cells. Qwen3.8 goes
the other way, 91.3M to 101.7M. The cost of framing is a per-model interaction, not a constant.

## The honest limit

Ten kernels, three attempts, one campaign. The paired sets are 3 and 7 kernels wide, which is not
enough to separate a small framing effect on speed from noise -- and the one arm-level number that
looks like a large effect is the selection artifact above. What the experiment establishes is a
null on speed and a large effect on completion, and only the second is bigger than this design's
noise.

The non-submission column is the finding worth carrying: an agent that solved a kernel and never
submitted it scores zero, and framing changed how often that happened by a factor of six.

## Provenance

The rows here are the 2026-09-06 campaign. Two earlier campaigns (2026-09-01, 2026-09-04) were
collected and then withdrawn as superseded; the 09-01 run in particular sat near the floor at two
to three solved kernels per arm, where a single 104x cell moved an arm's geomean by a factor of
three.

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

`collect_git.py` writes each arm under `artifacts/<arm>/`, keyed by arm name alone. Collecting two
campaigns into one output directory therefore has the second delete the first's saved sources; give
each campaign its own `--out` if more than one is ever kept.
