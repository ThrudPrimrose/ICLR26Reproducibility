# ICLR26 Reproducibility

Data and reproduction commands for the ICLR26 submission's agent-optimization experiments: LLM
agents rewriting numerical kernels for speed, scored against compiler and framework baselines on
CSCS Beverin (AMD MI300A nodes, 4 APUs per node, 24 Zen4 cores per APU). Each grade runs on one
APU's 24 physical cores. Scores, statistics and plots come from HPCAgent-Bench; this repository
holds only committed data and the commands that turn it into figures and tables.

## Folders

| folder | what it is |
|---|---|
| `experiments/<name>/` | one experiment: `data/<name>.db`, `figures/`, `tables/` (when it has tables), `reproduce.sh`, `SHA256SUMS` |
| `experiments/common.sh` | shared shell helpers every `reproduce.sh` sources (`extract`, `check`) |
| `skill_histories/` | every version of the **language packet** (Language Skill Packet) the agents read |
| `notes/` | earlier notes, kept for provenance, not part of the reproduction path |
| `requirements.txt` | Python dependencies for extraction and plotting |
| `HPCAGENT_BENCH_COMMIT` | the HPCAgent-Bench commit the committed figures were built with; a different commit prints a note |

## Experiments

| name | question | what varies | models | kernels |
|---|---|---|---|---|
| `llr-focus40-cpu` | does a packet or the Canonical Parallel Form help, on CPU? | no packet, Language Skill Packet, CPF page, CPF as source, perf playbook (perf-playbook-cpu); CPF conditions are C only | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, GLM-5.3 | 40 `llr-focus40` kernels, C and Fortran |
| `llr-focus40-gpu` | does the packet help on GPU, across delivery languages? | HIP, Triton (Python delivery), C + OpenMP offload, with/without Language Skill Packet | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | same 40 kernels, on the MI300A GPU |
| `llrblind` | does removing the score tool and capping submissions to one change the outcome? | Language Skill Packet on/off, C vs Fortran, one submission, no score route | GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code | same 40 kernels, CPU |
| `git-scicomp` | does handing the agent the whole repository beat handing it the bare kernel? | bare kernel vs repository, 3 agents per kernel | Qwen3.8-27B, GPT-OSS-120B | 10 scientific-computing kernels |
| `canon` | what does DaCe canonicalization buy against plain compilers, with no agent? | toolchain column: numba, cc, cc_autopar, dace_cpu, dace_cpu_canonicalize (+ GPU columns) | none, deterministic compiler baselines | same 40 kernels |

## Reproduce

Needs `git`, Python 3.12 or newer, `sha256sum`, and an HPCAgent-Bench checkout (the first command
below clones it). Run every command from this folder (the repository root).

```sh
git clone https://github.com/spcl/HPCAgent-Bench.git hpcagent-bench
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
HPCAGENT_BENCH="$PWD/hpcagent-bench" PYTHON="$PWD/.venv/bin/python" experiments/git-scicomp/reproduce.sh
```

Use HPCAgent-Bench's latest `main`, not a pinned commit.

Replace `git-scicomp` with any experiment name under `experiments/` to reproduce that one instead.
`HPCAGENT_BENCH` and `PYTHON` are read by `experiments/common.sh`, which every `reproduce.sh`
sources; both must be set for every run.

Each `reproduce.sh` rebuilds `figures/` (and `tables/`, where the experiment has them) from the committed `data/<name>.db` and
prints `OK: every figure and table matches SHA256SUMS` on success.

- `reproduce.sh --extract` first rebuilds the `.db` from the judge databases. This only works on
  the CSCS cluster, where the run directories live; `RUNS=<path>` overrides their default location.
- `reproduce.sh --record` rewrites `SHA256SUMS` instead of checking it.

## What the numbers mean

A per-kernel speedup is `median(baseline run time) / median(candidate run time)` over 20 timed
repeats each, credited only when a one-sided Mann-Whitney U test confirms the direction (p = 0.1);
otherwise the speedup is exactly 1.0, and a confirmed slow-down is below 1. That per-kernel value
is the credited speedup of the kernel's final answer. The baseline is Numba for the
loop-level-reasoning kernels (`llr-focus40-cpu`, `llr-focus40-gpu`, `llrblind`) and C -O3 + autopar
for `git-scicomp`. Scoring: within one agent episode the LAST verified submission counts; across an
arm's episodes the maximum is kept; suspect rows are dropped before any aggregate. Every OVERALL
speedup (per arm, per experiment, or any other summary) is the GEOMETRIC MEAN of the per-kernel
speedups, never the median: a speedup is a ratio, and a median over a mostly-flat distribution
reports "no effect" everywhere a geometric mean would not. Tokens are the total per kernel, summed
over every episode that touched it, never averaged. Every row in this repository is measured under
this one rule: rows originally graded before 2026-09-13 used an older rule and were re-timed so no
two numbers here come from different definitions.

## Full artifact

The earlier, more detailed artifact (per-wave provenance, campaign history, superseded readings)
is kept on branch `archive/paper-artifacts-20260915` and is not part of the reproduction path above.
