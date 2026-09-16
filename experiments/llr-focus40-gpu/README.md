# llr-focus40-gpu

Note: `data/llr-focus40-gpu.db` is not committed yet, so `reproduce.sh` exits 2 with one line naming it;
the databases land once the re-grade of rows graded before 2026-09-13 finishes (see Status in the top-level README).
`--extract` rebuilds it on the cluster.

## Question

Does the same packet help on the GPU, and how do the three delivery languages compare?

## Conditions

HIP, Triton (Python delivery), C + OpenMP offload, each with and without the Language Skill
Packet. Speedups are over the same Numba baseline as the CPU track.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, driven by Claude Code. Same 40 `llr-focus40` kernels,
on the MI300A GPU.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh"                     # data/llr-focus40-gpu.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/llr-focus40-gpu/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/gpu.pdf`, `tables/gpu.csv` | geometric mean speed-up and total tokens per arm, across HIP, Triton, OpenMP offload |
| `figures/paired_skills.pdf`, `tables/paired_skills.csv` | per-kernel paired speed-up/cost ratio, skills on vs off, with significance |
| `tables/paired_skills-absolute.csv` | the absolute per-kernel points the paired table divides, written beside it |

## Data provenance

Run root `<RUNS>/gpu-llr-focus40-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. Excluded: jobs 631274-631277, graded
while the judge refused Python, so every row in them is actually a C submission mislabeled Triton.

## Caveats

Triton coverage is low on every model (agents rarely land a correct kernel). Device residency
(the kernel actually running on the GPU, not falling back to host) is not enforced per submission.
