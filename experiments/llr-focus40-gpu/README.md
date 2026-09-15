# llr-focus40-gpu

## Question

Does the same packet help on the GPU, and how do the three delivery languages compare?

## Conditions

HIP, Triton (Python delivery), C + OpenMP offload, each with and without the Language Skill
Packet. Speedups are over the same Numba baseline as the CPU track.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, driven by Claude Code. Same 40 `llr-focus40` kernels,
on the MI300A GPU.

## Command

```sh
./reproduce.sh              # data/llr-focus40-gpu.db -> figures/ + tables/, then check checksums
./reproduce.sh --extract    # first rebuild the .db from the judge databases (cluster only)
```

## Outputs

| file | how to read it |
|---|---|
| `figures/gpu.pdf`, `tables/gpu.csv` | geometric mean speed-up and total tokens per arm, across HIP, Triton, OpenMP offload |
| `figures/paired_skills.pdf`, `tables/paired_skills.csv` | per-kernel paired speed-up/cost ratio, skills on vs off, with significance |

## Data provenance

Run root `<RUNS>/gpu-llr-focus40-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. Excluded: jobs 631274-631277, graded
while the judge refused Python, so every row in them is actually a C submission mislabeled Triton.

## Caveats

Triton coverage is low on every model (agents rarely land a correct kernel). Device residency
(the kernel actually running on the GPU, not falling back to host) is not enforced per submission.
