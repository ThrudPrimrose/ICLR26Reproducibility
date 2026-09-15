# git-scicomp -- the repository framing across models

The agent is handed the repository a scientific-computing kernel lives in rather than the bare
kernel, over ten kernels with three agents each. This wave ran the repository arm on three models
and no kernel arm, so it prices the framing; the repository-vs-kernel A/B is the earlier two-model run.

Speedups are over **C -O3 + autopar**, the track's declared baseline: Numba runs 16-165x slower than
C on these kernels and does not finish XL, so it would credit the agent for its own failure.

## Reproduce

- `$OPTARENA/experiments/submit-git-scicomp.sh` with `LAYOUTS=repo` -- launch (cluster).
- `./collect.sh` -- judge databases -> `data/observations.csv` (cluster).
- `./plot.sh` -- figures and per-figure tables.

## Open

- Qwen3.8 is 9 agent episodes short of three per kernel (none on `fv3_dycore`).
