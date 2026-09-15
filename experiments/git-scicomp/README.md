# git-scicomp

## Question

Does handing the agent the repository a kernel lives in beat handing it the bare kernel?

## Conditions

Bare kernel vs whole repository. Three agent episodes per kernel.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, driven by Claude Code. 10 scientific-computing kernels. Speedups are
over C -O3 + autopar, not Numba: Numba runs 16-165x slower than C on these kernels and does not
finish the XL preset, which would credit the agent for Numba's own failure.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh"                     # data/git-scicomp.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/git-scicomp/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/git.pdf`, `tables/git.csv` | geometric mean speed-up and total tokens per arm, repository framing vs kernel framing |

## Data provenance

Run root `<RUNS>/git-scicomp-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. No jobs excluded.

## Caveats

Qwen3.8-27B is short of three episodes per kernel on this roster (none reached `fv3_dycore`), so
its coverage is partial.
