# llr-focus40-cpu

## Question

Does a packet or a compiler-derived form help an agent write a faster CPU kernel?

## Conditions

No packet (control), Language Skill Packet, Canonical Parallel Form page (the CPF tool and its
page), Canonical Parallel Form as the starting source (cpfsrc), perf playbook (perf-playbook-cpu).
CPF conditions (`cpf`, `cpfsrc`) are C only; Fortran has no CPF spelling.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, GLM-5.3, driven by Claude Code. 40 `llr-focus40`
kernels, C and Fortran, one agent per kernel per arm.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh"                     # data/llr-focus40-cpu.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/llr-focus40-cpu/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/lang_skills.pdf`, `tables/lang_skills.csv` | geometric mean speed-up and total tokens per arm, no packet vs Language Skill Packet |
| `figures/cpfsrc.pdf`, `tables/cpfsrc.csv` | geometric mean speed-up and total tokens per arm, no packet vs Canonical Parallel Form as source (C only) |
| `figures/cpf.pdf`, `tables/cpf.csv` | geometric mean speed-up and total tokens per arm, no packet vs Canonical Parallel Form page (C only) |
| `figures/paired_skills.pdf`, `tables/paired_skills.csv` | per-kernel paired speed-up/cost ratio, skills on vs off, with significance |
| `figures/paired_cpfsrc.pdf`, `tables/paired_cpfsrc.csv` | per-kernel paired speed-up/cost ratio, cpfsrc vs no packet, with significance |

## Data provenance

Run root `<RUNS>/cpf-llr-focus40-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. Excluded: job 631231, cancelled while
the CPF forms directory was being overwritten.

## Caveats

Significance per contrast is in the `paired_*.csv` tables (the `*_verdict` columns). GLM-5.3 ran
only the C skills arm; the figures reflect the rows that exist.
