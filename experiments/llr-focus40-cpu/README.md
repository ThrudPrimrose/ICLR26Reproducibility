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

## Command

```sh
./reproduce.sh              # data/llr-focus40-cpu.db -> figures/ + tables/, then check checksums
./reproduce.sh --extract    # first rebuild the .db from the judge databases (cluster only)
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

Neither intervention is statistically significant: none of the packet tests clears q < 0.36 and
none of the CPF tests clears q < 0.071. GLM-5.3 arms and the CPF page (`cpf`) arms are thin or
empty in places (loader/mount issues during the run); the figures reflect whatever rows exist.
