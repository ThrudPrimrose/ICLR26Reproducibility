# llrblind

## Question

Does removing the score tool and allowing only one submission change the outcome, versus the
normal iterate-and-score loop?

## Conditions

CPU, `llr-focus40` roster, C and Fortran, Language Skill Packet on and off. One submission per
episode (`AGENT_SINGLE_SUBMISSION=1`); no score route at all (`AGENT_SCORE_TOOL=0` withholds the
tool, `HPCAGENT_BENCH_SERVICE_SCORE_ENABLED=0` closes the HTTP route an agent could otherwise
call itself).

## Models and kernels

GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code, driven by Claude Code. Same 40 `llr-focus40` kernels.

## Command

```sh
./reproduce.sh              # data/llrblind.db -> figures/ + tables/, then check checksums
./reproduce.sh --extract    # first rebuild the .db from the judge databases (cluster only)
```

## Outputs

| file | how to read it |
|---|---|
| `figures/llrblind.pdf`, `tables/llrblind.csv` | geometric mean speed-up and total tokens per arm, no score tool, skills on/off |
| `tables/paired_arms.csv` | per-kernel paired ratios: C vs Fortran, and skills vs no-skills, per model, with significance |
| `tables/arms.csv` | geometric mean speed-up and total tokens per arm, behind the paired table |

## Data provenance

Run root `<RUNS>/llrblind-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. No jobs excluded.

## Caveats

The recorded speed-up is the judge's significance-gated minimum gain: a verified submission that
is slower or within noise records at exactly 1.0. `llrblind-qwen38-c-skills` ran under a tighter
global token cap than the other arms and most of its agents never reached submit; treat its
numbers as answer quality only, not as a coverage or submission-rate comparison against the other
arms.
