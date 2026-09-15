# canon

## Question

What does DaCe canonicalization buy against plain compilers, with no agent in the loop?

## Conditions

One toolchain per column, same 40 `llr-focus40` kernels: `numba`, `cc`, `cc_autopar`, `dace_cpu`,
`dace_cpu_canonicalize`, plus the equivalent GPU columns. Deterministic compiler baselines only,
run once each.

## Models and kernels

None (a compiler ablation, no agent). Same 40 `llr-focus40` kernels as the other `llr-focus40*`
experiments.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `CANON_SWEEP` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/canon/reproduce.sh"                     # data/canon.db -> figures/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/canon/reproduce.sh" --extract           # CSCS only: rebuild the .db from $CANON_SWEEP first
"$ARTIFACT_ROOT/experiments/canon/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
CANON_SWEEP=/capstor/scratch/cscs/ybudanaz/x86_64/canon-llr40-20260910 \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/canon/reproduce.sh
```

`--extract` runs `$HPCAGENT_BENCH/scripts/collect_canon.py --run-dir $CANON_SWEEP --db data/canon.db`.
Every run then draws the figure and its table with `$HPCAGENT_BENCH/scripts/plot_canon_speedup.py --db data/canon.db --out figures`.

## Outputs

| file | how to read it |
|---|---|
| `figures/canon_speedup.pdf`, `figures/canon_speedup.png` | geometric mean speed-up per toolchain column over Numba (the headline number); median shown alongside as a spread cue only, never as the overall figure |
| `figures/canon_speedup.csv` | the numbers behind the figure, one row per toolchain column |

## Data provenance

Sweep 631260, all seven columns in one job, 4 ranks x 24 physical cores (one APU each, the grading
width), preset `fuzzed`. No jobs excluded.

## Caveats

`dace_cpu` solves 39 of 40 kernels (`fuse_diamond` crashes) and `dace_gpu` solves 38 of 40; every
other column solves 40 of 40. Both median and geometric mean are reported because autopar and
Numba help a lot on a few kernels and not at all on most, so either alone misleads; the geometric
mean is still the headline column.
