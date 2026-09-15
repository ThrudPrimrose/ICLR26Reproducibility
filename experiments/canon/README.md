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

## Command

```sh
./reproduce.sh              # data/canon.db -> figures/ + tables/, then check checksums
./reproduce.sh --extract    # first rebuild the .db from the sweep run directory (cluster only)
```

`--extract` runs `collect_canon.py --run-dir <sweep> --db data/canon.db` to turn the sweep's
per-rank output into the committed database, then `plot_canon_speedup.py --db data/canon.db --out
figures/` to draw the figure and print its table.

## Outputs

| file | how to read it |
|---|---|
| `figures/canon_speedup.pdf` | geometric mean speed-up per toolchain column over Numba (the headline number); median shown alongside as a spread cue only, never as the overall figure |
| printed table (from `plot_canon_speedup.py`) | the numeric result behind the figure |

## Data provenance

Sweep 631260, all seven columns in one job, 4 ranks x 24 physical cores (one APU each, the grading
width), preset `fuzzed`. No jobs excluded.

## Caveats

`dace_cpu` solves 39 of 40 kernels (`fuse_diamond` crashes) and `dace_gpu` solves 38 of 40; every
other column solves 40 of 40. Both median and geometric mean are reported because autopar and
Numba help a lot on a few kernels and not at all on most, so either alone misleads; the geometric
mean is still the headline column.
