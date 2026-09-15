# canon -- the DaCe canonicalization ablation

What canonicalization is worth against the compilers, with no agent in the loop: the same forty
`llr-focus40` kernels, one toolchain per column.

## Reproduce

- `$OPTARENA/experiments/submit-canon-llr40.sh` -- the sweep (cluster).
- `collect_canon.py --run-dir <sweep dir>` -- per-rank CSVs -> `data/canon_llr40.csv`.
- `plot_canon_speedup.py` -> `figures/canon_speedup.{pdf,png}`; its printed table is the result.

The denominator is **Numba**, the declared baseline of `loop_level_reasoning` and the one every
agent submission on the track is graded against (`--baseline cc` gives the sequential-C view).
Median and geometric mean are both drawn because each alone misleads: autopar and Numba help a lot
on a few kernels and not at all on most.

## Provenance

Sweep **631260**, 2026-09-10, all seven columns in one job on the promoted image, 4 ranks x 24
physical cores (one APU each, the grading width), preset `fuzzed`, DaCe `extended` at 01dc75795 2026-09-09 18:01.
`dace_cpu` is 39/40 (`fuse_diamond` crashes) and `dace_gpu` 38/40; every other column is 40/40.
Earlier sweeps ran on the image as it was being rebuilt and are not comparable.
