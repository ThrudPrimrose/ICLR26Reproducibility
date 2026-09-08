# canon -- the DaCe canonicalization ablation

What canonicalization is worth against the compilers, with no agents anywhere in the loop. Every
column runs the same forty kernels (`llr-focus40`) from the same generated references; the only
thing that varies is which toolchain compiles them.

## The figure

`figures/canon_speedup.{pdf,png}` -- median speedup over sequential C, one bar per column, with the
geometric mean marked separately.

Both statistics are drawn because either alone misleads here. `cc_autopar` and `numba` sit at
exactly **1.00x median** while their geomeans are **1.91x** and **1.86x**: each helps a great deal
on a handful of kernels and not at all on more than half, so a median-only figure reads as "these do
nothing" and a geomean-only figure reads as "these roughly double it". Neither is the story.

| column | median | geomean | n |
|---|---|---|---|
| C -O3 (baseline) | 1.00x | 1.00x | 40 |
| C -O3 + autopar | 1.00x | 1.91x | 40 |
| Numba | 1.00x | 1.86x | 40 |
| DaCe canon CPU | 7.01x | 7.76x | 40 |
| DaCe canon GPU | 66.83x | 53.72x | 32 |

`n=32` for the GPU column: eight kernels had no validated GPU row in this sweep. A speedup is taken
only over kernels the baseline AND the column both measured and validated, so the GPU bar is a
median over those 32 and not over a padded 40.

## Provenance

One sweep, one job, one node. Columns from two sweeps are not comparable and are never mixed here.

- run `canon-llr40-627149`, job **627149** on `nid002956`, 2026-09-07 22:18, 01:41:42 elapsed
- 4 ranks x 24 physical cores, one socket each -- the width an agent submission is graded at
- preset `fuzzed` (anchored on XL), the preset agent grades use
- DaCe `extended` at **902617707**, committed 22:17:53, one minute before the job started

That commit **predates** twelve correctness fixes landed on 2026-09-08. One of them changes this
table: `BandCarriedLoops` was banding four polybench stencils (`adi`, `heat_3d`, `jacobi_1d`,
`jacobi_2d`) whose dependences cross a band boundary, which is a race that only appears above one
thread. Those four are in the CPU column above and were measured from racy code. A re-run on a tree
carrying the fixes is the number to quote for the CPU column; this table is kept as what was
measured, not as what is current.

## Reproducing

```bash
python3 plot_canon_speedup.py                     # from the committed CSV, no cluster needed
python3 collect_canon.py --run-dir <sweep dir>    # cluster only: re-read a sweep's rank shards
```

`data/canon_llr40.csv` holds all seven columns the sweep measured, including `dace_cpu` and
`dace_gpu` (DaCe without canonicalization). The figure draws five of them: it asks what
canonicalization is worth against the compilers, not what DaCe is worth against itself.
