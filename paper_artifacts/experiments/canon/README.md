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

> **WITHDRAWN 2026-09-09.** `data/canon_llr40.csv` and the table that stood here were deleted, and
> no non-agentic number is published from this experiment until it is re-measured. Every canon
> sweep so far ran against the image that is being rebuilt now; a baseline measured on one
> toolchain is not a baseline for submissions graded on another, which is the whole contract this
> experiment exists to hold up. Re-run `submit-canon-llr40.sh` on the promoted image, then
> `collect_canon.py` and `plot_canon_speedup.py`, and put the new table here.
>
> What was withdrawn, for the record: sweep 627149 (DaCe `902617707`) read canon CPU 7.01x median
> and canon GPU 66.83x over `n=32`; sweep 628242 (DaCe `d137040ea`), which the deleted CSV held,
> read 10.24x and 94.84x over 40 and 40.

A speedup is taken only over kernels the baseline AND the column both measured and validated, so a
GPU bar is a median over the kernels that produced a validated row and never over a padded 40.

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

`collect_canon.py` writes all seven columns a sweep measured, including `dace_cpu` and `dace_gpu`
(DaCe without canonicalization). The figure draws five of them: it asks what canonicalization is
worth against the compilers, not what DaCe is worth against itself. `data/canon_llr40.csv` is
absent until a sweep on the promoted image regenerates it, so `plot_canon_speedup.py` has nothing
to read yet -- that is deliberate, not a missing file.
