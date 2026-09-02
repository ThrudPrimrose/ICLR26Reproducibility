# llr9 artifacts

llr8's waves for every kernel they still speak for, plus the `llr40v9` campaign for the six they do
not. The construction rule, and why each of the six is fresh, is in `../collect_llr9.py` and the
top-level README; this directory is what was submitted.

## `sources/` -- the final submitted source of every cell

Same shape and same rule as llr8: one file per `(model, language, skills, kernel)` cell, holding the
submission with the latest judge stamp -- the one `../data/kernels.csv` quotes as `last_speedup`.
482 files.

```
sources/<model>-<language>[-skills]/<kernel>__submitted.<ext>
sources/index.csv
```

`index.csv` carries `wave` beside `job` and `run_id`, so a refreshed kernel's file is visibly from
`v9` and an inherited one visibly from a `w<N>` wave.

## The waves

Fourteen batches: llr8's thirteen with the six refreshed kernels and the dropped duplicate filtered
out, and `v9` for the refreshed six. Compare the `calls` column with llr8's to see exactly what the
filter removed.

| wave | jobs | arms | kernels drawn | calls | submissions | cells first solved here |
|---|---|---|---|---|---|---|
| w2 | 610653-610672 | 7 | 38 | 1638 | 298 | 145 |
| w3 | 611560-611567 | 8 | 38 | 1057 | 184 | 102 |
| w4 | 612042-612051 | 10 | 34 | 1433 | 142 | 74 |
| w6 | 612291-612315 | 6 | 22 | 644 | 94 | 29 |
| w7 | 612301-612302 | 2 | 3 | 26 | 1 | 0 |
| w8 | 612477 | 1 | 15 | 232 | 31 | 14 |
| w9 | 612478 | 1 | 18 | 289 | 40 | 15 |
| w10 | 612995 | 1 | 17 | 353 | 49 | 16 |
| w11 | 612996 | 1 | 20 | 348 | 54 | 18 |
| w12 | 613035-613045 | 11 | 21 | 481 | 63 | 28 |
| w13 | 613252-613254 | 3 | 8 | 147 | 17 | 7 |
| w14 | 613359-613362 | 4 | 3 | 88 | 9 | 3 |
| w15 | 613533 | 1 | 14 | 443 | 63 | 3 |
| v9 | 618217-618234 | 12 | 6 | 583 | 79 | 32 |

`v9` IS A SNAPSHOT. Six of its twelve arms were still running when it was collected and two of its
completed arms had hit the wall clock with three or four of the six kernels reached; the C++ arms
were cancelled and are not registered. The per-cell coverage table is in the top-level README.
Re-running `../collect_llr9.py` picks up whatever has landed since.
