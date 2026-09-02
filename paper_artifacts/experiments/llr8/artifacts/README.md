# llr8 artifacts

What was submitted, and what the campaign was made of. The rows are in `../data/`, the plots in
`../figures/`; this directory is where a row is traced back to the code that produced it.

## `sources/` -- the final submitted source of every cell

One file per `(model, language, skills, kernel)` cell: the source of the submission with the LATEST
judge stamp among that cell's trustworthy submissions, which is the same submission
`../data/kernels.csv` quotes as `last_speedup`. 474 files.

```
sources/<model>-<language>[-skills]/<kernel>__submitted.<ext>
sources/index.csv
```

`index.csv` joins each file back to the row it came from: `model, language, skills, benchmark, wave,
job, run_id, ts_ms, speedup, sha256, n_bytes, path`. The `wave` column is how a file is attributed
to the batch that produced it -- a cell can be touched by several waves and only one of them holds
its last submission.

## `problems/` -- the packets the launcher dispatched

The wave-2 and later packets carry `llr6` filenames because that is what the launcher was pointed
at; they are the llr8 packets. `problems-llr8kimi-*.jsonl` are the Kimi batch draws, which take
their own success denominator (10 of 10 a batch drew is not 10 of 40).

## `provenance/` -- per-arm run records

`arm.env` and `submit.sh` for the wave-1 `llr8` arms, written by `../record_arm.py`. Wave 1 itself is
excluded from the dataset; see the top-level README. `llr8w2.md` beside them is wave 2's run record:
which job was which arm, its Slurm state and elapsed time, and how that wave was collected.

## The waves

Thirteen submission batches over the 40-kernel `llr-focus40` tag. "Cells first solved here" is what
the wave ADDED: a `(model, language, skills, kernel)` no earlier wave had ever solved.

| wave | jobs | arms | kernels drawn | calls | submissions | cells first solved here |
|---|---|---|---|---|---|---|
| w2 | 610653-610672 | 7 | 40 | 1675 | 308 | 153 |
| w3 | 611560-611567 | 8 | 40 | 1116 | 199 | 111 |
| w4 | 612042-612051 | 10 | 35 | 1455 | 146 | 76 |
| w6 | 612291-612315 | 6 | 23 | 646 | 95 | 30 |
| w7 | 612301-612302 | 2 | 3 | 26 | 1 | 0 |
| w8 | 612477 | 1 | 17 | 248 | 34 | 15 |
| w9 | 612478 | 1 | 18 | 289 | 40 | 15 |
| w10 | 612995 | 1 | 19 | 381 | 51 | 17 |
| w11 | 612996 | 1 | 20 | 348 | 54 | 18 |
| w12 | 613035-613045 | 11 | 22 | 488 | 63 | 28 |
| w13 | 613252-613254 | 3 | 9 | 167 | 20 | 8 |
| w14 | 613359-613362 | 4 | 3 | 88 | 9 | 3 |
| w15 | 613533 | 1 | 15 | 454 | 67 | 4 |

There is no w1 (excluded, see the top-level README) and no w5. w7 added nothing: both its arms ran
and neither solved a kernel an earlier wave had not. w8/w9 and w10/w11 are two halves of one draw.
w12 and w13 straddle two campaign-family directories, which is why jobs and not directory names are
what maps a run to a wave.
