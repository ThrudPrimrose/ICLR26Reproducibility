# llr8 artifacts

What was submitted, and what the campaign was made of. The rows are in `../data/`, the plots in
`../figures/`; this directory is where a row is traced back to the code that produced it.

## `sources/` -- the final submitted source of every cell

One file per `(model, language, skills, kernel)` cell: the source of the submission with the LATEST
judge stamp among that cell's trustworthy submissions, which is the same submission
`../data/kernels.csv` quotes as `last_speedup`. 477 files.

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

Fifteen submission batches over the 40-kernel `llr-focus40` tag. "Cells first solved here" is what
the wave ADDED: a `(model, language, skills, kernel)` no earlier wave had ever solved.

| wave | jobs | arms | kernels drawn | calls | submissions | cells first solved here |
|---|---|---|---|---|---|---|
| w2 | 610653-610672 | 7 | 40 | 1675 | 308 | 153 |
| w3 | 611560-611567 | 8 | 40 | 1116 | 199 | 111 |
| w4 | 612042-612051 | 10 | 35 | 1455 | 146 | 76 |
| w5 | 612240-612243 | 4 | 16 | 209 | 36 | 16 |
| w6 | 612291-612315 | 6 | 23 | 646 | 95 | 30 |
| w7 | 612301-612302 | 2 | 3 | 26 | 1 | 0 |
| w8 | 612477 | 1 | 17 | 248 | 34 | 15 |
| w9 | 612478 | 1 | 18 | 289 | 40 | 15 |
| w10 | 612995 | 1 | 19 | 381 | 51 | 17 |
| w11 | 612996 | 1 | 20 | 348 | 54 | 18 |
| w12 | 613035-613045 | 11 | 22 | 488 | 63 | 16 |
| w13 | 613252-613254 | 3 | 9 | 167 | 20 | 5 |
| w14 | 613359-613362 | 4 | 3 | 88 | 9 | 2 |
| w15 | 613533 | 1 | 15 | 454 | 67 | 4 |
| w16 | 617916-618083 | 5 | 5 | 70 | 5 | 1 |

There is no w1 (excluded, see the top-level README). w7 added nothing: both its arms ran
and neither solved a kernel an earlier wave had not. w8/w9 and w10/w11 are two halves of one draw.
w12 and w13 straddle two campaign-family directories, which is why jobs and not directory names are
what maps a run to a wave.

**w5 changed the cost figure and none of the results.** It was uncollected until 2026-09-02 because
the registry said a gap in the numbering was fine and there was no w5. There is: jobs 612240-612243,
Slurm names `llr8w5-*`, submitted 29 minutes after the last w4 job ended, all four COMPLETED. Only
the `run_id` says `llr8w4`, which is why it read as a resubmission of 612044/612045/612048/612049
rather than a wave.

It is a COMPLETION wave over w4's Fortran legs, on three independent facts: the runs are strictly
sequential rather than overlapping; the submitted-kernel sets of each w4/w5 pair are DISJOINT, all
four pairs; and w5 drew almost exactly the kernels its w4 counterpart had called but never
submitted. A resubmission re-runs the same draw -- this re-runs the complement.

Registering it added **no cell, no timed number and no solve**, and moved **no speed-up**: all 19 of
its timed cells had been re-drawn by a strictly later wave (w12, w13, w14 or w16) that owns the
last-submission stamp. What it corrected is `tokens`, which sums over waves, and only on the four
Fortran legs it ran:

| leg | tokens before | after | |
|---|---|---|---|
| Qwen3.8 / Fortran / base | 300.01M | 351.60M | +17.2% |
| Qwen3.8 / Fortran / skills | 233.90M | 244.54M | +4.5% |
| GPT-OSS-120B / Fortran / base | 225.37M | 232.92M | +3.4% |
| GPT-OSS-120B / Fortran / skills | 209.76M | 218.88M | +4.3% |

Read that as a cost correction, never as a measurement change: those legs were under-reporting what
they spent because a wave they really ran was missing from the sum. The rise is smaller than w5's
raw spend because `tsvc_2_s2233` is excluded from the table and took 6% to 57% of each job's tokens.

Two things about w5 are NOT known and are not guessed at here: whether it was intended as the wave
the registry calls w6 (also a completion wave over this campaign, run the same day), and what wrote
the `adhoc` `run_id` rows in 612044, which stay excluded from every count as usual.

**w16 is the SPEED-UP completion wave**, and its one new solve understates it. After w15 the
campaign had six cells that had been drawn, sometimes solved, but never left a trustworthy timing --
so the figures plotted nothing for them, and each was the missing half of a pair whose other leg WAS
timed. w16 ran one arm per hole and closed three of the six:

| hole | w16 got |
|---|---|
| Qwen3.8 / Fortran / base / `tsvc_2_s319` | 8.33x |
| Qwen3.8 / Fortran / skills / `quasi_affine_reduce_odd` | 10.47x |
| Kimi K2.7 / Fortran / skills / `tsvc_2_s323` | 1.01x, and `solved` 0 -> 1 |

Three remain, all GPT-OSS-120B Fortran, and the reasons are recorded rather than retried:
`ext_break_capture` (base) drew five score calls and every one graded incorrect, while `tsvc_2_s311`
(skills) and `quasi_affine_reduce_odd` (base) reached the submit route and were routed to `attempts`
with reason `nondeterministic-or-public-mismatch`. That is the determinism gate, a standing harness
issue affecting 254 results campaign-wide, not a failure to optimise; it is left exactly as it fell.

**Job 617920 is registered out of w16, deliberately.** It writes the same `run_id` prefix as 618083
(`llr8w16-kimi27sglang-fortran-skills`; only the Slurm job name differs, `llr8w16b`), so registering
both would pool two runs of one arm and count one agent's draw twice. 618083 is the one kept, and
NOT by the usual "most calls" rule, which points the other way here: 617920 has 23 calls to 618083's
20. It is kept because 617920 made no `submit` call at all -- 23 score calls, then the job failed --
while 618083 reached the submit route and left the graded row. More calls and no submission is a run
that went on longer, not further. This is why w16 shows 70 calls here against the 93 that carry a
`llr8w16-*` prefix on disk.
