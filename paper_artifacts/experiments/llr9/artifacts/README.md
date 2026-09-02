# llr9 artifacts

llr9 is the experiment over the `llr-focus40` tag as the benchmark repository holds it after the
2026-09-01 re-cut: llr8's waves for the thirty-four kernels still in the tag, the `llr40v9` campaign
for the six re-measured, and nothing for the six the re-cut removed. The construction rule and why
each kernel is in its set are in `../collect_llr9.py` and the top-level README; this directory is
what was submitted.

## `sources/` -- the final submitted source of every cell

One file per `(model, language, skills, kernel)` cell, holding the submission with the latest judge
stamp -- the one `../data/kernels.csv` quotes as `last_speedup`. 430 files.

```
sources/<model>-<language>[-skills]/<kernel>__submitted.<ext>
sources/index.csv
```

`index.csv` carries `wave` beside `job` and `run_id`, so a refreshed kernel's file is visibly from
`v9` and an inherited one visibly from a `w<N>` wave.

## The waves

Sixteen batches: llr8's fifteen with the refreshed six, the duplicate and the five replaced
kernels filtered out, and `v9` for the refreshed six. w5 and w16 arrive here as they do in llr8 -- llr9
inherits every REGISTERED llr8 wave -- and the kernels they touched are still in the re-cut tag, so
almost none is filtered. Like in llr8, w5 moved no speed-up here either: it corrected `tokens` on
the two Fortran base/skills legs it ran and nothing else. Compare the `calls` column with llr8's to see
exactly what the filter removed.

| wave | jobs | arms | kernels drawn | calls | submissions | cells first solved here |
|---|---|---|---|---|---|---|
| w2 | 610653-610672 | 7 | 34 | 1546 | 281 | 131 |
| w3 | 611560-611567 | 8 | 34 | 913 | 154 | 87 |
| w4 | 612042-612051 | 10 | 30 | 1372 | 130 | 69 |
| w5 | 612240-612243 | 4 | 15 | 186 | 33 | 15 |
| w6 | 612291-612315 | 6 | 20 | 478 | 71 | 24 |
| w7 | 612301-612302 | 2 | 3 | 26 | 1 | 0 |
| w8 | 612477 | 1 | 13 | 181 | 28 | 12 |
| w9 | 612478 | 1 | 17 | 287 | 39 | 14 |
| w10 | 612995 | 1 | 14 | 311 | 40 | 13 |
| w11 | 612996 | 1 | 18 | 317 | 48 | 16 |
| w12 | 613035-613045 | 8 | 18 | 347 | 40 | 6 |
| w13 | 613252-613254 | 3 | 6 | 108 | 13 | 2 |
| w14 | 613359-613362 | 4 | 3 | 88 | 9 | 2 |
| w15 | 613533 | 1 | 11 | 321 | 49 | 3 |
| w16 | 617916-618083 | 5 | 5 | 70 | 5 | 1 |
| v9 | 618217-618234 | 12 | 6 | 762 | 108 | 37 |

w12 shows 8 arms against llr8's 11: three of its arms drew nothing but replaced kernels, so they
contribute no rows. The collector names them rather than failing -- an arm the filter emptied is not
an arm that returned nothing.

## Which cells are final and which are a snapshot

The inherited `w<N>` waves are FINAL: every job in them has long since ended.

`v9` is a SNAPSHOT. Four of its twelve arms are final, four timed out at 8h and have been
resubmitted at 16h as jobs **619183, 619184, 619185, 619186**, and four were still running when it
was collected.

| arm | job | state at collection | reached of 6 |
|---|---|---|---|
| GPT-OSS-120B / C / base | 618220 | final | 6 |
| GPT-OSS-120B / Fortran / base | 618222 | final | 6 |
| GPT-OSS-120B / C / skills | 618229 | final | 6 |
| GPT-OSS-120B / Fortran / skills | 618231 | final | 6 |
| Qwen3.8 / C / base | 618217 -> 619183 | snapshot, TIMEOUT, rerunning | 3 |
| Qwen3.8 / Fortran / base | 618219 -> 619184 | snapshot, TIMEOUT, queued | 3 |
| Kimi K2.7 / C / base | 618223 -> 619185 | snapshot, TIMEOUT, rerunning | 4 |
| Kimi K2.7 / Fortran / base | 618225 -> 619186 | snapshot, TIMEOUT, queued | 4 |
| Qwen3.8 / C / skills | 618226 | snapshot, running | 2 |
| Qwen3.8 / Fortran / skills | 618228 | snapshot, running | 2 |
| Kimi K2.7 / C / skills | 618232 | snapshot, running | 3 |
| Kimi K2.7 / Fortran / skills | 618234 | snapshot, running | 2 |

To finalise: when 619183-619186 land, swap those four `job` values into the matching `llr40v9`
entries in `../../../benchlib/shards.py` -- a resubmission SUPERSEDES the job it replaces, it does
not join it, and two jobs under one arm name would count one agent's draw twice -- then re-run
`../collect_llr9.py` and `../aggregate_llr9.py`. The still-running skills arms need no registry edit
at all; a re-collect picks them up where they are.
