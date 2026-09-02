# Paper artifacts

Every number and figure in the paper, with the code that produced it. One directory per experiment;
nothing is shared between two experiments except the engine under `benchlib/`.

```
paper_artifacts/
  benchlib/            the shared engine: shard reading, the arm registry, pooling, the figure style
  experiments/
    llr8/              the thirteen-wave llr-focus40 campaign
    llr9/              llr8 with six kernels re-measured, and one dropped
    git/               kernel framing against repository framing
    canon/             the dace canonicalization ablation (no agents)
    evasion/           the anti-cheat sweep over submitted sources
  *.md                 every document, at this level
  requirements.txt reproduce.sh conftest.py
```

Each experiment holds `data/` (the collected rows), `figures/` (plots built from that data only),
`artifacts/` (what was submitted, plus a README describing it) and the `collect*` / `aggregate*` /
`plot_*` scripts that produced them. A script belongs to exactly one experiment and lives with it.

```bash
./reproduce.sh                  # every figure, from the committed CSVs -- Python >= 3.10 + matplotlib
./reproduce.sh --collect        # re-read the judge databases first (cluster only)
python3 -m pytest               # the tests, from this directory
```

`benchlib/` is the one thing not inside an experiment, because two experiments genuinely share it:
`shards` (judge databases to per-wave CSVs, and the registry of which job is which arm), `kernels`
(the pooling rules), `sources` (the final submitted source of each cell), `constructs` (what the
agents wrote), and `dumbbell` / `beforeafter` / `style` (the figures). The tag size and the
exclusion set are NOT in it: those are per-experiment decisions and each experiment's
`aggregate_*.py` states its own.

## The experiments

| experiment | what it varies | models | data |
|---|---|---|---|
| `llr8` | the skills packet, in the prompt or absent | Qwen3.8, GPT-OSS-120B, Kimi K2.7 | 15 waves, 480 kernel rows over 12 legs |
| `llr9` | the same, over the `llr-focus40` tag as re-cut on 2026-09-01 | the same three | 16 waves, 443 kernel rows over 12 legs |
| `git` | kernel framing against repository framing | Qwen3.8, GPT-OSS-120B | 4 arms, 120 cells |
| `canon` | dace canonicalization, on or off | none (a compiler ablation) | takes a sweep directory |
| `evasion` | nothing; it audits the submitted sources | all of the above | one candidate table |

Only three models appear in any agent figure: Qwen3.8, GPT-OSS-120B and Kimi K2.7. Qwen3-Coder-30B
ran only in the excluded first wave and is in no table here.

Every speed-up is reported as a GEOMEAN, never a median: a speed-up is a ratio, so the mean that
matches the product over the set is the geometric one, and the median of a mostly-flat distribution
reports "no effect" everywhere.

## An experiment is the union of its waves

The campaign does not fit in one job, so it is deliberately submitted in batches. `llr8` is fifteen
of them and `llr9` sixteen. This is a size constraint, not an accident, and it has consequences the
layout keeps visible rather than flattening away:

- **The dataset is the union.** One wave directory is never the experiment. Most waves are
  COMPLETION waves that re-run only the kernels an earlier arm never submitted, and `w8`/`w9` (and
  `w10`/`w11`) are two HALVES of one 40-kernel draw, split to fit the node budget. `w16` is the last
  of them and completes SPEED-UPS rather than solves: five arms over the six cells that had been
  drawn, sometimes solved, but had never left a trustworthy timing.
- **A kernel can appear in several waves, and the HIGHEST WAVE WINS.** A later wave is a later
  decision about the same kernel, so its submission is the one reported; the judge's millisecond
  stamp then breaks ties WITHIN that wave, which is where an agent that submitted several times for
  one kernel is resolved. A kernel whose winning wave carries no stamp, or whose latest stamp inside
  it is tied between two rows, reports no last value and says so in `ordering`. It is never resolved
  from directory order or row order.
- **It is deliberately NOT last-by-timestamp.** The two rules agree only while wave numbering and
  wall-clock order agree, and this launcher has broken that correspondence three times (see the
  table below). A wave carrying the previous wave's campaign token can finish at any wall-clock
  time, and under last-by-timestamp it would silently take the winner away from the wave that
  supersedes it. Measured when the rule was introduced, the two agree on **every** cell -- 477 of
  477 in llr8, 430 of 430 in llr9, of which 31 and 27 span more than one wave -- so no published
  number moved. It is there for the fourth instance.
- **The per-wave directories stay.** `data/w<N>/` is the provenance; `data/kernels.csv` is derived
  and sits beside them, not instead of them.
- **ATTRIBUTE BY `run_id` PREFIX *AND* JOB ID. Never by either alone, and never by directory name.**
  The launcher's campaign token has now mis-set the wave identity THREE times, so this is a standing
  defect and not a one-off:

  | # | what happened | what it cost |
  |---|---|---|
  | 1 | `w12` and `w13` inherited `RUN_ROOT` from the env file they came from, so one wave's jobs sit under two campaign-family DIRECTORIES | picking the family holding the wave's first job lost `w12`'s Kimi Fortran arm and two of `w13`'s three |
  | 2 | job 618083 runs under Slurm name `llr8w16b` but writes `run_id` prefix `llr8w16`, identical to 617920's | two jobs, one arm identity -- registering both would have double-pooled |
  | 3 | jobs 612240-612243 run under Slurm names `llr8w5-*` but write `run_id` prefix `llr8w4` | a whole wave wore the previous wave's name and went uncollected for four days |

  The job id is the only identifier that has never been wrong. The registry in `benchlib/shards.py`
  is what binds a job id to a wave, which is why the collection is registry-driven rather than a
  directory scan. Expect a fourth instance.

Each experiment's `artifacts/README.md` has the per-wave table: which jobs, how many arms, how many
kernels, and what that wave added that no earlier one had.

## What llr9 is

**llr9 is the experiment over the `llr-focus40` tag as the benchmark repository now holds it.** The
tag was re-cut on 2026-09-01 and stayed at forty: it took five kernels authored that day in, put
`tsvc_2_s2233` back, deleted the duplicate `tsvc_2_s13110`, and untagged the five kernels the new
ones replaced. llr9 follows the tag, so it is llr8's waves for the thirty-four kernels still in it,
the `llr40v9` campaign for the six that had to be re-measured, and nothing for the six the re-cut
removed. The rule is executed in `experiments/llr9/collect_llr9.py`, which names all three sets and
filters at COLLECTION time, so the wave CSVs never claim coverage the figures do not have.

**Six kernels are REFRESHED** -- taken from `llr40v9` (run root `llr40v9-20260902`) and filtered out
of every inherited llr8 wave:

- `compact_threshold_pack`, `scan_affine_decay`, `scatter_accum_dup`, `segment_reduce_ragged` and
  `versioned_distance_update` are new, authored 2026-09-01. They have no llr8 history, so for them
  the filter removes nothing.
- `argmax_with_index` is an old kernel RE-MEASURED. Its Fortran arm was unwinnable until
  2026-09-01: `out_index` is declared `index_array: true` and the ABI seam rebases index buffers in
  both directions, so a Fortran submission must store the **1-based** position. Nothing in the task
  said so and `skills/lang-fortran/SKILL.md` stated the opposite. Measured with the same loop body
  and only the stored base differing, 1-based grades `correct=True` 5/5 hidden and 0-based grades
  `correct=False, out_index: integer mismatch: 1 of 1 elements`. Its pre-2026-09-01 Fortran rows
  therefore measure the prompt, not the model. Its C rows are unaffected (`index_base` is 0 there)
  but are dropped with them: half a kernel measured under two prompts is not one kernel, and
  `llr40v9` re-ran C as well. In the new run it scores 15.4x to 20.4x in Fortran.

**One kernel is DROPPED as a duplicate.** `tsvc_2_s13110` was removed from the benchmark repository
as a byte-identical duplicate of `tsvc_2_s3110` -- same `_numpy.py`, same `_dace.py`, byte-identical
`_reference.c`, same presets and sizes. Its 80 llr8 calls and 21 llr8 submissions are **dropped, not
folded into `s3110`**: an agent that drew both drew the same code twice as two independent problems,
so merging the cells would report one kernel's two attempts as one attempt while doubling that
kernel's token cost.

**Five kernels are DROPPED as replaced.** `ext_break_find_first`, `ext_break_post_body`,
`tsvc_2_s232`, `wavefront2d` and `wf_north_west` were untagged in the same re-cut that took the five
new kernels in. They were not retired for being uninteresting: the three `ext_break` kernels
collapsed into one, the four wavefronts into two, and `s232` came out because agents were
hyper-specialising on it. llr8 measured all five; llr9 drops them, because carrying a kernel the
benchmark no longer offers would report a roster that does not exist. Three `llr8w12` arms drew
nothing but replaced kernels and so contribute no rows to llr9; the collector names them rather than
failing, because an arm the filter emptied is not an arm that returned nothing.

**Everything else is llr8's, unchanged**, because re-running a kernel nothing changed about would
spend a day of nodes measuring the same thing twice. That inheritance is mechanical -- llr9 takes
every REGISTERED llr8 wave -- so registering `llr8w16` added it to llr9 as well as to llr8. All five
kernels w16 touched are still in the re-cut tag, so none of them is filtered, and llr9 gains the
same three timed cells llr8 does.

`experiments/llr9/test_llr9_filter.py` checks the result against the tag itself, not against a
number: a hardcoded 40 passes for a table holding the wrong forty. It reads the manifests from an
`optarena` checkout beside this one and skips when there is none, with a size check standing in.

**The tag is 40; the pooled table holds 39.** The one missing is `tsvc_2_s2233`, which took 296
judge calls across llr8 and graded ok zero times in every arm of every wave -- a kernel no arm can
score measures the harness, not the model. It counts towards the denominator, because an arm was
given forty kernels and one of them cannot be scored; it is kept out of the table so its unscoreable
calls do not inflate the token cost of the kernels that can be. **llr8 is untouched by all of this**
and keeps its own forty: llr8 is the measurement that was actually taken.

## llr9 coverage: snapshot against final

The `llr40v9` half is INCOMPLETE at the time of collection and the tables say so rather than
smoothing it over. Re-running `collect_llr9.py` picks up whatever has landed since; nothing else
needs to change.

| arm | job | state at collection | reached of 6 |
|---|---|---|---|
| GPT-OSS-120B / C / base | 618220 | **final** (COMPLETED) | 6 |
| GPT-OSS-120B / Fortran / base | 618222 | **final** (COMPLETED) | 6 |
| GPT-OSS-120B / C / skills | 618229 | **final** (COMPLETED) | 6 |
| GPT-OSS-120B / Fortran / skills | 618231 | **final** (COMPLETED) | 6 |
| Qwen3.8 / C / base | 618217 -> **619183** | snapshot; TIMEOUT at 8h, rerunning at 16h | 3 |
| Qwen3.8 / Fortran / base | 618219 -> **619184** | snapshot; TIMEOUT at 8h, queued at 16h | 3 |
| Kimi K2.7 / C / base | 618223 -> **619185** | snapshot; TIMEOUT at 8h, rerunning at 16h | 4 |
| Kimi K2.7 / Fortran / base | 618225 -> **619186** | snapshot; TIMEOUT at 8h, queued at 16h | 4 |
| Qwen3.8 / C / skills | 618226 | snapshot; still running | 2 |
| Qwen3.8 / Fortran / skills | 618228 | snapshot; still running | 2 |
| Kimi K2.7 / C / skills | 618232 | snapshot; still running | 3 |
| Kimi K2.7 / Fortran / skills | 618234 | snapshot; still running | 2 |

**When 619183-619186 land they SUPERSEDE 618217/618219/618223/618225**, which is a registry edit and
not a merge: swap those four `job` values in `benchlib/shards.py` and re-run `collect_llr9.py`. Two
jobs writing the same arm name are two runs of one arm, and pooling them would count one agent's
draw twice.

**The C++ arms were cancelled** (618218, 618221, 618224) and are deliberately not registered: C++ is
not part of this experiment. Their rows are left in the shards.

Per cell, over the six refreshed kernels (`-` where the arm never reached the kernel):

| leg | argmax_with_index | compact_threshold_pack | scan_affine_decay | scatter_accum_dup | segment_reduce_ragged | versioned_distance_update |
|---|---|---|---|---|---|---|
| GPT-OSS-120B / C / base | 20.4x | no submission | 5.6x | no submission | 1.1x | 19.8x |
| GPT-OSS-120B / C / skills | 15.3x | 11.3x | no submission | no submission | no submission | 3.6x |
| GPT-OSS-120B / Fortran / base | 15.4x | no submission | 1.0x | 10.1x | no submission | 3.6x |
| GPT-OSS-120B / Fortran / skills | 3.8x | no submission | no submission | 9.5x | 11.2x | 3.6x |
| Kimi K2.7 / C / base | 20.8x | 10.3x | 6.6x | 19.0x | - | - |
| Kimi K2.7 / C / skills | 18.3x | 13.2x | no submission | - | - | - |
| Kimi K2.7 / Fortran / base | 18.1x | 12.0x | 5.0x | 10.6x | - | - |
| Kimi K2.7 / Fortran / skills | 18.8x | 12.9x | - | - | - | - |
| Qwen3.8 / C / base | 20.8x | 17.9x | 6.6x | - | - | - |
| Qwen3.8 / C / skills | 10.6x | 10.9x | - | - | - | - |
| Qwen3.8 / Fortran / base | 18.6x | 11.4x | 5.0x | - | - | - |
| Qwen3.8 / Fortran / skills | 20.4x | 11.2x | - | - | - | - |

`argmax_with_index` and `compact_threshold_pack` are the two every cell has now reached, which is
what makes their before-and-after readable today. The other four are a partial draw and the llr9
figures carry them as such. These numbers move every time the running arms are re-collected; the
four GPT-OSS rows are the only ones that will not.

## Where the raw data is

The judge writes one SQLite shard per rank under
`<RUN_ROOT>/[<campaign-family>/]<jobid>/judge/rank-*/hpcagent_bench*.db`, with the submitted sources
as blobs in a `*_prompts/` tree beside each shard. That is ~3 GB and is not in this repository. The
default run root is `/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`; pass `--run-root`
to point elsewhere.

`llr8` wave 1 is excluded and this is deliberate. The `llr8` campaign (jobs 608446-608987) ran on
2026-08-25, before the C reference fix of 08-26: 208 of 298 `_reference.c` in the corpus were
verbatim TSVC, so an agent that followed the reference it was shown built a shared library that
could not load and the judge recorded that as `incorrect`. Those C results measure the corpus, not
the model.

## What was deleted, and why

Everything below came from campaigns whose run roots no longer exist. Deleted rather than archived:
a stale CSV that still parses is the one that gets plotted.

| what | files | size | rows |
|---|---|---|---|
| `analysis/` -- llr6v10 skills ablation, campaign readout, arm dumps, HTML figure, probes | 22 | 151 KiB | 0 |
| `problems/problems-llr2-*.jsonl`, `problems-llr4-*.jsonl` | 4 | 12.3 MiB | 968 |
| `experiments/llr4-*` per-arm provenance directories | 21 | 68 KiB | 0 |
| `problems-llr8/` -- wave-1 packets; 3 of the 4 byte-identical to the kept set | 4 | 1.5 MiB | 160 |
| `figures/` -- all previous plots, regenerated per experiment | 8 | 892 KiB | 0 |
| **total** | **59** | **14.8 MiB** | **1128** |

Kept and moved: `problems/problems-llr6-*.jsonl` and `problems/problems-llr8kimi-*.jsonl` are the
packets the llr8 waves actually dispatched (the wave-2 packets carry llr6 filenames), and are now
under `experiments/llr8/artifacts/problems/`.

## What waves 5 and 16 were

Both ran on the cluster and were missing from the dataset until 2026-09-02. Both are now registered.
They are opposite kinds of correction and it is worth not confusing them.

### w5 (jobs 612240-612243) -- corrected the COST, changed no result

The registry used to say "a gap is fine (there is no w5)". There is. Slurm names it `llr8w5-*`; only
its `run_id` says `llr8w4`, which is why it read as a resubmission of the registered
612044/612045/612048/612049 rather than a wave of its own. It was submitted 29 minutes after the last
w4 job ended and all four arms COMPLETED.

It is a COMPLETION wave over w4's Fortran legs, on three independent facts: the runs are strictly
sequential rather than overlapping; the submitted-kernel sets of each w4/w5 pair are **disjoint, all
four pairs**; and w5 drew almost exactly the kernels its w4 counterpart called but never submitted.
A resubmission re-runs the same draw; this re-runs the complement, so treating it as superseding w4
would have discarded 50 accepted submissions over 21 cells w4 never produced.

Registering it added **no cell, no timed number and no solve, and moved no speed-up** -- all 19 of
its timed cells had been re-drawn by a strictly later wave (w12, w13, w14 or w16) that owns the
last-submission stamp. What it corrected is `tokens`, which sums over waves:

| leg | before | after | |
|---|---|---|---|
| Qwen3.8 / Fortran / base | 300.01M | 351.60M | +17.2% |
| Qwen3.8 / Fortran / skills | 233.90M | 244.54M | +4.5% |
| GPT-OSS-120B / Fortran / base | 225.37M | 232.92M | +3.4% |
| GPT-OSS-120B / Fortran / skills | 209.76M | 218.88M | +4.3% |

**That is a cost correction, not a measurement change.** Those four legs were under-reporting what
they spent because a wave they really ran was missing from the sum; the speed-up and success figures
are bit-for-bit what they were. The rise is smaller than w5's raw spend because `tsvc_2_s2233` is
excluded from the table and took 6% to 57% of each job's tokens.

Two things about w5 are NOT known and are recorded as unknown rather than guessed: whether it was
intended as the wave the registry calls w6 (also a completion wave over this campaign, run the same
day), and what wrote the `adhoc` `run_id` rows in 612044, which stay excluded from every count.

### w16 (jobs 617916-617919, 618083) -- filled holes, changed no cost

`llr8w16` was uncollected until 2026-09-02 and is now registered, because it fills holes and
duplicates nothing: after w15, llr8 had exactly SIX cells with no timed last submission, w16 ran one
arm per hole, and every cell it touched was one the figures had been plotting nothing for.

| hole | w16 result | outcome |
|---|---|---|
| Qwen3.8 / Fortran / base / `tsvc_2_s319` | 8.33x | filled |
| Qwen3.8 / Fortran / skills / `quasi_affine_reduce_odd` | 10.47x | filled |
| Kimi K2.7 / Fortran / skills / `tsvc_2_s323` | 1.01x, `solved` 0 -> 1 | filled |
| GPT-OSS-120B / Fortran / skills / `tsvc_2_s311` | `submit ok`, routed to `attempts` | still blank |
| GPT-OSS-120B / Fortran / base / `quasi_affine_reduce_odd` | `submit ok`, routed to `attempts` | still blank |
| GPT-OSS-120B / Fortran / base / `ext_break_capture` | 5 score calls, all incorrect | still blank |

The two `attempts` rows carry reason `nondeterministic-or-public-mismatch` -- the determinism gate,
a standing harness issue affecting 254 results campaign-wide, not a failure to optimise. They are
left exactly as they fell; re-litigating that gate inside a collection would mix a policy change
into a data fix.

**Job 617920 is deliberately not registered.** It writes the same `run_id` prefix as 618083
(`llr8w16-kimi27sglang-fortran-skills`; only the Slurm job name differs, `llr8w16b`), so registering
both would pool two runs of one arm and count one agent's draw twice. 618083 is kept, and NOT by the
usual "keep the job with the most calls" rule, which points the other way here: 617920 has 23 calls
to 618083's 20. It is kept because 617920 made no `submit` call at all -- 23 score calls, then the
job failed -- while 618083 reached the submit route and left the graded row. A run with more calls
and no submission went on longer, not further. This is why w16 contributes 70 calls where 93 carry
an `llr8w16-*` prefix on disk.
