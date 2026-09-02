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
| `llr8` | the skills packet, in the prompt or absent | Qwen3.8, GPT-OSS-120B, Kimi K2.7 | 13 waves, 480 kernel rows over 12 legs |
| `llr9` | the same, over a kernel set corrected on 2026-09-01 | the same three | 14 waves, 497 kernel rows over 12 legs |
| `git` | kernel framing against repository framing | Qwen3.8, GPT-OSS-120B | 4 arms, 120 cells |
| `canon` | dace canonicalization, on or off | none (a compiler ablation) | takes a sweep directory |
| `evasion` | nothing; it audits the submitted sources | all of the above | one candidate table |

Only three models appear in any agent figure: Qwen3.8, GPT-OSS-120B and Kimi K2.7. Qwen3-Coder-30B
ran only in the excluded first wave and is in no table here.

Every speed-up is reported as a GEOMEAN, never a median: a speed-up is a ratio, so the mean that
matches the product over the set is the geometric one, and the median of a mostly-flat distribution
reports "no effect" everywhere.

## An experiment is the union of its waves

The campaign does not fit in one job, so it is deliberately submitted in batches. `llr8` is thirteen
of them and `llr9` fourteen. This is a size constraint, not an accident, and it has consequences the
layout keeps visible rather than flattening away:

- **The dataset is the union.** One wave directory is never the experiment. Most waves are
  COMPLETION waves that re-run only the kernels an earlier arm never submitted, and `w8`/`w9` (and
  `w10`/`w11`) are two HALVES of one 40-kernel draw, split to fit the node budget.
- **A kernel can appear in several waves.** It is de-duplicated by the judge's own millisecond
  stamp -- the LAST submission is the one the agent stood behind -- never by picking a directory. A
  kernel whose latest stamp is missing or tied reports no last value and says so in `ordering`.
- **The per-wave directories stay.** `data/w<N>/` is the provenance; `data/kernels.csv` is derived
  and sits beside them, not instead of them.
- **A directory name is not attribution.** Several waves' `.env` files mis-inherited `RUN_ROOT`, so
  one wave's jobs can sit under another wave's directory: `llr8w12` put ten arms under
  `llr8w4-20260829/` and its Kimi Fortran arm under `llr8w8-20260830/`. Job ids are unique and every
  row carries a `run_id` prefix, so both are read and the directory name is ignored.

Each experiment's `artifacts/README.md` has the per-wave table: which jobs, how many arms, how many
kernels, and what that wave added that no earlier one had.

## What llr9 is

**llr9 = llr8 for every kernel llr8 still speaks for, plus the `llr40v9` campaign for six it does
not, minus one that no longer exists.** The rule is executed in `experiments/llr9/collect_llr9.py`,
which names the three sets and filters at collection time, so the wave CSVs never claim coverage the
figures do not have.

**Six kernels are REFRESHED** -- taken from `llr40v9` (run root `llr40v9-20260902`, jobs
618217-618234) and filtered out of every inherited llr8 wave:

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
  `llr40v9` re-ran C as well. In the new run it scores 15.4x to 20.2x in Fortran.

**One kernel is DROPPED.** `tsvc_2_s13110` was removed from the benchmark repository as a
byte-identical duplicate of `tsvc_2_s3110` -- same `_numpy.py`, same `_dace.py`, byte-identical
`_reference.c`, same presets and sizes. Its 80 llr8 calls and 21 llr8 submissions are **dropped, not
folded into `s3110`**: an agent that drew both drew the same code twice as two independent problems,
so merging the cells would report one kernel's two attempts as one attempt while doubling that
kernel's token cost. `experiments/llr9/test_llr9_filter.py` asserts both halves of that decision.

**Everything else is llr8's, unchanged**, because re-running a kernel nothing changed about would
spend a day of nodes measuring the same thing twice.

That makes the llr9 kernel set 44: llr8's 40, less the duplicate, plus the five new ones.

## llr9 coverage: what is actually there

The `llr40v9` run is INCOMPLETE at the time of collection, and the tables say so rather than
smoothing it over. Re-running `collect_llr9.py` picks up whatever has landed since.

- **Leg 1 (no skills)** ran to the wall clock. `oss120b` C and Fortran completed and reached all six
  kernels; `qwen38` C and Fortran and `kimi27sglang` C and Fortran hit the 8-hour limit having
  reached three and four of the six.
- **Leg 2 (skills)** was still running at collection. Only `oss120b-fortran-skills` (618231) had
  completed; 618226, 618228, 618229, 618232 and 618234 were mid-run.
- **The C++ arms were cancelled** (618218, 618221, 618224) and are deliberately not registered: C++
  is not part of this experiment. Their rows are left in the shards.

Per cell, over the six refreshed kernels (`-` where the arm never reached the kernel):

| leg | argmax_with_index | compact_threshold_pack | scan_affine_decay | scatter_accum_dup | segment_reduce_ragged | versioned_distance_update |
|---|---|---|---|---|---|---|
| GPT-OSS-120B / C / base | 20.4x | no submission | 5.6x | no submission | 1.1x | 19.8x |
| GPT-OSS-120B / C / skills | 15.3x | 11.3x | no submission | no submission | no submission | - |
| GPT-OSS-120B / Fortran / base | 15.4x | no submission | 1.0x | 10.1x | no submission | 3.6x |
| GPT-OSS-120B / Fortran / skills | 3.8x | no submission | no submission | 9.5x | 11.2x | 3.6x |
| Kimi K2.7 / C / base | 20.8x | 10.3x | 6.6x | 19.0x | - | - |
| Kimi K2.7 / C / skills | 17.7x | - | - | - | - | - |
| Kimi K2.7 / Fortran / base | 18.1x | 12.0x | 5.0x | 10.6x | - | - |
| Kimi K2.7 / Fortran / skills | 18.8x | - | - | - | - | - |
| Qwen3.8 / C / base | 20.8x | 17.9x | 6.6x | - | - | - |
| Qwen3.8 / C / skills | 10.6x | - | - | - | - | - |
| Qwen3.8 / Fortran / base | 18.6x | 11.4x | 5.0x | - | - | - |
| Qwen3.8 / Fortran / skills | 20.2x | - | - | - | - | - |

`argmax_with_index` is the only one of the six every cell reached, which is what makes it the one
whose before-and-after is readable today. The other five are a partial draw and the llr9 figures
carry them as such.

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

## Two things this layout cannot express

**An uncollected llr8 wave 16 exists.** Five Fortran arms under jobs 617916-617920 and 618083 write
`run_id` prefixes `llr8w16-*` -- 93 calls and 5 submissions over one kernel per arm, physically
inside `llr8w4-20260829/` and `llr8w8-20260830/`. They are in no registry entry and therefore in no
table here. Registering a wave is a decision about what belongs to the campaign, so it is left to be
made rather than guessed at.

**Four more jobs write `llr8w4-*` run ids than the registry lists.** Jobs 612240-612243 carry the
same arm names as the registered 612044/612045/612048/612049 -- 209 calls and 33 submissions over
one to three kernels each. Whether they are a resubmission that supersedes those jobs or a second
batch alongside them is exactly what a registry exists to record and what the rows cannot say, so
they are reported here rather than merged.

**The `llr-focus40` tag has moved under llr9.** It was re-cut on 2026-09-01 to stay at 40: it took
the five new kernels in, put `tsvc_2_s2233` back, and untagged `ext_break_find_first`,
`ext_break_post_body`, `tsvc_2_s232`, `wavefront2d` and `wf_north_west`. llr9 KEEPS those five,
because dropping a measured kernel to match a tag re-cut after the measurement would change what the
experiment reports without anyone deciding to. `tsvc_2_s2233` stays excluded from both experiments:
it took 296 judge calls across llr8 and graded ok zero times in every arm of every wave.
