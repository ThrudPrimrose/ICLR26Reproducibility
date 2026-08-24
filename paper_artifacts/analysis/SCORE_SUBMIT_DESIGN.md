# score vs submit: what the focus40 data says about the design

Collected 2026-08-24 from 605458/605459 (qwen30b) and 605460/605461 (oss120b), C, all on the
`llr-focus40` tag. Fortran arms excluded until the ABI defect is fixed.

## The two routes today

`score` grades on the public seed, records nothing, and is free in grade terms.
`submit` runs the SAME XL-fuzzed preset with a different (hidden) seed and is the only
recorded grade. Best verified submission counts.

| arm | score calls | submit calls | submit ok | overfit | incorrect | score_error |
|---|---|---|---|---|---|---|
| qwen30b | 929 | 426 | 382 | 11 | 8 | 23 |
| qwen30b +skills | 911 | 407 | 360 | 16 | 12 | 16 |
| oss120b | 545 | 192 | 158 | 0 | 1 | 32 |
| oss120b +skills | 350 | 155 | 135 | 1 | 0 | 19 |

## What is already working -- do not "fix" it

1. **Agents submit what they should.** Agents that scored correct but never submitted:
   0/118 and 0/120 (qwen), 3/119 and 4/116 (oss). The llr5-era non-submission problem is
   gone on this tag. No auto-submit is needed to rescue lost results.
2. **The hidden seed earns its place.** It caught 11, 16, 0 and 1 overfits. A nonzero catch
   is the point: it is the only thing standing between "fits the public seed" and "correct".
   It must run on every recorded grade, always. Any redesign keeps it.
3. **Failed scores are exploration, not defects.** Incorrect submissions do NOT rise with
   skills (Fisher p = 0.369 / 1.000) while incorrect scores do (p = 0.008 / 0.012). The gate
   holds. Agents should be encouraged to try more, not less.

## The real defect: submit loses correct results to a judge OOM

90 calls across the four arms died with
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 3.88 GiB for an array with
shape (22820, 22820)`.

- **All 90 are on the `submit` route.** None on `score`.
- Concentrated in the three largest-array kernels: `wf_diff_skew` 54, `wf_triangular` 31,
  `wf_north_west` 5.
- For oss120b that is **32 of 192 submits (17%)** lost to a fault the agent did not cause and
  cannot fix -- its code was correct.

### Root cause, measured (supersedes the fragmentation reading below)

It is not fragmentation and not concurrency. Job 606482 reproduces the SAME OOM in a
single-node, single-grade stub run, and the traceback names the site:

    native_call.run_followup -> followup.reduce -> grading._grade_against
      -> grading.compare_arrays -> frameworks/utilities.py:63  denom = np.abs(e).copy()

`compare_arrays` runs INSIDE the timed child, which is capped at
`RLIMIT_AS = MEMORY_COPIES (2) x arrays`. Modelled directly (`scratchpad/as_probe.py`):

    RLIMIT_AS = current_vmsize + 7.76 GiB      (the harness formula at 22820^2)
      alloc 1 (3.88 GiB): OK
      alloc 2 (3.88 GiB): FAIL  ENOMEM        <- e and a alone exhaust the budget

The expected and actual arrays alone consume the ENTIRE child budget, before the comparison
allocates a single temporary -- and `compare_arrays` needs about six full-size temporaries
(`denom`, `e - a`, `np.abs(...)`, `rel`, `rel[both_finite]`, `np.allclose` internals),
roughly 23 GiB. The failure is arithmetic, not luck.

That also explains the submit/score asymmetry with no appeal to load: the hidden-case
comparison runs in the memory-capped CHILD, a plain `score` comparison runs in the uncapped
PARENT. `MEMORY_COPIES = 2` budgets for the DATA and not for the comparison, so the cap is
structurally too small for any kernel whose arrays approach it.

Note the cap is on VIRTUAL ADDRESS SPACE: the probe above failed on untouched `mmap`
reservations with zero RSS. Any allocator that reserves VA eagerly makes this WORSE.

Superseded reading, kept for the record: the node is not full. On an idle compute node
22820^2 allocates fine, `MemAvailable` 501 GB, no cgroup cap on the step, `ulimit -v`
unlimited -- which is what made fragmentation look like the explanation.

The mechanism is in `harness/scoring.py:verify_references`, which returns BOTH reference
outputs at once:

```python
return _numpy_reference(spec, data), _numpy_reference(spec, redata)
```

So a submit holds the public AND fresh-seed allocation sets live simultaneously. For a
single-array kernel like `wf_diff_skew` that is roughly seven copies alive at once --
`data`, `redata`, both numpy references, `o1` and `o2` (the determinism leg needs two agent
outputs), and `c_public` for the dual-oracle leg. `score` needs about three.

| XL LEN_2D | per array | per submit (7 live) | x10 concurrent grades |
|---|---|---|---|
| 22820 | 3.88 GiB | 27.2 GiB | **272 GiB** |
| 16746 | 2.09 GiB | 14.6 GiB | 146 GiB |
| 12000 | 1.07 GiB | 7.5 GiB | 75 GiB |

With 120 agents over 12-16 judge nodes (7.5-10 per node) and no concurrency bound anywhere
in the upstream service, the 22820 row is the failure. A 3.88 GiB CONTIGUOUS request fails on
fragmentation well before the node is full.

Peak RSS on a judge node during a submit burst is still unmeasured. It no longer gates the
diagnosis -- the child cap above is sufficient on its own -- but it still governs how many
ranks per node are safe.

### The size ladders are also wrong

| kernel | S | M | L | XL |
|---|---|---|---|---|
| wf_diff_skew | 64 | 12289 (1.13G) | 16746 (2.09G) | 22820 (3.88G) |
| wf_triangular | 64 | 17409 (2.26G) | 19932 (2.96G) | 22820 (3.88G) |
| wf_north_west | 64 | 12289 (1.13G) | 16746 (2.09G) | **12000 (1.07G)** |

`wf_north_west`'s XL is BELOW its own M and L. The 22820 -> 12000 change fixed the OOM by
making XL no longer the largest preset, so "XL" now grades an easier case than "L". Shrinking
one rung in isolation is what produced that; any size change has to move the whole ladder.

`wf_north_west` XL is already 22820 -> 12000 (3.88 -> 1.07 GiB). `wf_diff_skew` and
`wf_triangular` are still 22820 and hold 85 of the 90 losses.

## Why the peak is ~8x the arrays, not 2x

The one-at-a-time discipline exists only in the TIMED CHILD. `native_call.run_followup`
materialises one held-out input set, calls, reduces and drops it, against an RLIMIT_AS the
harness derives as `MEMORY_COPIES` (2) x arrays. That cap counts the DATA only, so it does
NOT protect the child -- it is what makes the child OOM once the grading comparison runs
inside it (see the measured root cause above).

The judge PARENT has no such cap, and `scoring.independent_verify` builds everything up front.
At `scoring.py:378`:

```python
o1, o2, ro = _run(data), _run(data), _run(redata)
```

live simultaneously: `data`, `redata` (built at line 348, unused until 378), `np_public`,
`np_re` (both from line 355, the second unused until the reverify leg), `o1`, `o2`, `ro`, and
then `c_pub`. Eight full array sets for a kernel whose declared arrays are one set.

`verify_references` also passes `[(REVERIFY_LABEL, lambda: redata)]` into `_run_c_reference` --
a builder in form only, closing over a dict that is already materialised. `run_followup`'s own
docstring is explicit that this is the pattern to avoid: "Followups arrive as builders rather
than as data because every one of them is the size of the public run ... handing them over as
dicts kept 6 full input sets resident at once."

Sequencing the three legs holds the peak at 4 sets and weakens nothing:

| leg | live sets |
|---|---|
| determinism (`o1`, `o2` vs `np_public`) | data, np_public, o1, o2 = 4 |
| dual-oracle (`o1` vs `c_pub`), after freeing `o2` | data, np_public, o1, c_pub = 4 |
| fresh-seed (`ro` vs `np_re`), after freeing the public set | redata, np_re, ro = 3 |

That is a 2x cut with no change to kernel sizes and no check removed. `redata` and `np_re`
should be deferred behind real builders, as the hidden-case path already does.

## Changes, in order of expected effect

0. **Size the child's cap for the comparison, not just the data**, or compare in chunks.
   This is the actual defect; every size change below is a workaround for it. `MEMORY_COPIES`
   must account for the ~6 full-size temporaries `compare_arrays` allocates, or the comparison
   must move out of the capped child.
1. **Cut XL for `wf_diff_skew` and `wf_triangular` to 12000**, matching the fix already
   applied to `wf_north_west` -- through `apply_sizes.py`, not by hand: the hand edit left
   `wf_north_west`'s XL BELOW its own M and L. Recovers ~85 lost grades per campaign.
2. **Bound grading concurrency per judge node by BYTES, not by request count.** A semaphore
   sized from the preset's array footprint lets ten small kernels run at once and serialises
   the 4 GiB ones. Without it the failure returns whenever a preset grows.
3. **Never charge the agent for a harness fault.** A `score_error` caused by a judge OOM
   should be retried server-side and must not consume the agent's attempt or appear as a
   failed grade -- today it reads to the agent as "my code failed", inviting it to break
   working code.
4. **Free the public-seed arrays before allocating the hidden-seed ones.** `submit` runs the
   preset twice; if both allocation sets are live it doubles the peak for no reason.
5. **Do not merge the routes, and do not drop the hidden seed.** The separation is what makes
   exploration free and recording trustworthy. If round budget needs relief -- oss120b lost
   40% of its iterations per kernel to skills-packet rent (15 -> 9) -- take it from the
   packet's length, not from the verification.

## Allocate on demand, and admit by bytes

Two separate faults, both visible at `scoring.py:344-380`.

**1. Preallocation for legs that have not started.** `redata` is built at line 348 and first
used at line 378. `np_re` is built at line 355 and first used in the reverify leg after the
determinism and dual-oracle legs have finished. Both sit resident through work that does not
touch them. `verify_references` compounds it by passing `[(REVERIFY_LABEL, lambda: redata)]`
into `_run_c_reference` -- an interface that exists to take BUILDERS, handed an already
materialised dict.

The discipline already exists in this codebase and is applied to hidden cases:

```python
hidden_data = [(case.label, functools.partial(_data_seeded, task.kernel, case.preset, ...))
               for case in cases]
```

`run_followup` then builds one, calls, reduces, and `del`s it. The verify path should do the
same: pass `functools.partial(_data_seeded, ..., reverify_seed)` rather than a built dict, and
let the fresh-seed set come into existence only when its leg runs. Combined with sequencing,
peak drops from 8 array sets to 4 -- and to 3 if the public set is released before the
reverify leg rather than after it.

**2. No admission control, so N parents allocate concurrently.** CPU serialises the timed
children naturally: one judge rank owns one socket, `OMP_NUM_THREADS=GRADE_CPUS`, and the
parent BLOCKS in `_call_isolated` while its child runs. So only one child is ever computing
per rank, and that is what keeps timings honest.

Memory does not serialise. uvicorn accepts requests concurrently and runs blocking handlers on
a threadpool, so several parents can each be materialising their own array sets while only one
child is on the cores. One child running, N parents resident -- that is the OOM, and it is why
"one child at a time" does not by itself bound memory.

The fix is a byte-budget semaphore at REQUEST admission, not at child spawn:

```
budget = node_ram_share - headroom
cost   = sets_in_flight * array_bytes(spec, preset)     # 3-4 after sequencing
```

A grade acquires `cost` before it allocates anything and releases after its last comparison.
Small kernels then run many-at-once; a 3.88 GiB wavefront kernel runs alone. This is what makes
the failure impossible rather than unlikely, and it is what lets `--ntasks-per-node=4` be safe:
four ranks x 4 sets x 3.88 GiB = 62 GiB, admitted against a known budget instead of hoped for.

Sizing the budget from the preset's declared arrays -- not from a fixed constant -- is what
stops the defect returning the next time a kernel grows. The current `XL_BYTE_CEILING` is a
per-kernel cap with no notion of how many grades share a node.

## Harvest fallback: an agent that never submits still has a result

What is recorded is what counts. An agent that times out, runs out of tokens, or loses its
submit to a harness fault has produced correct, graded work that the campaign then throws away.

The rule: **at run teardown, for every (agent, kernel) with no submission, promote the last
score that graded correct.** The judge already held that source when it graded it, so capture
it at score time -- keep the last correct-scoring source per (run, kernel) in the service and
promote at harvest, rather than trying to reconstruct it from the DB (`submissions` stores
`prompt_hash` and timings, not the source).

Promote through the SAME verification a submit gets, including the hidden seed. The agent is
gone, so there is no round to charge and no reason to weaken the gate -- and the hidden seed is
what separates "fits the public seed" from "correct" (it caught 11, 16, 0, 1 overfits here).
A promoted row must be flagged `promoted` so the analysis can tell an agent's own terminal
answer from a harvested one.

This is NOT the auto-submit that should be avoided. Auto-submitting every correct score would
run the hidden seed dozens of times per kernel and multiply the judge memory pressure that is
already the top defect. This runs it at most ONCE per (agent, kernel), only when the agent
produced no submission at all.

Expected recovery on focus40 is small -- non-submission is 0/118, 0/120, 3/119, 4/116 -- but it
is the correct floor, and it also catches the 90 submits lost to the judge OOM, which is where
the recoverable volume actually is (32 of 192 submits for oss120b).

## What NOT to do

- **No auto-submit on every correct score.** See the harvest fallback above for the narrow
  form that is correct: once per agent-kernel, only when nothing was submitted.
- **No discouraging exploration.** A failed score costs rounds, not correctness. The
  submission gate demonstrably holds.

## Seed separation: verified correct

`service.py:497` sets `hidden = route != "score"`, and that flag is the only thing that
separates the two routes' grading:

| | score | submit |
|---|---|---|
| public inputs | `seeds.public_tests` = 42 | `seeds.public_tests` = 42 |
| held-out correctness suite | not run | `hidden_cases()` under a per-process 8-byte `os.urandom` seed |
| independent re-verify fresh seed | not run | `seeds.reverify` = 777 |
| preset | body may override | body override IGNORED, always `cfg.preset` |

So an agent iterating on `score` never sees a hidden-seed verdict to overfit against, and a
recorded row always grades the run's configured size. Both properties hold in the code.

One inconsistency worth closing. Two different "hidden" seeds exist and their comments argue
OPPOSITE threat models for the same repo:

- `config.yaml:11` -- `reverify: 777`, "never returned to the agent, so a fixed constant is safe".
- `hidden_tests/__init__.py:31` -- "not a fixed public constant (the source is public, so a
  hard-coded seed could just be read off)", and draws `os.urandom(8)`.

Both survive today because the agent image ships no `hpcagent_bench` package, so `config.yaml`
never reaches the agent. But the reverify value is never needed reproducibly either -- a correct
kernel generalises to any inputs -- so drawing it from the same per-process random source costs
nothing and removes the disagreement. Defence in depth is currently carried entirely by
`hidden_tests`.

## Timed-work economics: the multiplier is cells, not reps

`measurement.repeat: 20` is reps per CELL, not per grade. The real cost is:

    timed runs = configs x perf.n_large_shapes x repeat x 2 sides x (1 + warmup)

| kernel | configs | cells | timed runs per grade |
|---|---|---|---|
| no config space | 1 | 3 | 3 x 20 x 2 x 2 = **240** |
| at `perf.max_configs` | 5 | 15 | 15 x 20 x 2 x 2 = **1200** |

**Do not cut the 20.** The backend is `mannwhitney_delta`, a rank test that credits a speedup
only when the shift is significant; it needs ~20 samples per side to have the power to declare
anything, which is exactly what `config.yaml:102` says (">= 20 keeps the CI meaningful"). Below
that the gate stops rejecting noise, which is the one thing it exists to do. 606482 is the
worked example of what happens when repeat drops: at `--repeat 1` the backend refuses outright
and the whole run reports nothing.

**Cut cells instead, and note they are nearly redundant.** `fuzz.xl_lo_mult` / `xl_hi_mult` are
0.85 / 1.15, so all three timed shapes are drawn within +/-15% of XL. Three shapes that close in
size mostly re-measure the same point -- which is what the 20 reps already do, better and more
cheaply. Either drop `n_large_shapes` to 2 (a straight 33% cut with little statistical loss) or
widen the band so the three shapes actually probe different size regimes. The current setting
pays for coverage it does not get.

**The cells are combined by GEOMEAN, not min** (`metric.py`): `raw_speedup` is the geomean over
timed cells, and the `gsd_z` dispersion gate floors the ranked score to 1.0 when
`geomean / gsd^z <= 1`. So a single bad shape does not veto the result the way a min would --
it drags the geomean and widens the gsd, and a wide enough spread is what trips the gate.

## On retrying an OOM

Retry-with-backoff is the wrong primary fix for the failure measured above, because that
failure is arithmetic and not transient: expected + actual alone equal the child's entire
`RLIMIT_AS` budget, so the retry has exactly as much room as the attempt that just failed and
fails identically. Sleeping buys nothing when nothing else on the node is holding the memory.

It IS the right backstop for the residual node-level case (several grades materialising at
once), where the pressure really is transient. Two conditions on doing it:

1. **Drop the traceback before retrying.** A Python exception holds the frames that hold the
   arrays, so an OOM traceback keeps the very allocations that caused it alive. Retrying inside
   the `except` block starts with LESS memory than the first attempt had. Bind the error, `del`
   the locals and clear `__traceback__` first.
2. **Prefer admission control.** A byte-budget semaphore prevents the overshoot instead of
   recovering from it, and it is deterministic. Retry is what catches what the budget mis-sized.

On reclaiming: for these array sizes glibc serves the request with `mmap` and returns it with
`munmap` on free, so RSS goes back to the OS as soon as the last reference drops -- no
`malloc_trim` needed. What DOES delay it is reference cycles (tracebacks again, and any cached
`VerifyResult` holding outputs), so an explicit `del` at the end of each leg plus a
`gc.collect()` between grades is the cheap correct reclaim. `malloc_trim(0)` only helps the
parent's many-small-allocation churn, not the big arrays.
