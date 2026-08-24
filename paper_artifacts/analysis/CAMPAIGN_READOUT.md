# focus40 campaign readout: skills, the judge, and the Fortran layout defect

Consolidates `SKILLS_ABLATION_LLR6V10.md`, `SKILLS_INCORRECT_RATE.md`, `SCORE_SUBMIT_DESIGN.md`,
`INFERENCE_AND_JUDGE_TUNING.md` and `FORTRAN_ABI_DEFECT.md`. Everything below is measured on the
`llr-focus40` tag only -- 605458/605459 (qwen30b, C), 605460/605461 (oss120b, C), 605696/605697
(qwen30b, Fortran). No other tag is mixed in.

## 1. Did skills increase FALSE SUBMISSIONS? No.

The premise came from counting `incorrect` over ALL judged calls, which mixes two different
events: `score` (free iteration, records nothing) and `submit` (the only recorded grade).

| route | qwen30b no-skills | qwen30b skills | oss120b no-skills | oss120b skills |
|---|---|---|---|---|
| incorrect **submissions** | 8 / 426 (1.9%) | 12 / 407 (2.9%) | 1 / 192 (0.5%) | **0 / 155 (0.0%)** |
| incorrect **scores** | 101 / 929 (10.9%) | 137 / 911 (**15.0%**) | 34 / 545 (6.2%) | 39 / 350 (**11.1%**) |

Two-sided Fisher exact:

| comparison | qwen30b | oss120b |
|---|---|---|
| incorrect submissions | p = 0.369 | p = 1.000 |
| incorrect scores | **p = 0.008** | **p = 0.012** |

**The submission gate holds.** Agents submit what already scored correct; oss120b went 1 -> 0.
What rises significantly, in both models, is incorrect SCORES -- wasted iteration on the free
route. The packet's cost is paid in rounds, not in recorded wrong answers.

### Why scores got worse

Construct adoption, measured from Write/Edit tool inputs (once per agent):

| construct | qwen no-skills | qwen skills | oss no-skills | oss skills |
|---|---|---|---|---|
| `omp simd` | 59% | **95%** | 53% | 46% |
| `collapse(...)` | 18% | **35%** | 0% | 5% |
| `ivdep` / `__builtin_assume` | 16% | **1%** | 18% | **2%** |

The packet moves agents toward `omp simd` and `collapse` -- both UNCHECKED assertions of
independence, wrong numbers with no diagnostic -- and away from `ivdep`/`assume`.

Second cost: **the packet buys fewer rounds.** oss120b drops from 15 to 9 calls per kernel
(-40%); qwen is flat at 31 -> 30. So rent alone explains oss120b but not qwen.

**What is NOT established:** construct choice does not cleanly predict incorrectness within an
arm, and the direction flips between models. "Aggressive constructs cause wrong answers" is
plausible and unproven -- it is confounded with kernel difficulty. Do not report it as a
mechanism.

### The corrective is cheaper tries, not fewer tries

A failed `score` is the agent experimenting on the route that is free and records nothing. The
levers are to make a try more informative:

1. Say what a construct ASSERTS and what its failure looks like, not "avoid it". Adoption is
   already high; the missing piece is the failure signature.
2. Name the false-dependence case. WAR/WAO are not real dependences -- privatise and the loop is
   parallel. `ext_war_unit` went 0/15 -> 7/37 on oss120b: a case where the agent should try MORE.
3. `reduction` on floats reassociates. At rtol 1e-9 a logically correct reduction can still miss.
   Diagnosis, not prohibition.
4. Give rounds back -- shorten the packet, not the verification.

Do not expect these to move speedup: the pooled skills ablation is p = 1.000 over 102 paired
kernels, and the Fortran layout rule was delivered verbatim and ignored.

## 2. Judge design as it stands today

### Ranks and CPUs

A node is 4 sockets x 24 cores = 96 physical cores (192 threads). `GRADE_CPUS` resolves to
`detect_cores_per_socket()` = 24, and `role_srun` launches the judge with
`--ntasks-per-node=1 --cpus-per-task=24`. **One rank per node uses a quarter of the node.**
Confirmed against the run dirs: 12 judge DBs for `JUDGE_NODES=12`, 16 for 16.

### Serialization: CPU serialises, memory does not

The timed child takes every core the rank owns (`OMP_NUM_THREADS=GRADE_CPUS`) and the parent
BLOCKS in `_call_isolated` while it runs. So exactly one timed child computes per rank -- that is
a CORRECTNESS constraint on the timings, not just a memory one.

Memory has no such discipline. uvicorn accepts requests concurrently and runs blocking handlers
on a threadpool, so several judge PARENTS can each be materialising their own array sets while
only one child is on the cores. **One child running, N parents resident.**

### Memory: eight array sets where the kernel declares one

`scoring.independent_verify` builds everything up front. At `scoring.py:378`:

```python
o1, o2, ro = _run(data), _run(data), _run(redata)
```

live simultaneously: `data`, `redata` (built line 348, unused until 378), `np_public`, `np_re`
(both line 355, second unused until the reverify leg), `o1`, `o2`, `ro`, then `c_pub`. Eight full
sets. `verify_references` also passes `[(REVERIFY_LABEL, lambda: redata)]` into
`_run_c_reference` -- a builder in FORM only, closing over an already-materialised dict, against
an interface whose own docstring says handing over dicts "kept 6 full input sets resident".

Sequencing the three legs holds the peak at 4 sets and removes no check:

| leg | live sets |
|---|---|
| determinism (`o1`, `o2` vs `np_public`) | 4 |
| dual-oracle (`o1` vs `c_pub`), after freeing `o2` | 4 |
| fresh-seed (`ro` vs `np_re`), after freeing the public set | 3 |

| judge ranks/node | today (8 sets) | sequenced (4 sets) |
|---|---|---|
| 1 | 31 GiB | 15 GiB |
| 4 | 124 GiB | **62 GiB** |

Sequencing is what makes 4 ranks/node safe on a ~500 GiB node. Do it first.

### Agent counts and utilisation

Both models: `INFERENCE_NODES=1`, `AGENTS_PER_NODE=120`, TP=4, no PP.

| arm | agents | mean Running | inference util | mean Waiting | zero-gen |
|---|---|---|---|---|---|
| qwen30b | 120 | 38.5 | **32%** | 0.02 | 0% |
| qwen30b +skills | 120 | 45.6 | 38% | 0.47 | 0% |
| oss120b | 120 | 12.9 | **11%** | 0.11 | 0% |
| oss120b +skills | 120 | 15.8 | 13% | 1.22 | 3.8% |

Nothing is queued at the engine and aggregate throughput rises monotonically with concurrency.
There is no stall and no saturation -- **the engine is STARVED**. 68-89% of agent-seconds are
spent somewhere other than the LLM. So: SGLang will not help here (its 4.3x win was against a
kimi PP=4 stall that does not exist in these arms), do not add inference nodes, do not raise
`AGENTS_PER_NODE`.

Because the engine batches, unblocking agents is free speed: qwen30b runs at 2.3 tok/s per agent
at Running=30 and 7.05 tok/s at Running=120.

### Judge sizing

| arm | ranks | grades/h/rank | grades/h/agent | median gap | mean gap |
|---|---|---|---|---|---|
| qwen30b | 12 | 28.8 | 2.88 | 35.5 s | 125 s |
| oss120b | 16 | 41.2 | 5.49 | 25.4 s | 87 s |

An agent demands 3-5.5 grades/hour; a rank delivers 28-41/hour. Median arrival gap is 3-4x below
the mean, so arrivals are BURSTY. With inference at 11-38% and the judge at ~30%, neither tier is
throughput-saturated: **the system is LATENCY-bound** on the write -> grade -> read round trip.

    judge_ranks >= agents * 5 / 100      (2x burst headroom)
    judge_nodes  = ceil(ranks / 4)       once --ntasks-per-node=4 lands

| agents | ranks (2x headroom) | NODES at 4 ranks/node |
|---|---|---|
| 40 | 4 | **1** |
| 120 | 12 | **3** |

Caveat: the 100 grades/h saturated-rank figure is INFERRED, never measured -- these arms never
saturated a rank. Measure it before trusting the table. Rank imbalance is 1.6-2.5x busiest to
idlest, which is free capacity.

## 3. Verification, and why submits failed more than scores

`score` grades on the public seed and records nothing. `submit` runs the same XL-fuzzed preset
under a per-process 8-byte secret seed (`hidden_tests.HiddenCase` carries preset, seed, variant,
config) and is the only recorded grade. The hidden seed caught 11, 16, 0 and 1 overfits across
the four arms -- a nonzero catch is the whole point, and it must run on every recorded grade.

**90 calls died with** `Unable to allocate 3.88 GiB for an array with shape (22820, 22820)`.
All 90 on `submit`, none on `score`, concentrated in `wf_diff_skew` (54), `wf_triangular` (31),
`wf_north_west` (5). For oss120b that is **32 of 192 submits (17%) lost to a fault the agent did
not cause and cannot fix** -- its code was correct.

### Root cause, sharpened by an isolated reproduction (job 606482)

The earlier reading blamed judge-parent concurrency plus allocator fragmentation. A single-node,
single-grade stub run reproduces the SAME OOM with no concurrency at all, and the traceback names
the exact site:

    native_call.run_followup -> followup.reduce -> grading._grade_against
      -> grading.compare_arrays -> frameworks/utilities.py:63  denom = np.abs(e).copy()

`compare_arrays` allocates several full-size temporaries per comparison -- `denom`, `e - a`,
`np.abs(...)`, `rel`, `rel[both_finite]`, plus `np.allclose`'s own -- and it runs INSIDE the
timed child, which is capped at `RLIMIT_AS = MEMORY_COPIES (2) x arrays` = 7.76 GiB at this size.

That explains the submit/score asymmetry directly: **the hidden-case comparison runs in the
memory-capped child, while a plain `score` comparison runs in the uncapped parent.** It is not
primarily a concurrency effect. `MEMORY_COPIES = 2` budgets for the DATA and not for the
comparison, so the cap is structurally too small for any kernel whose arrays approach it.

### Fixes, in order

1. Count comparison temporaries in the child's memory budget, or compare in chunks. This is the
   actual defect; the size ladder is a workaround for it.
2. Cut XL for `wf_diff_skew` and `wf_triangular` from 22820 to 12000 -- but through
   `apply_sizes.py`, not by hand. The hand edit to `wf_north_west` left XL (12000) BELOW its own
   M (12289) and L (16746), so "XL" now grades an easier case than "L".
3. Sequence the verify legs and defer `redata`/`np_re` behind real builders. Halves grade peak.
4. Admit grades by BYTES, not request count: `cost = sets_in_flight * array_bytes(spec, preset)`
   acquired before allocation. Small kernels run many-at-once; a 3.88 GiB wavefront runs alone.
5. Never charge the agent for a harness fault. A judge-side `score_error` reads to the agent as
   "my code failed", inviting it to break working code.

Do NOT add auto-submit: 0-4 agents lost a result to non-submission on this tag, and it would run
the hidden seed on every correct score -- multiplying the exact pressure that is the top defect.

## 4. Fortran: one ABI defect, not model capability

| arm | not-ok calls | kernels reached | kernels with ZERO ok call |
|---|---|---|---|
| qwen30b / Fortran, no skills | 150/358 = **41.9%** | 26/40 | **14** |
| qwen30b / Fortran, skills | 162/407 = **39.8%** | 28/40 | **11** |
| qwen30b / C | 157/1355 = 11.6% | 38/40 | **0** |

Not an evenly-raised error rate: a set of kernels fails 100% of the time in BOTH Fortran arms
while passing reliably in C. **10 of the 11 dead-in-both kernels touch a 2D array.**

The spec hands the agent two artifacts with OPPOSITE conventions and no mapping:

    reference_numpy:  aa[j, i] = aa[j-1, i] + bb[j, i]       # row-major
    signature:        real(c_double) :: aa(LEN_2D, LEN_2D)   # column-major

Same `bind(C)` buffer, so correct Fortran must write `aa(i+1, j+1)` -- indices REVERSED. The
emitter carries that signal ENTIRELY in extent order (`A(NK, NI)`), which for a SQUARE array
carries no information at all. Agents transliterate `aa(j, i)`, compute the TRANSPOSE, build
clean, run at full speed, and return wrong numbers forever. The judge says only
`aa: numeric mismatch`, so the repair loop has nothing to converge on and burns every round.

The skills packet already stated the rule verbatim ("subscripts reversed, starting at 1"),
verified present in the 605697 prompt and absent from 605696. **Both arms failed the same 11
kernels, 100% of attempts.** More skill text is not the fix.

Survivorship makes it worse: `submissions` holds only correctness-passing rows, so a dead kernel
leaves the DENOMINATOR rather than scoring zero.

| arm | median over SUBMITTED kernels | median counting the dead as 1.0x |
|---|---|---|
| Fortran no skills | 1.099 | **1.000** |
| Fortran skills | 1.496 | **1.000** |

### What was done

**(a) The skill page now treats layout as a correctness gate** (`edfeb355`). New section
"Translating the numpy reference -- three conversions, all silent if missed": subscripts reverse,
indices start at 1, `do` bounds are inclusive -- with a worked numpy -> Fortran mapping of a 2D
loop nest showing the CORRECT translation beside the WRONG transliteration, and the diagnostic
rule "a 2D kernel that builds clean and scores `numeric mismatch` is a transposed subscript until
proven otherwise". The old text stays, demoted to a pointer from the performance section.
Verified reaching the shipped packet (6/6 probes, 21,797 bytes) and carrying no kernel names.

**(b) Baselines, the mechanical fix.** `harness/agent.py:emit_reference_source(kernel, language)`
already runs NumpyToX and returns a correct per-language reference -- `StubAgent` submits exactly
this, so it is already the CI baseline. `materialize_shared.sh` does not ship it. The
`<kernel>_reference.c` that IS shipped is the vendored TSVC original, labelled "Not the scoring
oracle", and TSVC is a C suite -- so the Fortran track ships no baseline in its own language at
all. Shipping the emitted reference is mechanical, needs no agent behaviour change, and shows the
index order directly instead of asking the agent to remember a rule.

Two open items on it:
- **Name collision.** The shipped `<kernel>_reference.c` is the vendored original; the generated
  one is the `bind(C)` ABI form. They cannot share a name -- proposed `<module>_baseline.<ext>`.
- **Comparability.** Shipping a C baseline changes the C task and breaks comparability with prior
  C campaigns. Fortran-only shipping avoids that but makes the tracks asymmetric.

**(c) The gate has NOT yet passed.** Job 606482 was supposed to answer "is every emitted baseline
correct, C and Fortran". It reported 0/40 on both legs and that result is VOID: it ran at
`--repeat 1` while the default timing backend `mannwhitney_delta` requires >= 20 reps, so 38/40
and 37/40 kernels aborted in the TIMING stage before the code was ever delivered
(`delivered_language` empty, `hidden_total` 0, `max_rel_error` inf). Relaunched as **606563** at
`--repeat 20 --preset s` -- preset `s` because correctness of a transliteration is
size-independent and it dodges the `compare_arrays` OOM above.

**(d) Also recommended, not yet done:** make the judge's mismatch detail name the suspicion.
Comparing against the TRANSPOSED reference on mismatch and saying so converts an unrecoverable
loop into a one-round fix.

## 5. Also landed this session

`--repeat` in `regen_problems.sh` was agent multiplicity, not sampling: `make_problems.py` emits
the record N times with only `id` changed, so `--repeat 3` put THREE agents on one identical task
and bought no extra size or config coverage -- the judge draws the fuzzed size and config itself,
per grade. Now `--repeat 1`: 40 kernels, 40 agents. Verified 40 records / 40 distinct kernels in
all four focus40 files, skills and no-skills paired in identical kernel order, packet a
byte-identical cacheable prefix (14,481 B for C, 21,735 B for Fortran), and no focus40 kernel name
anywhere in either packet.
