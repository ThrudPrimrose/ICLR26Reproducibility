# Do skills cause more incorrect submissions? No -- more incorrect SCORES

Collected 2026-08-24 from 605458/605459 (qwen30b, C) and 605460/605461 (oss120b, C).
Fortran arms excluded: they are dominated by the ABI defect in `FORTRAN_ABI_DEFECT.md`
and cannot speak to this question until it is fixed.

## The premise, and why it looked true

Counting `incorrect` over ALL judged calls, skills look clearly worse:

| arm | leg | calls | incorrect | rate |
|---|---|---|---|---|
| qwen30b / C | no skills | 1355 | 109 | 8.0% |
| qwen30b / C | skills | 1318 | 149 | **11.3%** |
| oss120b / C | no skills | 737 | 35 | 4.7% |
| oss120b / C | skills | 505 | 39 | **7.7%** |

That table conflates two different events. `calls` mixes the `score` route -- the FREE
iteration loop, which records nothing -- with the `submit` route, the only recorded grade.

## Split by route, the effect reverses

| arm | leg | score calls | score incorrect | submit calls | submit incorrect |
|---|---|---|---|---|---|
| qwen30b / C | no skills | 929 | 101 (10.9%) | 426 | 8 (1.9%) |
| qwen30b / C | skills | 911 | 137 (**15.0%**) | 407 | 12 (2.9%) |
| oss120b / C | no skills | 545 | 34 (6.2%) | 192 | 1 (0.5%) |
| oss120b / C | skills | 350 | 39 (**11.1%**) | 155 | **0 (0.0%)** |

Two-sided Fisher exact:

| comparison | qwen30b | oss120b |
|---|---|---|
| incorrect **submissions** | p = 0.369 | p = 1.000 |
| incorrect **scores** | **p = 0.008** | **p = 0.012** |

**Incorrect submissions do not rise.** qwen goes 8 -> 12 of ~410 (not significant); oss goes
1 -> 0. The submission gate holds: agents submit what already scored correct.

**Incorrect scores rise significantly in both models.** The cost of the packet is paid in
wasted iteration, not in bad recorded results.

## What the agents actually write

Construct adoption, measured from the Write/Edit tool inputs in the transcripts (once per
agent: did its code ever contain the construct).

| construct | qwen30b no-skills | qwen30b skills | oss120b no-skills | oss120b skills |
|---|---|---|---|---|
| `omp parallel` | 87% | 93% | 82% | 94% |
| `omp simd` | 59% | **95%** | 53% | 46% |
| `reduction(...)` | 19% | 26% | 18% | 24% |
| `collapse(...)` | 18% | **35%** | 0% | 5% |
| `restrict` | 100% | 100% | 76% | 67% |
| `ivdep` / `__builtin_assume` | 16% | **1%** | 18% | **2%** |
| `#pragma unroll` | 3% | 0% | 2% | 1% |

The packet moves agents toward `omp simd` and `collapse` -- both UNCHECKED assertions of
independence -- and away from `ivdep`/`assume`.

## What is NOT established

Construct choice does not cleanly predict incorrectness WITHIN an arm, and the direction
flips between models:

| within-arm | qwen30b skills | oss120b skills |
|---|---|---|
| `omp simd` used vs not | 12.0% vs 0.0% incorrect | 4.5% vs 10.7% |
| `reduction` used vs not | 5.5% vs 13.4% | 13.1% vs 6.0% |

So "aggressive constructs cause wrong answers" is plausible but unproven here: construct
choice is confounded with kernel difficulty. Do not report it as a mechanism.

## Where the extra failed scores land

Spread, not concentrated -- top-5 risers hold +27 of qwen's +40, and several kernels improve
by 7-15 pp. But three kernels rise in BOTH models: `tsvc_2_s323`, `wf_north_west`,
`tsvc_2_s1244`. `ext_war_unit` goes 0/15 -> 7/37 on oss120b -- a write-after-read kernel,
i.e. a FALSE dependence, which is exactly the case where an unchecked parallel assertion is
both tempting and wrong.

## Second cost: the packet buys fewer rounds

| arm | calls / kernel (median) | total calls |
|---|---|---|
| qwen30b no skills -> skills | 31 -> 30 | 1355 -> 1318 |
| oss120b no skills -> skills | **15 -> 9** | 737 -> **505** |

oss120b loses 40% of its iterations per kernel to packet rent, then spends a higher share of
what remains on scores that come back incorrect. qwen's round count is flat, so rent alone
does not explain its rise.

## What to change -- without discouraging exploration

A failed `score` is not a defect. It is the agent trying something, on the route that is free
and records nothing, and the data says the submission gate holds behind it (incorrect
submissions do not rise: p = 0.369 / 1.000). Nothing here argues for making agents more
timid; the packet's cost is that exploration got LESS productive, not that it happened.

So the levers are the ones that make a try cheaper or more informative, not fewer:

1. **Say what the construct asserts, not "avoid it".** `omp simd`, `collapse` and
   `do concurrent` are claims the compiler never verifies -- a wrong one returns wrong
   numbers with no diagnostic. An agent that knows that can try it AND know what to check
   when the score comes back wrong. Adoption is already high; the missing piece is the
   failure signature, not permission.
2. **Name the false-dependence case explicitly.** WAR and WAO are not real dependences:
   privatise or rename and the loop is parallel. Only RAW forces sequential. `ext_war_unit`
   going 0/15 -> 7/37 on oss120b is this case, and it is a case where the agent should try
   MORE, not less.
3. **`reduction` on floats reassociates.** With rtol 1e-9 / atol 1e-11 a logically correct
   parallel reduction can still miss. Worth knowing as a diagnosis ("small delta -> suspect
   association before logic"), not as a prohibition.
4. **Give rounds back.** oss120b lost 40% of its iterations per kernel to packet rent
   (15 -> 9). That is the change with the clearest mechanism behind it: every round returned
   is another experiment the agent gets to run. See `SCORE_SUBMIT_DESIGN.md`.

Do not expect (1)-(3) to move speedup on their own: the packet's correct content already
failed to (pooled p = 1.000, `SKILLS_ABLATION_LLR6V10.md`), and the Fortran layout rule was
delivered verbatim and ignored.

## Reproduce

```bash
python3.11 submitted_source_stats.py     # construct adoption per arm
python3.11 construct_vs_correctness.py   # within-arm construct vs incorrect rate
```
