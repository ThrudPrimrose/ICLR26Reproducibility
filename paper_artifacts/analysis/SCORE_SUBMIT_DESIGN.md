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

It is not a per-allocation limit. Measured on an idle compute node: 22820^2 allocates fine,
`MemAvailable` 501 GB, no cgroup cap on the step, `ulimit -v` unlimited.

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

NOT YET CONFIRMED: peak RSS on a judge node during a submit burst. That is the decisive
measurement and it has not been taken.

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

## Changes, in order of expected effect

1. **Cut XL for `wf_diff_skew` and `wf_triangular` to 12000**, matching the fix already
   applied to `wf_north_west`. Recovers ~85 lost grades per campaign; one-line change each.
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

## What NOT to do

- **No auto-submit.** It solves a problem this tag does not have (0-4 agents lost a result to
  non-submission) and would run the hidden seed on every correct score, multiplying exactly
  the judge memory pressure that is already the top defect.
- **No discouraging exploration.** A failed score costs rounds, not correctness. The
  submission gate demonstrably holds.
