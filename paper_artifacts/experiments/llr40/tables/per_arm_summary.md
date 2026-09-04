# llr40 speed-up by arm

Snapshot: **2026-09-04T08:56:38Z**. The campaign was UNFINISHED when this was extracted, so every
count below is a snapshot of a live tree, not a finished campaign.

**How to read these numbers.**

- Every aggregate is a GEOMETRIC mean over ONE value per kernel (the best that group verified).
  The median beside it is a spread cue, never the headline.
- Non-positive speed-ups are dropped, not clamped. None occurred here: all 780 submissions carry a
  speed-up of 1.0x or more.
- `suspect` is 0 on all 780 rows. That means the implausible-speed-up check never FIRED -- it does
  NOT mean these values were vetted. **A double-digit speed-up in these tables is UNVETTED.**
- **The recorded speed-up is QUANTIZED to a 1% geometric ladder.** Every one of the 780 submission
  values is exactly `1.01^k` for an integer k (max deviation 1e-13 over all 780; exponents span
  k = 0 .. 554, giving 296 distinct values). Two numbers within 1% of each other are therefore the
  same bin, and the 4 exact C-equals-Fortran ties in the paired table are bin collisions, not two
  measurements that agreed. `call` rows are NOT on this ladder, so the snap is applied where the
  judge writes a graded record. Nothing in `hpcagent_bench/` performs it; the origin is unlocated.
- **Do not recompute a speed-up from `baseline_ns / native_ns`.** Those two columns are one
  representative sample, while `speedup` is the graded aggregate: the two disagree by a median of
  2.1%, a p90 of 8.0% and a maximum of 316%. The `speedup` column is the authoritative number and
  is what every table and figure here uses.
- Grouping is by `language`, the language the ARM asked for. `delivered_language` -- what the agent
  submitted -- is populated only on `call` rows and is empty on all 805 graded rows, so it cannot
  group a speed-up table. On the 4,450 rows carrying both, the two columns never disagree.
- `tsvc_2_s2233` is on the roster and has zero submissions in either campaign: a known open harness
  issue, not a model result. It is listed as absent rather than dropped.

## Per-arm summary

| arm | model | language | skills | runs | submissions | kernels | geomean_su | median_su | min_su | max_su | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llr40v9-kimi27sglang-cpp | kimi27sglang | cpp | off | 1 | 2 | 1 | 20.592 | 20.592 | 20.592 | 20.592 | 0.000 |
| llr40v10-qwen38-c | qwen38 | c | off | 4 | 131 | 36 | 15.269 | 16.722 | 1.361 | 207.136 | 0.000 |
| llr40v9-kimi27sglang-c | kimi27sglang | c | off | 2 | 48 | 4 | 14.179 | 17.223 | 6.558 | 21.006 | 0.000 |
| llr40v9-qwen38-c | qwen38 | c | off | 2 | 25 | 4 | 12.932 | 14.681 | 6.558 | 20.798 | 0.000 |
| llr40v9-qwen38-fortran-skills | qwen38 | fortran | on | 1 | 16 | 3 | 12.857 | 11.563 | 9.016 | 20.388 | 0.000 |
| llr40v9-kimi27sglang-c-skills | kimi27sglang | c | on | 1 | 26 | 4 | 12.428 | 15.783 | 5.113 | 19.206 | 0.000 |
| llr40v9-qwen38-cpp | qwen38 | cpp | off | 1 | 1 | 1 | 12.274 | 12.274 | 12.274 | 12.274 | 0.000 |
| llr40v9-kimi27sglang-fortran-skills | kimi27sglang | fortran | on | 1 | 18 | 3 | 11.678 | 15.126 | 5.537 | 19.016 | 0.000 |
| llr40v10-qwen38-fortran | qwen38 | fortran | off | 4 | 71 | 33 | 11.629 | 13.694 | 1.000 | 247.766 | 0.000 |
| llr40v9-qwen38-c-skills | qwen38 | c | on | 1 | 5 | 2 | 10.947 | 10.954 | 10.572 | 11.335 | 0.000 |
| llr40v10-kimi27sglang-c | kimi27sglang | c | off | 4 | 132 | 33 | 10.659 | 13.291 | 1.020 | 242.884 | 0.000 |
| llr40v9-qwen38-fortran | qwen38 | fortran | off | 2 | 19 | 4 | 10.494 | 10.819 | 5.013 | 20.798 | 0.000 |
| llr40v10-kimi27sglang-fortran | kimi27sglang | fortran | off | 3 | 74 | 30 | 9.552 | 12.097 | 1.000 | 121.033 | 0.000 |
| llr40v9-kimi27sglang-fortran | kimi27sglang | fortran | off | 2 | 51 | 6 | 9.429 | 11.737 | 3.574 | 21.006 | 0.000 |
| llr40v10-oss120b-c | oss120b | c | off | 2 | 75 | 36 | 9.313 | 9.151 | 1.000 | 124.700 | 0.000 |
| llr40v9-oss120b-c-skills | oss120b | c | on | 1 | 3 | 3 | 8.550 | 11.335 | 3.610 | 15.278 | 0.000 |
| llr40v10-oss120b-fortran | oss120b | fortran | off | 2 | 68 | 34 | 8.268 | 9.111 | 1.000 | 115.158 | 0.000 |
| llr40v9-oss120b-c | oss120b | c | off | 1 | 4 | 4 | 7.136 | 12.690 | 1.149 | 20.388 | 0.000 |
| llr40v9-oss120b-cpp | oss120b | cpp | off | 1 | 3 | 3 | 6.198 | 5.013 | 3.610 | 13.159 | 0.000 |
| llr40v9-oss120b-fortran-skills | oss120b | fortran | on | 1 | 4 | 4 | 6.177 | 6.635 | 3.610 | 11.223 | 0.000 |
| llr40v9-oss120b-fortran | oss120b | fortran | off | 1 | 4 | 4 | 4.853 | 6.816 | 1.000 | 15.430 | 0.000 |

