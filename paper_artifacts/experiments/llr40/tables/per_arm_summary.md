# llr40 speed-up by arm

Snapshot: **2026-09-08T18:42:47Z**. The campaign was UNFINISHED when this was extracted, so every
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
| gpuv2-llr40-qwen38-hip | llr40-qwen38 | hip | off | 3 | 86 | 27 | 88.060 | 64.023 | 8.751 | 2725.789 | 0.000 |
| gpuv2-llr40-qwen38-hip-skills | llr40-qwen38 | hip | on | 3 | 77 | 30 | 72.791 | 66.636 | 6.116 | 3228.163 | 0.000 |
| gpuv2-llr40-oss120b-hip-skills | llr40-oss120b | hip | on | 1 | 35 | 35 | 31.633 | 39.711 | 1.000 | 2698.801 | 0.000 |
| gpuv2-llr40-oss120b-hip | llr40-oss120b | hip | off | 1 | 36 | 35 | 30.433 | 32.545 | 1.000 | 2922.414 | 0.000 |
| llr40v9-kimi27sglang-cpp | kimi27sglang | cpp | off | 1 | 2 | 1 | 20.592 | 20.592 | 20.592 | 20.592 | 0.000 |
| llr40v10-qwen38-c | qwen38 | c | off | 4 | 147 | 36 | 15.863 | 16.722 | 1.661 | 207.136 | 0.000 |
| gpuv2-llr40-qwen38-omp | llr40-qwen38 | c | off | 3 | 24 | 16 | 15.829 | 16.308 | 1.030 | 527.795 | 0.000 |
| llr40v9-kimi27sglang-c | kimi27sglang | c | off | 2 | 48 | 4 | 14.179 | 17.223 | 6.558 | 21.006 | 0.000 |
| llr40v9-qwen38-c | qwen38 | c | off | 2 | 25 | 4 | 12.932 | 14.681 | 6.558 | 20.798 | 0.000 |
| llr40v9-qwen38-fortran-skills | qwen38 | fortran | on | 1 | 16 | 3 | 12.857 | 11.563 | 9.016 | 20.388 | 0.000 |
| llr40v10-kimi27sglang-fortran | kimi27sglang | fortran | off | 4 | 153 | 35 | 12.783 | 15.278 | 1.116 | 121.033 | 0.000 |
| llr40v9-kimi27sglang-c-skills | kimi27sglang | c | on | 1 | 26 | 4 | 12.428 | 15.783 | 5.113 | 19.206 | 0.000 |
| llr40v9-qwen38-cpp | qwen38 | cpp | off | 1 | 1 | 1 | 12.274 | 12.274 | 12.274 | 12.274 | 0.000 |
| llr40v11-qwen38-c | qwen38 | c | off | 1 | 33 | 24 | 12.203 | 11.546 | 1.196 | 247.766 | 0.000 |
| llr40v9-kimi27sglang-fortran-skills | kimi27sglang | fortran | on | 1 | 18 | 3 | 11.678 | 15.126 | 5.537 | 19.016 | 0.000 |
| llr40v10-kimi27sglang-c | kimi27sglang | c | off | 4 | 163 | 35 | 11.017 | 13.969 | 1.000 | 242.884 | 0.000 |
| gpuv2-llr40-qwen38-omp-skills | llr40-qwen38 | c | on | 2 | 25 | 17 | 10.989 | 9.571 | 1.000 | 711.388 | 0.000 |
| llr40v9-qwen38-c-skills | qwen38 | c | on | 1 | 5 | 2 | 10.947 | 10.954 | 10.572 | 11.335 | 0.000 |
| llr40v11-qwen38-fortran-skills | qwen38 | fortran | on | 1 | 29 | 23 | 10.878 | 10.468 | 1.000 | 250.243 | 0.000 |
| gpuv4-llr40-qwen38-pytriton-skills | llr40-qwen38 | python | on | 3 | 99 | 30 | 10.558 | 10.680 | 1.000 | 222.078 | 0.000 |
| llr40v10-qwen38-c-skills | qwen38 | c | on | 2 | 49 | 28 | 10.505 | 10.901 | 1.000 | 242.884 | 0.000 |
| llr40v9-qwen38-fortran | qwen38 | fortran | off | 2 | 19 | 4 | 10.494 | 10.819 | 5.013 | 20.798 | 0.000 |
| llr40v10-qwen38-fortran | qwen38 | fortran | off | 4 | 87 | 35 | 10.411 | 13.424 | 1.000 | 247.766 | 0.000 |
| llr40v11-qwen38-c-skills | qwen38 | c | on | 1 | 44 | 26 | 9.648 | 8.913 | 1.020 | 159.919 | 0.000 |
| llr40v11-kimi27sglang-fortran-skills | kimi27sglang | fortran | on | 2 | 60 | 26 | 9.447 | 6.765 | 1.196 | 245.313 | 0.000 |
| llr40v9-kimi27sglang-fortran | kimi27sglang | fortran | off | 2 | 51 | 6 | 9.429 | 11.737 | 3.574 | 21.006 | 0.000 |
| llr40v10-oss120b-c | oss120b | c | off | 2 | 75 | 36 | 9.313 | 9.151 | 1.000 | 124.700 | 0.000 |
| llr40v11-qwen38-fortran | qwen38 | fortran | off | 1 | 19 | 17 | 9.000 | 8.244 | 1.245 | 250.243 | 0.000 |
| gpuv4-llr40-qwen38-pytriton | llr40-qwen38 | python | off | 4 | 94 | 31 | 8.927 | 8.162 | 1.000 | 247.766 | 0.000 |
| llr40v9-oss120b-c-skills | oss120b | c | on | 1 | 3 | 3 | 8.550 | 11.335 | 3.610 | 15.278 | 0.000 |
| llr40v10-kimi27sglang-fortran-skills | kimi27sglang | fortran | on | 2 | 74 | 27 | 8.453 | 8.579 | 1.000 | 110.665 | 0.000 |
| llr40v11-kimi27sglang-c-skills | kimi27sglang | c | on | 2 | 43 | 25 | 8.363 | 7.031 | 1.020 | 209.208 | 0.000 |
| llr40v10-oss120b-fortran | oss120b | fortran | off | 2 | 68 | 34 | 8.268 | 9.111 | 1.000 | 115.158 | 0.000 |
| llr40v10-kimi27sglang-c-skills | kimi27sglang | c | on | 2 | 129 | 32 | 8.239 | 8.088 | 1.010 | 215.547 | 0.000 |
| v11w2-oss120b-c-skills | oss120b | c | on | 2 | 4 | 4 | 7.982 | 11.228 | 1.474 | 22.077 | 0.000 |
| llr40v11-kimi27sglang-c | kimi27sglang | c | off | 2 | 49 | 27 | 7.709 | 5.428 | 1.000 | 231.095 | 0.000 |
| llr40v10-qwen38-fortran-skills | qwen38 | fortran | on | 1 | 33 | 28 | 7.578 | 9.315 | 1.000 | 209.208 | 0.000 |
| v11w2-qwen38-fortran | qwen38 | fortran | off | 3 | 36 | 23 | 7.253 | 8.244 | 1.000 | 79.690 | 0.000 |
| llr40v9-oss120b-c | oss120b | c | off | 1 | 4 | 4 | 7.136 | 12.690 | 1.149 | 20.388 | 0.000 |
| v11w2-kimi27sglang-fortran | kimi27sglang | fortran | off | 3 | 49 | 14 | 6.670 | 9.981 | 1.000 | 50.422 | 0.000 |
| gpuv2-llr40-oss120b-omp-skills | llr40-oss120b | c | on | 1 | 23 | 22 | 6.362 | 6.790 | 1.105 | 463.754 | 0.000 |
| llr40v9-oss120b-cpp | oss120b | cpp | off | 1 | 3 | 3 | 6.198 | 5.013 | 3.610 | 13.159 | 0.000 |
| llr40v9-oss120b-fortran-skills | oss120b | fortran | on | 1 | 4 | 4 | 6.177 | 6.635 | 3.610 | 11.223 | 0.000 |
| llr40v11-oss120b-fortran | oss120b | fortran | off | 1 | 29 | 26 | 6.058 | 4.662 | 1.010 | 81.292 | 0.000 |
| llr40v10-oss120b-c-skills | oss120b | c | on | 2 | 50 | 33 | 6.041 | 4.404 | 1.000 | 219.879 | 0.000 |
| v11w2-qwen38-c-skills | qwen38 | c | on | 3 | 31 | 14 | 5.983 | 5.888 | 1.000 | 80.487 | 0.000 |
| v11w2-qwen38-fortran-skills | qwen38 | fortran | on | 3 | 33 | 17 | 5.909 | 5.705 | 1.000 | 64.663 | 0.000 |
| llr40v11-kimi27sglang-fortran | kimi27sglang | fortran | off | 2 | 38 | 26 | 5.799 | 4.254 | 1.000 | 93.443 | 0.000 |
| v11w2-oss120b-c | oss120b | c | off | 3 | 15 | 8 | 5.748 | 9.443 | 1.000 | 39.711 | 0.000 |
| llr40v11-oss120b-fortran-skills | oss120b | fortran | on | 1 | 27 | 26 | 5.659 | 3.683 | 1.000 | 152.158 | 0.000 |
| llr40v11-oss120b-c | oss120b | c | off | 1 | 39 | 32 | 5.566 | 4.211 | 1.000 | 235.741 | 0.000 |
| v11w2-kimi27sglang-c-skills | kimi27sglang | c | on | 3 | 48 | 15 | 5.158 | 4.817 | 1.000 | 64.663 | 0.000 |
| llr40v9-oss120b-fortran | oss120b | fortran | off | 1 | 4 | 4 | 4.853 | 6.816 | 1.000 | 15.430 | 0.000 |
| v11w2-kimi27sglang-fortran-skills | kimi27sglang | fortran | on | 3 | 43 | 14 | 4.848 | 4.185 | 1.000 | 88.027 | 0.000 |
| llr40v10-oss120b-fortran-skills | oss120b | fortran | on | 2 | 38 | 31 | 4.693 | 3.538 | 1.000 | 209.208 | 0.000 |
| llr40v11-oss120b-c-skills | oss120b | c | on | 1 | 36 | 35 | 4.490 | 3.948 | 1.000 | 58.539 | 0.000 |
| v11w2-qwen38-c | qwen38 | c | off | 3 | 37 | 16 | 4.468 | 4.077 | 1.000 | 70.021 | 0.000 |
| v11w2-oss120b-fortran-skills | oss120b | fortran | on | 4 | 30 | 13 | 4.371 | 4.538 | 1.000 | 43.865 | 0.000 |
| v11w2-kimi27sglang-c | kimi27sglang | c | off | 3 | 25 | 13 | 4.027 | 2.424 | 1.000 | 49.923 | 0.000 |
| gpuv2-llr40-oss120b-omp | llr40-oss120b | c | off | 1 | 26 | 25 | 3.709 | 3.503 | 1.000 | 255.273 | 0.000 |
| v11w2-oss120b-fortran | oss120b | fortran | off | 3 | 20 | 14 | 2.979 | 1.418 | 1.000 | 17.045 | 0.000 |
| gpuv4-llr40-oss120b-pytriton-skills | llr40-oss120b | python | on | 1 | 38 | 32 | 2.528 | 1.558 | 1.000 | 147.683 | 0.000 |
| gpuv4-llr40-oss120b-pytriton | llr40-oss120b | python | off | 1 | 38 | 35 | 1.943 | 1.596 | 1.000 | 20.592 | 0.000 |

