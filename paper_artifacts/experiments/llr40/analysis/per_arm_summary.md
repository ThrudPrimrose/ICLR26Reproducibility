# llr40 speed-up by arm and denominator

Snapshot: **2026-09-11T23:56:36Z**. The campaign was UNFINISHED when this was extracted, so every
count below is a snapshot of a live tree, not a finished campaign.

**How to read these numbers.**

- **Every table is keyed on `(arm, baseline)`.** `baseline` is the reference the judge divided by
  and it is a per-JOB property: some jobs graded against the single-core C lowering and some against
  parallel numba. The SAME agent work reads 95.3x under one and 1.82x under the other. Rows with
  different `baseline` values are not comparable and are never pooled. `denominator_split.csv` lists
  which job graded against which.
- **Two population policies, named per column.** `*_solved` is the geomean over the kernels the arm
  VERIFIED -- "how good when it works". `*_served` scores a kernel the arm was given and never
  verified at 1.0 -- "how good overall". The served roster is the kernels the arm has a recorded
  observation for, never the full 40: a kernel it never saw is a scheduling fact, and this campaign
  was a SNAPSHOT with jobs still queued, so several arms were served far fewer than 40.
- **Do not divide two rows of `per_arm_summary`.** Each row's geomean is over that arm's own kernel
  set. `arm_pairs.csv` is the comparison table: every pair restricted to the kernels both reached,
  with `unmatched_ratio` beside `matched_ratio` and `direction_flips` marking the pairs where the
  two disagree about which arm is ahead.
- Every aggregate is a GEOMETRIC mean over ONE value per kernel. The median beside it is a spread
  cue, never the headline.
- Non-positive speed-ups are dropped, not clamped. None occurred here: all 780 submissions carry a
  speed-up of 1.0x or more.
- `suspect` is 0 on all 780 rows. That means the implausible-speed-up check never FIRED -- it does
  NOT mean these values were vetted. **A double-digit speed-up in these tables is UNVETTED.**
- **The recorded speed-up is QUANTIZED to a 1% geometric ladder.** Every one of the 780 submission
  values is exactly `1.01^k` for an integer k (max deviation 1e-13 over all 780; exponents span
  k = 0 .. 554, giving 296 distinct values). Two numbers within 1% of each other are therefore the
  same bin, and the exact C-equals-Fortran ties in the paired table are bin collisions, not two
  measurements that agreed. `call` rows are NOT on this ladder, so the snap is applied where the
  judge writes a graded record. Nothing in `hpcagent_bench/` performs it; the origin is unlocated.
- **Do not recompute a speed-up from `baseline_ns / native_ns`.** Those two columns are one
  representative sample, while `speedup` is the graded aggregate: the two disagree by a median of
  2.1%, a p90 of 8.0% and a maximum of 316%. The `speedup` column is the authoritative number and
  is what every table and figure here uses.
- Grouping is by `language`, the language the ARM asked for, which lives on `runs`.
- `tsvc_2_s2233` is on the roster and has zero submissions in either campaign: a known open harness
  issue, not a model result. It is listed as absent rather than dropped.

## Per-arm summary, keyed on (arm, baseline)

| arm | baseline | campaign | model | language | skills | jobs | submissions | n_served | n_solved | geomean_solved | geomean_solved_low | geomean_solved_high | median_solved | min_solved | max_solved | geomean_served | median_baseline_ns | median_native_ns | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpuv2-llr40-qwen38-hip | numba | gpuv2 | llr40-qwen38 | hip | 0 | 3 | 86 | 32 | 27 | 87.768 | 44.748 | 172.146 | 64.023 | 8.326 | 2725.789 | 43.621 | 159788961.000 | 1467978.000 | 0 |
| gpuv2-llr40-qwen38-hip-skills | numba | gpuv2 | llr40-qwen38 | hip | 1 | 3 | 77 | 36 | 30 | 72.478 | 39.925 | 131.575 | 66.312 | 6.116 | 3228.163 | 35.495 | 123734468.000 | 1512162.000 | 0 |
| gpuv2-llr40-oss120b-hip | numba | gpuv2 | llr40-oss120b | hip | 0 | 1 | 36 | 39 | 35 | 30.433 | 16.492 | 56.158 | 32.545 | 1.000 | 2922.414 | 21.439 | 172861850.000 | 3200176.000 | 0 |
| llr40v9-kimi27sglang-cpp | c | llr40v9 | kimi27sglang | cpp | 0 | 1 | 2 | 1 | 1 | 20.592 | 20.592 | 20.592 | 20.592 | 20.592 | 20.592 | 20.592 | 363091737.000 | 17158200.000 | 0 |
| gpuv2-llr40-oss120b-hip-skills | numba | gpuv2 | llr40-oss120b | hip | 1 | 1 | 35 | 40 | 35 | 31.633 | 16.358 | 61.171 | 39.711 | 1.000 | 2698.801 | 20.541 | 179393013.000 | 3203948.000 | 0 |
| llr40v9-kimi27sglang-c | c | llr40v9 | kimi27sglang | c | 0 | 2 | 48 | 4 | 4 | 14.179 | 6.101 | 32.950 | 17.223 | 6.558 | 21.006 | 14.179 | 307241171.500 | 20215978.500 | 0 |
| llr40v9-qwen38-c | c | llr40v9 | qwen38 | c | 0 | 2 | 25 | 4 | 4 | 12.932 | 5.663 | 29.533 | 14.681 | 6.558 | 20.798 | 12.932 | 306700838.500 | 20783515.500 | 0 |
| llr40v9-qwen38-fortran-skills | c | llr40v9 | qwen38 | fortran | 1 | 1 | 16 | 3 | 3 | 12.857 | 4.550 | 36.334 | 11.563 | 9.016 | 20.388 | 12.857 | 251007348.000 | 17471876.000 | 0 |
| llr40v9-qwen38-cpp | c | llr40v9 | qwen38 | cpp | 0 | 1 | 1 | 1 | 1 | 12.274 | 12.274 | 12.274 | 12.274 | 12.274 | 12.274 | 12.274 | 363380689.000 | 29039267.000 | 0 |
| llr40v9-kimi27sglang-c-skills | c | llr40v9 | kimi27sglang | c | 1 | 1 | 26 | 4 | 4 | 12.213 | 4.648 | 32.092 | 15.717 | 5.063 | 18.274 | 12.213 | 327531865.000 | 25726586.000 | 0 |
| llr40v10-kimi27sglang-c | c | llr40v10 | kimi27sglang | c | 0 | 1 | 54 | 20 | 18 | 15.177 | 8.017 | 28.729 | 11.170 | 1.746 | 242.884 | 11.563 | 217315048.500 | 19640968.500 | 0 |
| llr40v9-kimi27sglang-fortran-skills | c | llr40v9 | kimi27sglang | fortran | 1 | 1 | 18 | 3 | 3 | 11.038 | 2.327 | 52.355 | 12.900 | 5.537 | 18.828 | 11.038 | 292433706.000 | 19169135.000 | 0 |
| llr40v9-qwen38-fortran | c | llr40v9 | qwen38 | fortran | 0 | 2 | 19 | 4 | 4 | 10.468 | 4.128 | 26.544 | 10.761 | 5.013 | 20.798 | 10.468 | 307566388.000 | 25881765.500 | 0 |
| llr40v9-kimi27sglang-fortran | c | llr40v9 | kimi27sglang | fortran | 0 | 2 | 51 | 6 | 6 | 9.274 | 4.738 | 18.154 | 11.509 | 3.574 | 21.006 | 9.274 | 307260937.000 | 24154425.500 | 0 |
| llr40v10-oss120b-c | c | llr40v10 | oss120b | c | 0 | 2 | 75 | 40 | 36 | 9.313 | 6.058 | 14.317 | 9.151 | 1.000 | 124.700 | 7.450 | 216692261.000 | 25975880.500 | 0 |
| v11w2-qwen38-fortran | numba | v11w2 | qwen38 | fortran | 0 | 4 | 36 | 23 | 23 | 7.250 | 3.946 | 13.320 | 8.244 | 1.000 | 79.690 | 7.250 | 156814879.000 | 18602385.000 | 0 |
| llr40v10-qwen38-c-skills | numba | llr40v10 | qwen38 | c | 1 | 2 | 49 | 32 | 28 | 9.534 | 5.189 | 17.515 | 9.523 | 1.000 | 242.884 | 7.192 | 175249398.000 | 18029217.000 | 0 |
| llr40v10-qwen38-c | c | llr40v10 | qwen38 | c | 0 | 2 | 74 | 40 | 29 | 14.783 | 9.357 | 23.353 | 17.387 | 1.000 | 164.765 | 7.048 | 315959981.000 | 17541486.000 | 0 |
| v11w2-kimi27sglang-fortran | numba | v11w2 | kimi27sglang | fortran | 0 | 3 | 49 | 14 | 14 | 6.628 | 3.078 | 14.272 | 9.981 | 1.000 | 49.923 | 6.628 | 269969958.500 | 25677508.500 | 0 |
| gpuv4-llr40-qwen38-pytriton-skills | numba | gpuv4 | llr40-qwen38 | pytriton | 1 | 3 | 99 | 38 | 30 | 10.544 | 6.008 | 18.504 | 10.680 | 1.000 | 222.078 | 6.422 | 184145107.000 | 18509216.500 | 0 |
| llr40v10-oss120b-fortran | c | llr40v10 | oss120b | fortran | 0 | 2 | 68 | 40 | 34 | 8.239 | 5.267 | 12.887 | 9.111 | 1.000 | 115.158 | 6.005 | 183139164.500 | 26164795.500 | 0 |
| v11w2-qwen38-c-skills | numba | v11w2 | qwen38 | c | 1 | 3 | 31 | 14 | 14 | 5.949 | 2.520 | 14.046 | 5.888 | 1.000 | 80.487 | 5.949 | 171181929.000 | 26002393.000 | 0 |
| v11w2-qwen38-fortran-skills | numba | v11w2 | qwen38 | fortran | 1 | 3 | 33 | 17 | 17 | 5.898 | 3.024 | 11.503 | 5.705 | 1.000 | 64.663 | 5.898 | 140066260.000 | 18426676.000 | 0 |
| v11w2-oss120b-c | numba | v11w2 | oss120b | c | 0 | 5 | 15 | 8 | 8 | 5.726 | 1.548 | 21.175 | 9.371 | 1.000 | 39.711 | 5.726 | 242635713.500 | 42235280.000 | 0 |
| llr40v10-qwen38-fortran | c | llr40v10 | qwen38 | fortran | 0 | 2 | 45 | 36 | 28 | 8.956 | 5.122 | 15.658 | 11.827 | 1.000 | 169.757 | 5.502 | 277271700.500 | 21439359.500 | 0 |
| gpuv4-llr40-qwen38-pytriton | numba | gpuv4 | llr40-qwen38 | pytriton | 0 | 4 | 94 | 40 | 31 | 8.824 | 5.112 | 15.234 | 8.162 | 1.000 | 247.766 | 5.406 | 184071723.000 | 18790607.000 | 0 |
| llr40v10-kimi27sglang-c-skills | numba | llr40v10 | kimi27sglang | c | 1 | 2 | 129 | 40 | 32 | 8.006 | 4.981 | 12.869 | 8.005 | 1.010 | 163.134 | 5.281 | 165612180.500 | 17461751.500 | 0 |
| v11w2-oss120b-c-skills | numba | v11w2 | oss120b | c | 1 | 3 | 4 | 5 | 4 | 7.982 | 1.232 | 51.696 | 11.228 | 1.474 | 22.077 | 5.268 | 485479093.000 | 34745171.000 | 0 |
| llr40v10-kimi27sglang-fortran | c | llr40v10 | kimi27sglang | fortran | 0 | 2 | 62 | 36 | 25 | 10.815 | 6.643 | 17.606 | 13.694 | 1.000 | 121.033 | 5.225 | 310904839.000 | 18561897.000 | 0 |
| llr40v10-kimi27sglang-c | numba | llr40v10 | kimi27sglang | c | 0 | 3 | 109 | 40 | 31 | 8.289 | 5.144 | 13.356 | 10.364 | 1.000 | 219.879 | 5.150 | 175732877.000 | 17474308.000 | 0 |
| v11w2-kimi27sglang-c-skills | numba | v11w2 | kimi27sglang | c | 1 | 3 | 48 | 15 | 15 | 5.147 | 2.483 | 10.673 | 4.817 | 1.000 | 63.389 | 5.147 | 191572556.000 | 28582527.000 | 0 |
| llr40v9-qwen38-c-skills | c | llr40v9 | qwen38 | c | 1 | 1 | 5 | 3 | 2 | 10.947 | 7.033 | 17.040 | 10.954 | 10.572 | 11.335 | 4.930 | 307377702.500 | 27316383.500 | 0 |
| llr40v11-qwen38-c | numba | llr40v11 | qwen38 | c | 0 | 1 | 33 | 38 | 24 | 12.203 | 7.375 | 20.192 | 11.546 | 1.196 | 247.766 | 4.855 | 214401355.500 | 17663119.500 | 0 |
| v11w2-kimi27sglang-fortran-skills | numba | v11w2 | kimi27sglang | fortran | 1 | 3 | 43 | 14 | 14 | 4.838 | 1.918 | 12.202 | 4.185 | 1.000 | 88.027 | 4.838 | 100054598.500 | 22333027.500 | 0 |
| llr40v10-qwen38-c | numba | llr40v10 | qwen38 | c | 0 | 2 | 73 | 40 | 31 | 7.511 | 4.460 | 12.648 | 7.538 | 1.000 | 207.136 | 4.772 | 183913746.000 | 18406972.000 | 0 |
| llr40v10-kimi27sglang-fortran | numba | llr40v10 | kimi27sglang | fortran | 0 | 2 | 91 | 40 | 31 | 7.458 | 4.778 | 11.643 | 9.289 | 1.116 | 101.185 | 4.746 | 183484067.000 | 17580048.000 | 0 |
| llr40v11-qwen38-c-skills | numba | llr40v11 | qwen38 | c | 1 | 1 | 44 | 38 | 26 | 9.644 | 5.752 | 16.171 | 8.913 | 1.020 | 159.919 | 4.715 | 179212968.000 | 17332159.500 | 0 |
| v11w2-qwen38-c | numba | v11w2 | qwen38 | c | 0 | 3 | 37 | 16 | 16 | 4.465 | 2.182 | 9.135 | 4.056 | 1.000 | 70.021 | 4.465 | 76819311.500 | 18639115.000 | 0 |
| llr40v10-qwen38-fortran-skills | numba | llr40v10 | qwen38 | fortran | 1 | 1 | 33 | 38 | 28 | 7.573 | 4.465 | 12.844 | 9.315 | 1.000 | 205.086 | 4.445 | 183876373.000 | 20000700.500 | 0 |
| llr40v11-qwen38-fortran-skills | numba | llr40v11 | qwen38 | fortran | 1 | 1 | 29 | 37 | 23 | 10.878 | 5.833 | 20.288 | 10.468 | 1.000 | 250.243 | 4.409 | 184196860.000 | 18540855.000 | 0 |
| llr40v10-oss120b-c-skills | numba | llr40v10 | oss120b | c | 1 | 2 | 50 | 40 | 33 | 6.012 | 3.463 | 10.438 | 4.404 | 1.000 | 219.879 | 4.392 | 136935309.000 | 21753773.000 | 0 |
| llr40v10-kimi27sglang-fortran-skills | numba | llr40v10 | kimi27sglang | fortran | 1 | 2 | 74 | 39 | 27 | 8.416 | 5.292 | 13.383 | 8.494 | 1.000 | 110.665 | 4.370 | 183870104.000 | 19204998.000 | 0 |
| llr40v11-kimi27sglang-fortran-skills | numba | llr40v11 | kimi27sglang | fortran | 1 | 2 | 60 | 40 | 26 | 9.444 | 5.630 | 15.840 | 6.765 | 1.196 | 245.313 | 4.304 | 221456368.000 | 17811111.500 | 0 |
| v11w2-kimi27sglang-c | numba | v11w2 | kimi27sglang | c | 0 | 3 | 25 | 13 | 13 | 4.009 | 1.793 | 8.963 | 2.424 | 1.000 | 47.030 | 4.009 | 91436397.000 | 20443256.000 | 0 |
| llr40v11-oss120b-c | numba | llr40v11 | oss120b | c | 0 | 1 | 39 | 39 | 32 | 5.391 | 3.184 | 9.129 | 4.211 | 1.000 | 235.741 | 3.984 | 155557321.500 | 21730788.000 | 0 |
| llr40v11-kimi27sglang-c | numba | llr40v11 | kimi27sglang | c | 0 | 2 | 49 | 40 | 27 | 7.658 | 4.320 | 13.574 | 5.428 | 1.000 | 231.095 | 3.952 | 199463960.000 | 23415250.000 | 0 |
| v11w2-oss120b-fortran-skills | numba | v11w2 | oss120b | fortran | 1 | 6 | 30 | 14 | 13 | 4.334 | 1.958 | 9.593 | 4.538 | 1.000 | 43.865 | 3.903 | 184289690.000 | 33728541.000 | 0 |
| llr40v11-kimi27sglang-c-skills | numba | llr40v11 | kimi27sglang | c | 1 | 2 | 43 | 40 | 25 | 8.356 | 4.710 | 14.825 | 7.031 | 1.020 | 209.208 | 3.769 | 170627554.000 | 17599134.000 | 0 |
| llr40v11-oss120b-c-skills | numba | llr40v11 | oss120b | c | 1 | 1 | 36 | 40 | 35 | 4.490 | 2.899 | 6.954 | 3.948 | 1.000 | 58.539 | 3.722 | 155924975.000 | 24008319.000 | 0 |
| llr40v9-oss120b-c | c | llr40v9 | oss120b | c | 0 | 1 | 4 | 6 | 4 | 7.136 | 0.822 | 61.971 | 12.690 | 1.149 | 20.388 | 3.707 | 262936613.500 | 26069174.000 | 0 |
| gpuv2-llr40-qwen38-omp | numba | gpuv2 | llr40-qwen38 | omp | 0 | 3 | 24 | 35 | 16 | 15.829 | 6.998 | 35.803 | 16.308 | 1.030 | 527.795 | 3.534 | 184019061.500 | 15112344.500 | 0 |
| llr40v9-oss120b-fortran-skills | c | llr40v9 | oss120b | fortran | 1 | 1 | 4 | 6 | 4 | 6.177 | 2.393 | 15.949 | 6.635 | 3.610 | 11.223 | 3.367 | 416788948.000 | 85313824.500 | 0 |
| llr40v10-oss120b-fortran-skills | numba | llr40v10 | oss120b | fortran | 1 | 2 | 38 | 40 | 31 | 4.693 | 2.811 | 7.837 | 3.538 | 1.000 | 209.208 | 3.314 | 171228640.000 | 30673263.000 | 0 |
| llr40v11-oss120b-fortran | numba | llr40v11 | oss120b | fortran | 0 | 1 | 29 | 40 | 26 | 6.053 | 3.548 | 10.328 | 4.662 | 1.000 | 81.292 | 3.223 | 148570924.000 | 18505458.500 | 0 |
| llr40v11-kimi27sglang-fortran | numba | llr40v11 | kimi27sglang | fortran | 0 | 2 | 38 | 40 | 26 | 5.777 | 3.276 | 10.188 | 4.233 | 1.000 | 93.443 | 3.127 | 140098030.000 | 18305390.500 | 0 |
| llr40v11-oss120b-fortran-skills | numba | llr40v11 | oss120b | fortran | 1 | 1 | 27 | 40 | 26 | 5.659 | 3.141 | 10.196 | 3.683 | 1.000 | 152.158 | 3.085 | 158143181.500 | 19348641.000 | 0 |
| llr40v11-qwen38-fortran | numba | llr40v11 | qwen38 | fortran | 0 | 1 | 19 | 34 | 17 | 9.000 | 4.146 | 19.539 | 8.244 | 1.245 | 250.243 | 3.000 | 203539644.000 | 20109996.000 | 0 |
| v11w2-oss120b-fortran | numba | v11w2 | oss120b | fortran | 0 | 5 | 20 | 14 | 14 | 2.979 | 1.437 | 6.176 | 1.418 | 1.000 | 17.045 | 2.979 | 184970727.500 | 40533992.500 | 0 |
| llr40v10-qwen38-fortran | numba | llr40v10 | qwen38 | fortran | 0 | 2 | 42 | 35 | 25 | 4.601 | 2.424 | 8.736 | 3.719 | 1.000 | 247.766 | 2.975 | 130097987.000 | 20712391.000 | 0 |
| llr40v9-oss120b-c-skills | c | llr40v9 | oss120b | c | 1 | 1 | 3 | 6 | 3 | 8.550 | 1.289 | 56.705 | 11.335 | 3.610 | 15.278 | 2.924 | 363654836.000 | 22033288.000 | 0 |
| gpuv2-llr40-qwen38-omp-skills | numba | gpuv2 | llr40-qwen38 | omp | 1 | 2 | 25 | 38 | 17 | 10.982 | 4.927 | 24.479 | 9.571 | 1.000 | 711.388 | 2.921 | 185680559.000 | 17929202.000 | 0 |
| llr40v9-oss120b-fortran | c | llr40v9 | oss120b | fortran | 0 | 1 | 4 | 6 | 4 | 4.853 | 0.698 | 33.763 | 6.816 | 1.000 | 15.430 | 2.866 | 415633715.500 | 105450841.000 | 0 |
| gpuv2-llr40-oss120b-omp-skills | numba | gpuv2 | llr40-oss120b | omp | 1 | 1 | 23 | 40 | 22 | 6.362 | 3.442 | 11.759 | 6.790 | 1.105 | 463.754 | 2.767 | 182657508.500 | 18119547.500 | 0 |
| llr40v9-oss120b-cpp | c | llr40v9 | oss120b | cpp | 0 | 1 | 3 | 6 | 3 | 6.198 | 1.166 | 32.935 | 5.013 | 3.610 | 13.159 | 2.490 | 363450546.000 | 30249272.000 | 0 |
| gpuv2-llr40-oss120b-omp | numba | gpuv2 | llr40-oss120b | omp | 0 | 1 | 26 | 40 | 25 | 3.709 | 2.154 | 6.386 | 3.503 | 1.000 | 255.273 | 2.269 | 146736004.000 | 19565672.000 | 0 |
| gpuv4-llr40-oss120b-pytriton-skills | numba | gpuv4 | llr40-oss120b | pytriton | 1 | 1 | 38 | 40 | 32 | 2.528 | 1.661 | 3.850 | 1.558 | 1.000 | 147.683 | 2.100 | 181794787.000 | 93881121.000 | 0 |
| gpuv4-llr40-oss120b-pytriton | numba | gpuv4 | llr40-oss120b | pytriton | 0 | 1 | 38 | 40 | 35 | 1.943 | 1.473 | 2.563 | 1.596 | 1.000 | 20.592 | 1.788 | 191109999.000 | 92007124.000 | 0 |

## Which job graded against which reference

| baseline | job | arm | submissions | kernels |
|---|---|---|---|---|
| c | 618217 | llr40v9-qwen38-c | 12 | 3 |
| c | 618218 | llr40v9-qwen38-cpp | 1 | 1 |
| c | 618219 | llr40v9-qwen38-fortran | 9 | 3 |
| c | 618220 | llr40v9-oss120b-c | 4 | 4 |
| c | 618221 | llr40v9-oss120b-cpp | 3 | 3 |
| c | 618222 | llr40v9-oss120b-fortran | 4 | 4 |
| c | 618223 | llr40v9-kimi27sglang-c | 16 | 4 |
| c | 618224 | llr40v9-kimi27sglang-cpp | 2 | 1 |
| c | 618225 | llr40v9-kimi27sglang-fortran | 21 | 4 |
| c | 618226 | llr40v9-qwen38-c-skills | 5 | 2 |
| c | 618228 | llr40v9-qwen38-fortran-skills | 16 | 3 |
| c | 618229 | llr40v9-oss120b-c-skills | 3 | 3 |
| c | 618231 | llr40v9-oss120b-fortran-skills | 4 | 4 |
| c | 618232 | llr40v9-kimi27sglang-c-skills | 26 | 4 |
| c | 618234 | llr40v9-kimi27sglang-fortran-skills | 18 | 3 |
| c | 619183 | llr40v9-qwen38-c | 13 | 4 |
| c | 619184 | llr40v9-qwen38-fortran | 10 | 4 |
| c | 619185 | llr40v9-kimi27sglang-c | 32 | 3 |
| c | 619186 | llr40v9-kimi27sglang-fortran | 30 | 6 |
| c | 621015 | llr40v10-oss120b-c | 37 | 33 |
| c | 621016 | llr40v10-qwen38-c | 31 | 21 |
| c | 621017 | llr40v10-kimi27sglang-c | 54 | 18 |
| c | 621019 | llr40v10-oss120b-fortran | 34 | 29 |
| c | 621020 | llr40v10-qwen38-fortran | 18 | 18 |
| c | 621021 | llr40v10-kimi27sglang-fortran | 9 | 7 |
| c | 621022 | llr40v10-kimi27sglang-fortran | 53 | 18 |
| c | 621382 | llr40v10-oss120b-c | 38 | 36 |
| c | 621383 | llr40v10-qwen38-c | 43 | 23 |
| c | 621384 | llr40v10-oss120b-fortran | 34 | 33 |
| c | 621385 | llr40v10-qwen38-fortran | 27 | 22 |
| numba | 621018 | llr40v10-kimi27sglang-c | 11 | 10 |
| numba | 621023 | llr40v10-oss120b-c-skills | 47 | 33 |
| numba | 621024 | llr40v10-qwen38-c-skills | 47 | 28 |
| numba | 621025 | llr40v10-kimi27sglang-c-skills | 67 | 18 |
| numba | 621026 | llr40v10-kimi27sglang-c-skills | 62 | 14 |
| numba | 621027 | llr40v10-oss120b-fortran-skills | 37 | 31 |
| numba | 621028 | llr40v10-qwen38-fortran-skills | 33 | 28 |
| numba | 621029 | llr40v10-kimi27sglang-fortran-skills | 34 | 12 |
| numba | 621030 | llr40v10-kimi27sglang-fortran-skills | 40 | 15 |
| numba | 621727 | llr40v10-qwen38-c | 40 | 23 |
| numba | 621728 | llr40v10-qwen38-fortran | 17 | 17 |
| numba | 621729 | llr40v10-kimi27sglang-c | 50 | 15 |
| numba | 621730 | llr40v10-kimi27sglang-c | 48 | 15 |
| numba | 621731 | llr40v10-kimi27sglang-fortran | 42 | 16 |
| numba | 621732 | llr40v10-kimi27sglang-fortran | 49 | 15 |
| numba | 622265 | llr40v10-qwen38-c | 33 | 25 |
| numba | 622266 | llr40v10-qwen38-fortran | 25 | 17 |
| numba | 625311 | llr40v10-oss120b-c-skills | 3 | 3 |
| numba | 625312 | llr40v10-qwen38-c-skills | 2 | 2 |
| numba | 625315 | llr40v10-oss120b-fortran-skills | 1 | 1 |
| numba | 625531 | llr40v11-oss120b-c | 39 | 32 |
| numba | 625532 | llr40v11-qwen38-c | 33 | 24 |
| numba | 625533 | llr40v11-kimi27sglang-c | 31 | 16 |
| numba | 625534 | llr40v11-kimi27sglang-c | 18 | 11 |
| numba | 625535 | llr40v11-oss120b-c-skills | 36 | 35 |
| numba | 625536 | llr40v11-qwen38-c-skills | 44 | 26 |
| numba | 625537 | llr40v11-kimi27sglang-c-skills | 21 | 12 |
| numba | 625538 | llr40v11-kimi27sglang-c-skills | 22 | 13 |
| numba | 625539 | llr40v11-oss120b-fortran | 29 | 26 |
| numba | 625540 | llr40v11-qwen38-fortran | 19 | 17 |
| numba | 625541 | llr40v11-kimi27sglang-fortran | 23 | 15 |
| numba | 625542 | llr40v11-kimi27sglang-fortran | 15 | 11 |
| numba | 625543 | llr40v11-oss120b-fortran-skills | 27 | 26 |
| numba | 625544 | llr40v11-qwen38-fortran-skills | 29 | 23 |
| numba | 625545 | llr40v11-kimi27sglang-fortran-skills | 39 | 15 |
| numba | 625546 | llr40v11-kimi27sglang-fortran-skills | 21 | 11 |
| numba | 626363 | v11w2-oss120b-c | 6 | 4 |
| numba | 626364 | v11w2-oss120b-c-skills | 2 | 2 |
| numba | 626365 | v11w2-qwen38-c | 22 | 11 |
| numba | 626366 | v11w2-qwen38-c-skills | 20 | 8 |
| numba | 626367 | v11w2-kimi27sglang-c | 11 | 7 |
| numba | 626368 | v11w2-kimi27sglang-c-skills | 18 | 10 |
| numba | 626369 | v11w2-oss120b-fortran | 6 | 6 |
| numba | 626370 | v11w2-oss120b-fortran-skills | 5 | 5 |
| numba | 626371 | v11w2-qwen38-fortran | 22 | 15 |
| numba | 626372 | v11w2-qwen38-fortran-skills | 26 | 13 |
| numba | 626373 | v11w2-kimi27sglang-fortran | 14 | 8 |
| numba | 626374 | v11w2-kimi27sglang-fortran-skills | 19 | 9 |
| numba | 626521 | gpuv2-llr40-qwen38-omp | 9 | 8 |
| numba | 626523 | gpuv4-llr40-qwen38-pytriton | 15 | 12 |
| numba | 626556 | gpuv2-llr40-qwen38-omp | 4 | 4 |
| numba | 626557 | gpuv4-llr40-qwen38-pytriton | 29 | 16 |
| numba | 626561 | gpuv2-llr40-qwen38-omp-skills | 11 | 11 |
| numba | 626645 | gpuv2-llr40-qwen38-hip | 24 | 14 |
| numba | 626646 | gpuv2-llr40-qwen38-hip-skills | 23 | 14 |
| numba | 626651 | gpuv4-llr40-qwen38-pytriton-skills | 30 | 18 |
| numba | 627005 | v11w2-oss120b-c | 1 | 1 |
| numba | 627007 | v11w2-qwen38-c | 8 | 4 |
| numba | 627008 | v11w2-qwen38-c-skills | 10 | 5 |
| numba | 627009 | v11w2-kimi27sglang-c | 4 | 3 |
| numba | 627010 | v11w2-kimi27sglang-c-skills | 21 | 3 |
| numba | 627011 | v11w2-oss120b-fortran | 4 | 4 |
| numba | 627012 | v11w2-oss120b-fortran-skills | 5 | 3 |
| numba | 627013 | v11w2-qwen38-fortran | 11 | 6 |
| numba | 627014 | v11w2-qwen38-fortran-skills | 4 | 3 |
| numba | 627015 | v11w2-kimi27sglang-fortran | 22 | 4 |
| numba | 627016 | v11w2-kimi27sglang-fortran-skills | 19 | 4 |
| numba | 627017 | gpuv2-llr40-qwen38-hip | 32 | 21 |
| numba | 627018 | gpuv2-llr40-qwen38-hip-skills | 30 | 19 |
| numba | 627019 | gpuv4-llr40-qwen38-pytriton | 20 | 16 |
| numba | 627020 | gpuv4-llr40-qwen38-pytriton-skills | 36 | 22 |
| numba | 627077 | gpuv2-llr40-qwen38-omp | 11 | 10 |
| numba | 627078 | gpuv2-llr40-qwen38-omp-skills | 14 | 11 |
| numba | 627167 | gpuv2-llr40-oss120b-hip | 36 | 35 |
| numba | 627168 | gpuv2-llr40-oss120b-hip-skills | 35 | 35 |
| numba | 627173 | gpuv2-llr40-oss120b-omp | 26 | 25 |
| numba | 627174 | gpuv2-llr40-oss120b-omp-skills | 23 | 22 |
| numba | 627175 | gpuv4-llr40-oss120b-pytriton | 38 | 35 |
| numba | 627176 | gpuv4-llr40-oss120b-pytriton-skills | 38 | 32 |
| numba | 627228 | gpuv2-llr40-qwen38-hip | 30 | 15 |
| numba | 627229 | gpuv2-llr40-qwen38-hip-skills | 24 | 15 |
| numba | 627230 | gpuv4-llr40-qwen38-pytriton | 30 | 18 |
| numba | 627231 | gpuv4-llr40-qwen38-pytriton-skills | 33 | 19 |
| numba | 627366 | v11w2-oss120b-c | 5 | 2 |
| numba | 627367 | v11w2-oss120b-c-skills | 1 | 1 |
| numba | 627368 | v11w2-qwen38-c | 7 | 1 |
| numba | 627369 | v11w2-qwen38-c-skills | 1 | 1 |
| numba | 627370 | v11w2-kimi27sglang-c | 10 | 3 |
| numba | 627371 | v11w2-kimi27sglang-c-skills | 9 | 2 |
| numba | 627372 | v11w2-oss120b-fortran | 4 | 3 |
| numba | 627373 | v11w2-oss120b-fortran-skills | 9 | 3 |
| numba | 627374 | v11w2-qwen38-fortran | 2 | 1 |
| numba | 627375 | v11w2-qwen38-fortran-skills | 3 | 1 |
| numba | 627376 | v11w2-kimi27sglang-fortran | 13 | 2 |
| numba | 627377 | v11w2-kimi27sglang-fortran-skills | 5 | 1 |
| numba | 628043 | v11w2-oss120b-c | 2 | 1 |
| numba | 628045 | v11w2-oss120b-fortran | 4 | 2 |
| numba | 628046 | v11w2-oss120b-fortran-skills | 8 | 2 |
| numba | 628047 | v11w2-qwen38-fortran | 1 | 1 |
| numba | 628162 | v11w2-oss120b-c | 1 | 1 |
| numba | 628163 | v11w2-oss120b-c-skills | 1 | 1 |
| numba | 628164 | v11w2-oss120b-fortran | 2 | 2 |
| numba | 628165 | v11w2-oss120b-fortran-skills | 2 | 2 |
| numba | 628542 | v11w2-oss120b-fortran-skills | 1 | 1 |

