# llr40 speed-up by language and denominator

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

**C++ ran no arm of its own.** The 6 C++ submissions are
incidental, not a condition. C and Fortran are comparable here; C++ is absent by design and is
excluded from the paired table and the paired figure.

## Per-language summary with the PAIRED C-against-Fortran test

| baseline | language | arms | kernels | geomean_su | median_su | min_su | max_su | paired_n | hl_c_over_fortran | hl_ci_low | hl_ci_high | hl_pvalue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c | c | 9 | 38 | 13.326 | 15.369 | 1.000 | 242.884 | 67 | 1.099 | 1.030 | 1.196 | 0.004 |
| c | cpp | 3 | 3 | 7.196 | 5.013 | 3.610 | 20.592 | 0 | -- | -- | -- | -- |
| c | fortran | 9 | 38 | 11.436 | 13.297 | 1.000 | 169.757 | 67 | 1.099 | 1.030 | 1.196 | 0.004 |
| numba | c | 21 | 40 | 15.616 | 13.900 | 1.062 | 711.388 | 195 | 1.088 | 1.030 | 1.167 | 0.001 |
| numba | fortran | 17 | 40 | 10.969 | 10.735 | 1.051 | 250.243 | 195 | 1.088 | 1.030 | 1.167 | 0.001 |
| numba | hip | 4 | 38 | 70.002 | 75.858 | 1.460 | 3228.163 | 0 | -- | -- | -- | -- |
| numba | python | 4 | 40 | 7.999 | 7.223 | 1.000 | 247.766 | 0 | -- | -- | -- | -- |

## Paired per-kernel view, C against Fortran

| baseline | benchmark | c_best_su | c_arms | c_submissions | fortran_best_su | fortran_arms | fortran_submissions | c_over_fortran |
|---|---|---|---|---|---|---|---|---|
| c | tsvc_2_s1232 | 242.884 | 3 | 8 | 169.757 | 2 | 3 | 1.431 |
| c | tsvc_2_s2275 | 125.947 | 3 | 10 | 121.033 | 2 | 3 | 1.041 |
| c | tsvc_2_s231 | 96.274 | 3 | 11 | 34.205 | 3 | 5 | 2.815 |
| c | fuse_diamond | 88.908 | 3 | 8 | 62.761 | 2 | 4 | 1.417 |
| c | tsvc_2_s233 | 46.103 | 3 | 9 | 29.757 | 1 | 2 | 1.549 |
| c | tsvc_2_s319 | 44.304 | 2 | 4 | 52.994 | 2 | 6 | 0.836 |
| c | tsvc_2_s2710 | 39.711 | 2 | 3 | 40.509 | 3 | 9 | 0.980 |
| c | tsvc_2_s275 | 28.032 | 1 | 3 | 52.994 | 3 | 5 | 0.529 |
| c | fuse_stencil_through_transient | 26.146 | 3 | 9 | 24.387 | 3 | 5 | 1.072 |
| c | tsvc_2_vag | 25.126 | 2 | 5 | 23.436 | 3 | 10 | 1.072 |
| c | scatter_accum_dup | 22.974 | 6 | 18 | 11.223 | 5 | 13 | 2.047 |
| c | tsvc_2_s4112 | 22.746 | 2 | 3 | 22.521 | 3 | 4 | 1.010 |
| c | tsvc_2_s3111 | 21.428 | 2 | 6 | 21.216 | 3 | 7 | 1.010 |
| c | argmax_with_index | 21.006 | 9 | 49 | 21.006 | 8 | 35 | 1.000 |
| c | versioned_distance_update | 19.788 | 4 | 5 | 30.055 | 6 | 9 | 0.658 |
| c | tsvc_2_s311 | 19.399 | 2 | 5 | 20.186 | 2 | 3 | 0.961 |
| c | compact_threshold_pack | 17.914 | 7 | 43 | 12.900 | 4 | 37 | 1.389 |
| c | tsvc_2_s318 | 17.737 | 2 | 3 | 16.218 | 3 | 9 | 1.094 |
| c | tsvc_2_s235 | 17.045 | 1 | 4 | 32.223 | 3 | 5 | 0.529 |
| c | tsvc_2_s316 | 13.694 | 2 | 3 | 13.831 | 3 | 9 | 0.990 |
| c | segment_reduce_ragged | 11.448 | 3 | 9 | 11.795 | 2 | 8 | 0.971 |
| c | tsvc_2_s3110 | 10.364 | 1 | 2 | 13.694 | 3 | 7 | 0.757 |
| c | quasi_affine_reduce_odd | 10.261 | 2 | 8 | 10.261 | 2 | 3 | 1.000 |
| c | ext_break_capture | 9.763 | 3 | 5 | -- | 0 | 0 | -- |
| c | fuse_move_ifs | 8.839 | 3 | 5 | 7.538 | 3 | 5 | 1.173 |
| c | scan_affine_decay | 6.558 | 7 | 21 | 9.016 | 6 | 20 | 0.727 |
| c | tsvc_2_s152 | 6.056 | 3 | 6 | 4.963 | 3 | 5 | 1.220 |
| c | tsvc_2_s255 | 5.996 | 1 | 2 | 6.177 | 3 | 7 | 0.971 |
| c | tsvc_2_vpvts | 5.216 | 2 | 5 | 6.056 | 3 | 5 | 0.861 |
| c | tsvc_2_s252 | 5.063 | 2 | 4 | 4.448 | 3 | 6 | 1.138 |
| c | ext_war_unit | 5.013 | 3 | 6 | 5.762 | 2 | 7 | 0.870 |
| c | tsvc_2_vtvtv | 5.013 | 2 | 10 | 5.321 | 3 | 7 | 0.942 |
| c | tsvc_2_s1244 | 4.629 | 3 | 8 | 1.000 | 1 | 2 | 4.629 |
| c | wf_diff_skew | 3.469 | 1 | 2 | 2.574 | 3 | 6 | 1.348 |
| c | wf_triangular | 2.152 | 1 | 2 | 1.890 | 2 | 3 | 1.138 |
| c | tsvc_2_s115 | 2.088 | 2 | 4 | 2.130 | 2 | 5 | 0.980 |
| c | tsvc_2_s119 | 1.746 | 2 | 4 | 1.000 | 1 | 2 | 1.746 |
| c | tsvc_2_s3112 | 1.000 | 1 | 2 | 1.000 | 2 | 3 | 1.000 |
| c | tsvc_2_s2233 | -- no submission -- | 0 | 0 | -- | 0 | 0 | -- |
| c | tsvc_2_s323 | -- no submission -- | 0 | 0 | 1.020 | 3 | 3 | -- |
| numba | tsvc_2_s319 | 711.388 | 15 | 37 | 152.158 | 10 | 24 | 4.675 |
| numba | tsvc_2_s255 | 247.766 | 12 | 23 | 250.243 | 11 | 27 | 0.990 |
| numba | tsvc_2_s233 | 146.221 | 14 | 24 | 77.346 | 10 | 29 | 1.890 |
| numba | tsvc_2_s1232 | 124.700 | 14 | 39 | 93.443 | 10 | 18 | 1.335 |
| numba | tsvc_2_s323 | 98.209 | 9 | 10 | 57.385 | 10 | 14 | 1.711 |
| numba | tsvc_2_s1244 | 78.120 | 13 | 24 | 46.564 | 11 | 22 | 1.678 |
| numba | tsvc_2_s2233 | 67.289 | 6 | 21 | 64.663 | 6 | 27 | 1.041 |
| numba | fuse_diamond | 50.422 | 15 | 22 | 46.564 | 10 | 16 | 1.083 |
| numba | tsvc_2_s3111 | 29.462 | 15 | 32 | 30.659 | 11 | 23 | 0.961 |
| numba | wf_triangular | 27.480 | 13 | 19 | 28.032 | 10 | 13 | 0.980 |
| numba | versioned_distance_update | 26.939 | 11 | 18 | 35.241 | 10 | 20 | 0.764 |
| numba | argmax_with_index | 20.798 | 8 | 12 | 21.006 | 10 | 22 | 0.990 |
| numba | tsvc_2_s3110 | 20.798 | 13 | 22 | 19.986 | 10 | 13 | 1.041 |
| numba | tsvc_2_s311 | 20.388 | 14 | 30 | 20.592 | 11 | 12 | 0.990 |
| numba | ext_break_capture | 19.016 | 11 | 21 | 8.927 | 10 | 13 | 2.130 |
| numba | tsvc_2_s318 | 18.457 | 14 | 18 | 17.737 | 9 | 15 | 1.041 |
| numba | scatter_accum_dup | 18.093 | 9 | 16 | 12.274 | 7 | 11 | 1.474 |
| numba | fuse_stencil_through_transient | 16.876 | 15 | 35 | 11.001 | 10 | 13 | 1.534 |
| numba | compact_threshold_pack | 16.218 | 9 | 12 | 15.898 | 9 | 11 | 1.020 |
| numba | segment_reduce_ragged | 13.969 | 13 | 26 | 14.250 | 9 | 20 | 0.980 |
| numba | tsvc_2_s316 | 13.831 | 13 | 26 | 13.969 | 11 | 26 | 0.990 |
| numba | tsvc_2_s2275 | 12.274 | 14 | 31 | 4.361 | 11 | 20 | 2.815 |
| numba | tsvc_2_s152 | 11.223 | 13 | 30 | 4.191 | 11 | 20 | 2.678 |
| numba | scan_affine_decay | 10.572 | 8 | 12 | 7.689 | 5 | 6 | 1.375 |
| numba | quasi_affine_reduce_odd | 10.468 | 13 | 21 | 10.468 | 11 | 22 | 1.000 |
| numba | tsvc_2_vag | 9.106 | 12 | 17 | 1.051 | 8 | 14 | 8.664 |
| numba | tsvc_2_s3112 | 8.839 | 7 | 11 | 8.664 | 9 | 16 | 1.020 |
| numba | fuse_move_ifs | 7.389 | 11 | 21 | 4.963 | 10 | 15 | 1.489 |
| numba | ext_war_unit | 6.756 | 14 | 22 | 6.365 | 9 | 14 | 1.062 |
| numba | tsvc_2_s4112 | 6.756 | 10 | 20 | 1.051 | 8 | 21 | 6.428 |
| numba | wf_diff_skew | 6.623 | 10 | 12 | 4.769 | 10 | 13 | 1.389 |
| numba | tsvc_2_s115 | 6.558 | 13 | 25 | 9.571 | 11 | 16 | 0.685 |
| numba | tsvc_2_s252 | 6.493 | 14 | 28 | 5.705 | 10 | 20 | 1.138 |
| numba | tsvc_2_s119 | 5.216 | 10 | 21 | 3.434 | 11 | 13 | 1.519 |
| numba | tsvc_2_s2710 | 4.493 | 14 | 29 | 1.678 | 11 | 24 | 2.678 |
| numba | tsvc_2_s275 | 4.275 | 11 | 18 | 1.746 | 6 | 12 | 2.449 |
| numba | tsvc_2_s231 | 2.424 | 10 | 19 | 2.424 | 10 | 12 | 1.000 |
| numba | tsvc_2_s235 | 2.377 | 10 | 26 | 2.261 | 8 | 13 | 1.051 |
| numba | tsvc_2_vtvtv | 1.431 | 12 | 24 | 1.474 | 9 | 21 | 0.971 |
| numba | tsvc_2_vpvts | 1.062 | 12 | 38 | 1.094 | 8 | 10 | 0.971 |

