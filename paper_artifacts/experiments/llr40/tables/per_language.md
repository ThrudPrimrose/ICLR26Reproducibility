# llr40 speed-up by language

Snapshot: **2026-09-08T08:40:02Z**. The campaign was UNFINISHED when this was extracted, so every
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

**C++ ran no arm of its own.** The 6 C++ submissions are
incidental, not a condition. C and Fortran are comparable here; C++ is absent by design and is
excluded from the paired table and the paired figure.

## Per-language summary, geomean over one value per kernel

| language | arms | submissions | kernels | geomean_su | median_su | min_su | max_su |
|---|---|---|---|---|---|---|---|
| c | 28 | 1218 | 40 | 25.377 | 20.902 | 5.013 | 711.388 |
| cpp | 3 | 6 | 3 | 7.196 | 5.013 | 3.610 | 20.592 |
| fortran | 24 | 948 | 40 | 21.471 | 20.799 | 3.434 | 250.243 |
| hip | 4 | 234 | 38 | 70.241 | 75.858 | 1.460 | 3228.163 |
| python | 4 | 269 | 40 | 8.001 | 7.223 | 1.000 | 247.766 |

## Paired per-kernel view, C against Fortran

| benchmark | c_best_su | c_arms | c_submissions | fortran_best_su | fortran_arms | fortran_submissions | c_over_fortran |
|---|---|---|---|---|---|---|---|
| tsvc_2_s319 | 711.388 | 16 | 41 | 152.158 | 11 | 30 | 4.675 |
| tsvc_2_s255 | 247.766 | 13 | 25 | 250.243 | 12 | 34 | 0.990 |
| tsvc_2_s1232 | 242.884 | 15 | 47 | 169.757 | 11 | 21 | 1.431 |
| tsvc_2_s233 | 146.221 | 15 | 33 | 78.120 | 11 | 31 | 1.872 |
| tsvc_2_s2275 | 125.947 | 16 | 41 | 121.033 | 12 | 23 | 1.041 |
| tsvc_2_s323 | 98.209 | 9 | 10 | 57.385 | 11 | 17 | 1.711 |
| tsvc_2_s231 | 96.274 | 11 | 30 | 34.205 | 11 | 17 | 2.815 |
| fuse_diamond | 88.908 | 16 | 30 | 62.761 | 11 | 20 | 1.417 |
| tsvc_2_s1244 | 78.120 | 15 | 32 | 46.564 | 12 | 24 | 1.678 |
| tsvc_2_s2233 | 67.289 | 6 | 19 | 64.663 | 4 | 16 | 1.041 |
| tsvc_2_s2710 | 39.711 | 16 | 32 | 40.509 | 12 | 33 | 0.980 |
| tsvc_2_s3111 | 29.462 | 16 | 38 | 30.659 | 12 | 30 | 0.961 |
| fuse_stencil_through_transient | 28.882 | 16 | 44 | 24.387 | 12 | 18 | 1.184 |
| tsvc_2_s275 | 28.032 | 12 | 21 | 52.994 | 9 | 17 | 0.529 |
| wf_triangular | 27.480 | 14 | 21 | 28.032 | 11 | 16 | 0.980 |
| versioned_distance_update | 26.939 | 14 | 23 | 35.241 | 15 | 29 | 0.764 |
| tsvc_2_vag | 25.126 | 13 | 22 | 23.436 | 10 | 24 | 1.072 |
| scatter_accum_dup | 22.974 | 13 | 34 | 12.274 | 12 | 24 | 1.872 |
| tsvc_2_s4112 | 22.746 | 12 | 23 | 22.521 | 9 | 18 | 1.010 |
| argmax_with_index | 21.006 | 17 | 61 | 21.006 | 16 | 51 | 1.000 |
| tsvc_2_s3110 | 20.798 | 14 | 24 | 19.986 | 12 | 20 | 1.041 |
| tsvc_2_s311 | 20.592 | 15 | 35 | 20.592 | 12 | 15 | 1.000 |
| ext_break_capture | 19.016 | 12 | 26 | 8.927 | 8 | 10 | 2.130 |
| tsvc_2_s318 | 18.457 | 15 | 21 | 17.737 | 11 | 24 | 1.041 |
| compact_threshold_pack | 17.914 | 13 | 52 | 15.898 | 12 | 47 | 1.127 |
| tsvc_2_s235 | 17.045 | 11 | 30 | 32.223 | 10 | 18 | 0.529 |
| segment_reduce_ragged | 13.969 | 13 | 34 | 14.392 | 11 | 28 | 0.971 |
| tsvc_2_s316 | 13.831 | 14 | 29 | 13.969 | 12 | 35 | 0.990 |
| tsvc_2_s152 | 11.223 | 14 | 36 | 5.063 | 12 | 25 | 2.217 |
| scan_affine_decay | 10.572 | 13 | 31 | 9.016 | 10 | 24 | 1.173 |
| quasi_affine_reduce_odd | 10.468 | 14 | 29 | 10.468 | 12 | 25 | 1.000 |
| fuse_move_ifs | 8.839 | 12 | 26 | 7.538 | 11 | 20 | 1.173 |
| tsvc_2_s3112 | 8.839 | 8 | 13 | 8.664 | 10 | 19 | 1.020 |
| ext_war_unit | 6.756 | 15 | 28 | 6.365 | 10 | 21 | 1.062 |
| wf_diff_skew | 6.623 | 11 | 14 | 4.769 | 12 | 19 | 1.389 |
| tsvc_2_s115 | 6.558 | 14 | 29 | 9.571 | 12 | 21 | 0.685 |
| tsvc_2_s252 | 6.493 | 15 | 32 | 5.705 | 11 | 26 | 1.138 |
| tsvc_2_s119 | 5.216 | 12 | 25 | 3.434 | 12 | 15 | 1.519 |
| tsvc_2_vpvts | 5.216 | 14 | 43 | 6.056 | 11 | 15 | 0.861 |
| tsvc_2_vtvtv | 5.013 | 13 | 34 | 5.321 | 11 | 28 | 0.942 |

