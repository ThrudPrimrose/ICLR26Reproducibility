# llr40 speed-up by language

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

**C++ ran no arm of its own.** The 6 C++ submissions are
incidental, not a condition. C and Fortran are comparable here; C++ is absent by design and is
excluded from the paired table and the paired figure.

## Per-language summary, geomean over one value per kernel

| language | arms | submissions | kernels | geomean_su | median_su | min_su | max_su |
|---|---|---|---|---|---|---|---|
| c | 9 | 449 | 39 | 16.735 | 17.737 | 1.000 | 242.884 |
| cpp | 3 | 6 | 3 | 7.196 | 5.013 | 3.610 | 20.592 |
| fortran | 9 | 325 | 39 | 13.655 | 15.126 | 1.000 | 247.766 |

## Paired per-kernel view, C against Fortran

| benchmark | c_best_su | c_arms | c_submissions | fortran_best_su | fortran_arms | fortran_submissions | c_over_fortran |
|---|---|---|---|---|---|---|---|
| tsvc_2_s1232 | 242.884 | 3 | 15 | 169.757 | 3 | 5 | 1.431 |
| tsvc_2_s319 | 207.136 | 3 | 9 | 56.254 | 2 | 6 | 3.682 |
| tsvc_2_s2275 | 125.947 | 3 | 18 | 121.033 | 3 | 6 | 1.041 |
| tsvc_2_s233 | 98.209 | 3 | 12 | 29.757 | 1 | 2 | 3.300 |
| tsvc_2_s231 | 96.274 | 3 | 16 | 34.205 | 3 | 6 | 2.815 |
| fuse_diamond | 88.908 | 3 | 12 | 62.761 | 2 | 5 | 1.417 |
| tsvc_2_s2710 | 39.711 | 3 | 5 | 40.509 | 3 | 10 | 0.980 |
| tsvc_2_s255 | 38.543 | 3 | 8 | 247.766 | 3 | 9 | 0.156 |
| fuse_stencil_through_transient | 28.882 | 3 | 12 | 24.387 | 3 | 6 | 1.184 |
| tsvc_2_s3111 | 28.882 | 3 | 12 | 27.480 | 3 | 8 | 1.051 |
| tsvc_2_s275 | 28.032 | 2 | 4 | 52.994 | 3 | 5 | 0.529 |
| tsvc_2_vag | 25.126 | 2 | 6 | 23.436 | 3 | 10 | 1.072 |
| scatter_accum_dup | 22.974 | 6 | 25 | 11.678 | 5 | 13 | 1.967 |
| tsvc_2_s4112 | 22.746 | 3 | 4 | 22.521 | 3 | 5 | 1.010 |
| argmax_with_index | 21.006 | 9 | 49 | 21.006 | 8 | 38 | 1.000 |
| tsvc_2_s3110 | 20.798 | 3 | 7 | 13.694 | 3 | 7 | 1.519 |
| versioned_distance_update | 19.788 | 4 | 5 | 30.055 | 6 | 9 | 0.658 |
| tsvc_2_s311 | 19.593 | 3 | 10 | 20.388 | 3 | 5 | 0.961 |
| compact_threshold_pack | 17.914 | 7 | 45 | 15.126 | 4 | 37 | 1.184 |
| tsvc_2_s318 | 17.737 | 3 | 7 | 16.218 | 3 | 9 | 1.094 |
| tsvc_2_s235 | 17.045 | 2 | 6 | 32.223 | 3 | 5 | 0.529 |
| tsvc_2_s323 | 16.057 | 1 | 1 | 15.898 | 3 | 4 | 1.010 |
| ext_break_capture | 14.536 | 3 | 11 | 1.348 | 1 | 1 | 10.785 |
| segment_reduce_ragged | 13.969 | 3 | 14 | 13.424 | 3 | 9 | 1.041 |
| tsvc_2_s316 | 13.831 | 3 | 7 | 13.831 | 3 | 10 | 1.000 |
| wf_triangular | 13.291 | 3 | 5 | 3.719 | 2 | 4 | 3.574 |
| quasi_affine_reduce_odd | 10.364 | 3 | 12 | 10.364 | 3 | 6 | 1.000 |
| fuse_move_ifs | 8.839 | 3 | 10 | 7.538 | 3 | 6 | 1.173 |
| scan_affine_decay | 8.244 | 7 | 23 | 9.016 | 6 | 20 | 0.914 |
| tsvc_2_s152 | 6.056 | 3 | 19 | 5.063 | 3 | 9 | 1.196 |
| ext_war_unit | 5.996 | 3 | 9 | 5.762 | 3 | 9 | 1.041 |
| tsvc_2_s1244 | 5.592 | 3 | 10 | 1.020 | 2 | 3 | 5.482 |
| tsvc_2_vpvts | 5.216 | 3 | 6 | 6.056 | 3 | 5 | 0.861 |
| tsvc_2_s252 | 5.216 | 3 | 6 | 4.448 | 3 | 7 | 1.173 |
| tsvc_2_vtvtv | 5.013 | 2 | 10 | 5.321 | 3 | 7 | 0.942 |
| wf_diff_skew | 3.469 | 2 | 3 | 2.574 | 3 | 6 | 1.348 |
| tsvc_2_s115 | 3.018 | 3 | 9 | 2.130 | 3 | 7 | 1.417 |
| tsvc_2_s119 | 1.746 | 3 | 5 | 3.434 | 2 | 3 | 0.508 |
| tsvc_2_s3112 | 1.000 | 1 | 2 | 1.000 | 2 | 3 | 1.000 |
| tsvc_2_s2233 | -- no submission -- | 0 | 0 | -- | 0 | 0 | -- |

