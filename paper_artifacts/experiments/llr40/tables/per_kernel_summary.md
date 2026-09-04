# llr40 speed-up by kernel

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

## Per-kernel summary, geomean over one value per arm

| benchmark | arms | submissions | geomean_su | median_su | min_su | max_su | best_arm | best_source_path |
|---|---|---|---|---|---|---|---|---|
| tsvc_2_s1232 | 6 | 20 | 125.322 | 139.962 | 43.865 | 242.884 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s1232/621017.621017.llr40v10-kimi27sglang-c.n0.p13.w13/candidate_02_submission.txt |
| fuse_diamond | 5 | 17 | 65.050 | 60.915 | 57.385 | 88.908 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/fuse_diamond/621017.621017.llr40v10-kimi27sglang-c.n0.p4.w4/candidate_03_submission.txt |
| tsvc_2_s2275 | 6 | 24 | 57.480 | 112.642 | 4.318 | 125.947 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s2275/621016.621016.llr40v10-qwen38-c.n0.p17.w17/candidate_01_submission.txt |
| tsvc_2_s233 | 4 | 14 | 44.970 | 38.229 | 29.757 | 98.209 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s233/621727.621727.llr40v10-qwen38-c.n0.p19.w19/candidate_02_submission.txt |
| tsvc_2_s231 | 6 | 22 | 40.308 | 32.585 | 22.077 | 96.274 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s231/621016.621016.llr40v10-qwen38-c.n0.p18.w18/candidate_02_submission.txt |
| tsvc_2_s319 | 5 | 15 | 37.335 | 56.254 | 8.162 | 207.136 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s319/621727.621727.llr40v10-qwen38-c.n0.p31.w31/candidate_02_submission.txt |
| fuse_stencil_through_transient | 6 | 18 | 25.210 | 24.387 | 23.670 | 28.882 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/fuse_stencil_through_transient/621017.621017.llr40v10-kimi27sglang-c.n0.p6.w6/candidate_01_submission.txt |
| tsvc_2_vag | 5 | 16 | 23.389 | 22.974 | 22.746 | 25.126 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_vag/621383.621383.llr40v10-qwen38-c.n0.p34.w34/candidate_02_submission.txt |
| tsvc_2_s3111 | 6 | 20 | 23.358 | 21.429 | 21.006 | 28.882 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s3111/622265.622265.llr40v10-qwen38-c.n0.p27.w27/candidate_01_submission.txt |
| tsvc_2_s2710 | 6 | 15 | 19.016 | 29.124 | 1.661 | 40.509 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s2710/621022.621022.llr40v10-kimi27sglang-fortran.n0.p3.w3/candidate_06_submission.txt |
| tsvc_2_s311 | 6 | 15 | 16.988 | 19.593 | 12.152 | 20.388 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s311/622266.622266.llr40v10-qwen38-fortran.n0.p25.w25/candidate_01_submission.txt |
| tsvc_2_s275 | 5 | 9 | 16.910 | 23.907 | 1.661 | 52.994 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s275/621385.621385.llr40v10-qwen38-fortran.n0.p24.w24/candidate_02_submission.txt |
| tsvc_2_s255 | 6 | 17 | 16.191 | 7.104 | 5.996 | 247.766 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s255/621728.621728.llr40v10-qwen38-fortran.n0.p22.w22/candidate_01_submission.txt |
| argmax_with_index | 20 | 91 | 16.081 | 18.645 | 3.794 | 21.006 | llr40v9-kimi27sglang-c | sources/llr40v9-kimi27sglang-c/argmax_with_index/619185.619185.llr40v9-kimi27sglang-c.n0.p0.w0/candidate_09_submission.txt |
| tsvc_2_s235 | 5 | 11 | 15.492 | 23.204 | 2.239 | 32.223 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s235/621022.621022.llr40v10-kimi27sglang-fortran.n0.p0.w0/candidate_02_submission.txt |
| tsvc_2_s318 | 6 | 16 | 14.179 | 15.437 | 8.162 | 17.737 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s318/621016.621016.llr40v10-qwen38-c.n0.p30.w30/candidate_01_submission.txt |
| tsvc_2_s4112 | 6 | 9 | 13.313 | 22.298 | 1.020 | 22.746 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s4112/621382.621382.llr40v10-oss120b-c.n0.p33.w33/candidate_01_submission.txt |
| compact_threshold_pack | 11 | 82 | 13.147 | 12.646 | 11.335 | 17.914 | llr40v9-qwen38-c | sources/llr40v9-qwen38-c/compact_threshold_pack/618217.618217.llr40v9-qwen38-c.n0.p1.w1/candidate_06_submission.txt |
| scatter_accum_dup | 11 | 38 | 13.136 | 11.448 | 9.476 | 22.974 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/scatter_accum_dup/621017.621017.llr40v10-kimi27sglang-c.n0.p9.w9/candidate_03_submission.txt |
| tsvc_2_s3110 | 6 | 14 | 12.173 | 12.029 | 7.389 | 20.798 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s3110/621727.621727.llr40v10-qwen38-c.n0.p26.w26/candidate_01_submission.txt |
| tsvc_2_s316 | 6 | 17 | 11.316 | 13.762 | 4.275 | 13.831 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/tsvc_2_s316/621019.621019.llr40v10-oss120b-fortran.n0.p29.w29/candidate_01_submission.txt |
| quasi_affine_reduce_odd | 6 | 18 | 10.261 | 10.313 | 10.059 | 10.364 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/quasi_affine_reduce_odd/621728.621728.llr40v10-qwen38-fortran.n0.p7.w7/candidate_01_submission.txt |
| segment_reduce_ragged | 6 | 23 | 8.550 | 12.609 | 1.149 | 13.969 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/segment_reduce_ragged/621729.621729.llr40v10-kimi27sglang-c.n0.p10.w10/candidate_03_submission.txt |
| fuse_move_ifs | 6 | 16 | 7.353 | 7.463 | 5.819 | 8.839 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/fuse_move_ifs/621017.621017.llr40v10-kimi27sglang-c.n0.p5.w5/candidate_02_submission.txt |
| versioned_distance_update | 11 | 15 | 5.862 | 3.610 | 3.469 | 30.055 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/versioned_distance_update/621385.621385.llr40v10-qwen38-fortran.n0.p37.w37/candidate_03_submission.txt |
| tsvc_2_s152 | 6 | 28 | 5.392 | 5.500 | 4.538 | 6.056 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s152/621015.621015.llr40v10-oss120b-c.n0.p15.w15/candidate_01_submission.txt |
| ext_break_capture | 4 | 12 | 5.255 | 6.875 | 1.348 | 14.536 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/ext_break_capture/621729.621729.llr40v10-kimi27sglang-c.n0.p2.w2/candidate_02_submission.txt |
| tsvc_2_vtvtv | 5 | 17 | 5.033 | 5.013 | 4.865 | 5.321 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_vtvtv/621022.621022.llr40v10-kimi27sglang-fortran.n0.p16.w16/candidate_03_submission.txt |
| scan_affine_decay | 14 | 44 | 4.560 | 5.244 | 1.000 | 9.016 | llr40v9-qwen38-fortran-skills | sources/llr40v9-qwen38-fortran-skills/scan_affine_decay/618228.618228.llr40v9-qwen38-fortran-skills.n0.p2.w2/candidate_03_submission.txt |
| tsvc_2_s252 | 6 | 13 | 4.404 | 4.278 | 3.719 | 5.216 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s252/621018.621018.llr40v10-kimi27sglang-c.n0.p1.w1/candidate_02_submission.txt |
| tsvc_2_s323 | 4 | 5 | 4.017 | 8.459 | 1.000 | 16.057 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s323/621727.621727.llr40v10-qwen38-c.n0.p32.w32/candidate_01_submission.txt |
| tsvc_2_vpvts | 6 | 11 | 3.928 | 4.923 | 1.020 | 6.056 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_vpvts/621020.621020.llr40v10-qwen38-fortran.n0.p35.w35/candidate_01_submission.txt |
| wf_triangular | 5 | 9 | 3.756 | 3.719 | 1.890 | 13.291 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/wf_triangular/621730.621730.llr40v10-kimi27sglang-c.n0.p19.w19/candidate_01_submission.txt |
| ext_war_unit | 6 | 18 | 2.838 | 3.433 | 1.000 | 5.996 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/ext_war_unit/621729.621729.llr40v10-kimi27sglang-c.n0.p3.w3/candidate_03_submission.txt |
| tsvc_2_s1244 | 5 | 13 | 2.330 | 2.625 | 1.000 | 5.592 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s1244/622265.622265.llr40v10-qwen38-c.n0.p14.w14/candidate_01_submission.txt |
| wf_diff_skew | 5 | 9 | 1.952 | 2.239 | 1.000 | 3.469 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/wf_diff_skew/621015.621015.llr40v10-oss120b-c.n0.p38.w38/candidate_01_submission.txt |
| tsvc_2_s115 | 6 | 16 | 1.869 | 2.058 | 1.116 | 3.018 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s115/621729.621729.llr40v10-kimi27sglang-c.n0.p11.w11/candidate_02_submission.txt |
| tsvc_2_s119 | 5 | 8 | 1.559 | 1.361 | 1.000 | 3.434 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s119/621731.621731.llr40v10-kimi27sglang-fortran.n0.p12.w12/candidate_01_submission.txt |
| tsvc_2_s3112 | 3 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/tsvc_2_s3112/621019.621019.llr40v10-oss120b-fortran.n0.p28.w28/candidate_01_submission.txt |
| tsvc_2_s2233 | 0 | 0 | -- no submission -- | -- | -- | -- | -- | -- |

