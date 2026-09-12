# llr40 speed-up by kernel and denominator

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

## Per-kernel summary, geomean over one value per arm

| baseline | benchmark | arms | submissions | geomean_su | median_su | min_su | max_su | best_arm | best_source_path |
|---|---|---|---|---|---|---|---|---|---|
| c | tsvc_2_s1232 | 5 | 11 | 154.599 | 164.765 | 112.889 | 242.884 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s1232/621017.621017.llr40v10-kimi27sglang-c.n0.p13.w13/candidate_02_submission.txt |
| c | tsvc_2_s2275 | 5 | 13 | 96.274 | 121.033 | 42.154 | 125.947 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s2275/621016.621016.llr40v10-qwen38-c.n0.p17.w17/candidate_01_submission.txt |
| c | fuse_diamond | 5 | 12 | 65.050 | 60.915 | 57.385 | 88.908 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/fuse_diamond/621017.621017.llr40v10-kimi27sglang-c.n0.p4.w4/candidate_03_submission.txt |
| c | tsvc_2_s231 | 6 | 16 | 40.308 | 32.585 | 22.077 | 96.274 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s231/621016.621016.llr40v10-qwen38-c.n0.p18.w18/candidate_02_submission.txt |
| c | tsvc_2_s233 | 4 | 11 | 36.039 | 35.432 | 29.757 | 46.103 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s233/621017.621017.llr40v10-kimi27sglang-c.n0.p19.w19/candidate_07_submission.txt |
| c | tsvc_2_s2710 | 5 | 12 | 30.965 | 32.870 | 21.216 | 40.509 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s2710/621022.621022.llr40v10-kimi27sglang-fortran.n0.p3.w3/candidate_06_submission.txt |
| c | tsvc_2_s275 | 4 | 8 | 30.204 | 25.970 | 23.436 | 52.994 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s275/621385.621385.llr40v10-qwen38-fortran.n0.p24.w24/candidate_02_submission.txt |
| c | tsvc_2_s235 | 4 | 9 | 25.126 | 27.239 | 17.045 | 32.223 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s235/621022.621022.llr40v10-kimi27sglang-fortran.n0.p0.w0/candidate_02_submission.txt |
| c | fuse_stencil_through_transient | 6 | 14 | 24.347 | 24.266 | 23.436 | 26.146 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/fuse_stencil_through_transient/621016.621016.llr40v10-qwen38-c.n0.p6.w6/candidate_03_submission.txt |
| c | tsvc_2_vag | 5 | 15 | 23.389 | 22.974 | 22.746 | 25.126 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_vag/621383.621383.llr40v10-qwen38-c.n0.p34.w34/candidate_02_submission.txt |
| c | tsvc_2_s4112 | 5 | 7 | 22.254 | 22.298 | 21.428 | 22.746 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s4112/621382.621382.llr40v10-oss120b-c.n0.p33.w33/candidate_01_submission.txt |
| c | tsvc_2_s3111 | 5 | 13 | 21.174 | 21.216 | 21.006 | 21.428 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s3111/621016.621016.llr40v10-qwen38-c.n0.p27.w27/candidate_02_submission.txt |
| c | tsvc_2_s319 | 4 | 10 | 19.937 | 26.274 | 8.162 | 52.994 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s319/621022.621022.llr40v10-kimi27sglang-fortran.n0.p11.w11/candidate_04_submission.txt |
| c | tsvc_2_s311 | 4 | 8 | 17.474 | 19.496 | 12.152 | 20.186 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/tsvc_2_s311/621019.621019.llr40v10-oss120b-fortran.n0.p25.w25/candidate_01_submission.txt |
| c | tsvc_2_s318 | 5 | 12 | 14.109 | 15.898 | 8.162 | 17.737 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s318/621016.621016.llr40v10-qwen38-c.n0.p30.w30/candidate_01_submission.txt |
| c | argmax_with_index | 20 | 88 | 14.011 | 18.551 | 1.000 | 21.006 | llr40v9-kimi27sglang-fortran | sources/llr40v9-kimi27sglang-fortran/argmax_with_index/619186.619186.llr40v9-kimi27sglang-fortran.n0.p0.w0/candidate_05_submission.txt |
| c | compact_threshold_pack | 11 | 80 | 12.566 | 11.678 | 10.893 | 17.914 | llr40v9-qwen38-c | sources/llr40v9-qwen38-c/compact_threshold_pack/618217.618217.llr40v9-qwen38-c.n0.p1.w1/candidate_07_submission.txt |
| c | scatter_accum_dup | 11 | 31 | 12.386 | 10.678 | 9.476 | 22.974 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/scatter_accum_dup/621017.621017.llr40v10-kimi27sglang-c.n0.p9.w9/candidate_03_submission.txt |
| c | quasi_affine_reduce_odd | 4 | 11 | 10.210 | 10.261 | 10.059 | 10.261 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/quasi_affine_reduce_odd/621019.621019.llr40v10-oss120b-fortran.n0.p7.w7/candidate_01_submission.txt |
| c | tsvc_2_s3110 | 4 | 9 | 10.134 | 10.211 | 7.389 | 13.694 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s3110/621385.621385.llr40v10-qwen38-fortran.n0.p26.w26/candidate_02_submission.txt |
| c | tsvc_2_s316 | 5 | 12 | 8.114 | 13.694 | 1.000 | 13.831 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/tsvc_2_s316/621384.621384.llr40v10-oss120b-fortran.n0.p29.w29/candidate_01_submission.txt |
| c | segment_reduce_ragged | 5 | 17 | 7.158 | 11.223 | 1.149 | 11.795 | llr40v9-kimi27sglang-fortran | sources/llr40v9-kimi27sglang-fortran/segment_reduce_ragged/619186.619186.llr40v9-kimi27sglang-fortran.n0.p4.w4/candidate_04_submission.txt |
| c | fuse_move_ifs | 6 | 10 | 7.019 | 7.463 | 5.428 | 8.839 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/fuse_move_ifs/621017.621017.llr40v10-kimi27sglang-c.n0.p5.w5/candidate_02_submission.txt |
| c | ext_break_capture | 3 | 5 | 6.407 | 6.756 | 3.987 | 9.763 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/ext_break_capture/621383.621383.llr40v10-qwen38-c.n0.p2.w2/candidate_01_submission.txt |
| c | tsvc_2_s255 | 4 | 9 | 5.907 | 5.908 | 5.648 | 6.177 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s255/621022.621022.llr40v10-kimi27sglang-fortran.n0.p2.w2/candidate_02_submission.txt |
| c | versioned_distance_update | 11 | 15 | 5.862 | 3.610 | 3.469 | 30.055 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/versioned_distance_update/621385.621385.llr40v10-qwen38-fortran.n0.p37.w37/candidate_03_submission.txt |
| c | tsvc_2_s152 | 6 | 11 | 5.225 | 5.450 | 4.191 | 6.056 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s152/621015.621015.llr40v10-oss120b-c.n0.p15.w15/candidate_01_submission.txt |
| c | tsvc_2_vpvts | 5 | 10 | 5.144 | 5.216 | 4.583 | 6.056 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_vpvts/621020.621020.llr40v10-qwen38-fortran.n0.p35.w35/candidate_01_submission.txt |
| c | tsvc_2_vtvtv | 5 | 17 | 5.033 | 5.013 | 4.865 | 5.321 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_vtvtv/621022.621022.llr40v10-kimi27sglang-fortran.n0.p16.w16/candidate_03_submission.txt |
| c | tsvc_2_s252 | 5 | 10 | 4.258 | 4.108 | 3.719 | 5.063 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s252/621015.621015.llr40v10-oss120b-c.n0.p21.w21/candidate_01_submission.txt |
| c | scan_affine_decay | 14 | 42 | 3.917 | 5.038 | 1.000 | 9.016 | llr40v9-qwen38-fortran-skills | sources/llr40v9-qwen38-fortran-skills/scan_affine_decay/618228.618228.llr40v9-qwen38-fortran-skills.n0.p2.w2/candidate_03_submission.txt |
| c | ext_war_unit | 5 | 13 | 3.203 | 3.870 | 1.628 | 5.762 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/ext_war_unit/621385.621385.llr40v10-qwen38-fortran.n0.p3.w3/candidate_02_submission.txt |
| c | tsvc_2_s1244 | 4 | 10 | 2.732 | 3.604 | 1.000 | 4.629 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s1244/621383.621383.llr40v10-qwen38-c.n0.p14.w14/candidate_02_submission.txt |
| c | wf_diff_skew | 4 | 8 | 2.114 | 2.406 | 1.000 | 3.469 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/wf_diff_skew/621015.621015.llr40v10-oss120b-c.n0.p38.w38/candidate_01_submission.txt |
| c | tsvc_2_s115 | 4 | 9 | 1.733 | 2.058 | 1.000 | 2.130 | llr40v10-oss120b-fortran | sources/llr40v10-oss120b-fortran/tsvc_2_s115/621019.621019.llr40v10-oss120b-fortran.n0.p11.w11/candidate_01_submission.txt |
| c | wf_triangular | 3 | 5 | 1.596 | 1.890 | 1.000 | 2.152 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/wf_triangular/621382.621382.llr40v10-oss120b-c.n0.p39.w39/candidate_01_submission.txt |
| c | tsvc_2_s119 | 3 | 6 | 1.253 | 1.127 | 1.000 | 1.746 | llr40v10-kimi27sglang-c | sources/llr40v10-kimi27sglang-c/tsvc_2_s119/621017.621017.llr40v10-kimi27sglang-c.n0.p12.w12/candidate_01_submission.txt |
| c | tsvc_2_s323 | 3 | 3 | 1.013 | 1.020 | 1.000 | 1.020 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s323/621022.621022.llr40v10-kimi27sglang-fortran.n0.p12.w12/candidate_01_submission.txt |
| c | tsvc_2_s3112 | 3 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s3112/621382.621382.llr40v10-oss120b-c.n0.p28.w28/candidate_02_submission.txt |
| c | tsvc_2_s2233 | 0 | 0 | -- no submission -- | -- | -- | -- | -- | -- |
| numba | tsvc_2_s319 | 33 | 80 | 154.935 | 147.683 | 11.913 | 1886.259 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s319/627017.627017.gpuv2-llr40-qwen38-hip.n0.p31.w31/candidate_01_submission.txt |
| numba | tsvc_2_s255 | 31 | 76 | 81.868 | 209.208 | 3.140 | 3228.163 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s255/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p22.w22/candidate_01_submission.txt |
| numba | tsvc_2_s1232 | 31 | 75 | 77.895 | 79.690 | 1.000 | 1254.368 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s1232/626645.626645.gpuv2-llr40-qwen38-hip.n0.p13.w13/candidate_02_submission.txt |
| numba | tsvc_2_s2233 | 17 | 54 | 49.777 | 49.923 | 15.278 | 281.980 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s2233/627228.627228.gpuv2-llr40-qwen38-hip.n0.p16.w16/candidate_01_submission.txt |
| numba | tsvc_2_s233 | 31 | 68 | 41.723 | 56.817 | 1.375 | 650.451 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s233/627017.627017.gpuv2-llr40-qwen38-hip.n0.p19.w19/candidate_01_submission.txt |
| numba | fuse_diamond | 33 | 62 | 31.012 | 21.216 | 4.191 | 284.800 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/fuse_diamond/627167.627167.gpuv2-llr40-oss120b-hip.n0.p4.w4/candidate_01_submission.txt |
| numba | tsvc_2_s3111 | 34 | 70 | 25.661 | 29.171 | 3.140 | 380.067 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s3111/626645.626645.gpuv2-llr40-qwen38-hip.n0.p27.w27/candidate_01_submission.txt |
| numba | tsvc_2_s311 | 31 | 58 | 19.833 | 20.186 | 3.018 | 245.313 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s311/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p25.w25/candidate_01_submission.txt |
| numba | argmax_with_index | 25 | 47 | 19.741 | 15.741 | 1.000 | 250.243 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/argmax_with_index/627017.627017.gpuv2-llr40-qwen38-hip.n0.p0.w0/candidate_01_submission.txt |
| numba | tsvc_2_s323 | 25 | 34 | 15.150 | 15.898 | 1.000 | 98.209 | gpuv2-llr40-qwen38-omp-skills | sources/gpuv2-llr40-qwen38-omp-skills/tsvc_2_s323/627078.627078.gpuv2-llr40-qwen38-omp-skills.n0.p32.w32/candidate_01_submission.txt |
| numba | tsvc_2_s316 | 32 | 71 | 14.241 | 13.694 | 1.460 | 182.003 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s316/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p29.w29/candidate_02_submission.txt |
| numba | scatter_accum_dup | 22 | 36 | 13.570 | 10.262 | 1.000 | 228.807 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/scatter_accum_dup/627167.627167.gpuv2-llr40-oss120b-hip.n0.p9.w9/candidate_01_submission.txt |
| numba | tsvc_2_s3110 | 29 | 42 | 13.452 | 15.430 | 1.138 | 235.741 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s3110/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p26.w26/candidate_01_submission.txt |
| numba | segment_reduce_ragged | 26 | 52 | 13.347 | 13.030 | 5.762 | 111.772 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/segment_reduce_ragged/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p10.w10/candidate_01_submission.txt |
| numba | tsvc_2_s318 | 28 | 40 | 10.580 | 14.250 | 1.000 | 224.299 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s318/627017.627017.gpuv2-llr40-qwen38-hip.n0.p30.w30/candidate_02_submission.txt |
| numba | fuse_stencil_through_transient | 33 | 68 | 10.430 | 8.244 | 1.321 | 108.484 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/fuse_stencil_through_transient/627017.627017.gpuv2-llr40-qwen38-hip.n0.p6.w6/candidate_01_submission.txt |
| numba | quasi_affine_reduce_odd | 31 | 54 | 9.602 | 10.364 | 1.000 | 136.383 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/quasi_affine_reduce_odd/627017.627017.gpuv2-llr40-qwen38-hip.n0.p7.w7/candidate_01_submission.txt |
| numba | compact_threshold_pack | 20 | 27 | 8.738 | 14.179 | 1.000 | 16.218 | llr40v11-qwen38-c-skills | sources/llr40v11-qwen38-c-skills/compact_threshold_pack/625536.625536.llr40v11-qwen38-c-skills.n0.p1.w1/candidate_02_submission.txt |
| numba | tsvc_2_s1244 | 31 | 58 | 6.883 | 4.629 | 1.000 | 83.755 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s1244/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p14.w14/candidate_01_submission.txt |
| numba | fuse_move_ifs | 29 | 61 | 6.535 | 4.865 | 2.988 | 67.961 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/fuse_move_ifs/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p5.w5/candidate_01_submission.txt |
| numba | wf_triangular | 27 | 36 | 5.785 | 7.101 | 1.000 | 28.032 | llr40v11-qwen38-fortran-skills | sources/llr40v11-qwen38-fortran-skills/wf_triangular/625544.625544.llr40v11-qwen38-fortran-skills.n0.p39.w39/candidate_01_submission.txt |
| numba | tsvc_2_s2275 | 32 | 70 | 5.728 | 4.254 | 2.759 | 45.195 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/tsvc_2_s2275/627167.627167.gpuv2-llr40-oss120b-hip.n0.p17.w17/candidate_01_submission.txt |
| numba | versioned_distance_update | 27 | 45 | 5.667 | 3.794 | 1.000 | 41.323 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/versioned_distance_update/626645.626645.gpuv2-llr40-qwen38-hip.n0.p37.w37/candidate_01_submission.txt |
| numba | ext_break_capture | 27 | 40 | 4.901 | 4.538 | 1.000 | 54.059 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/ext_break_capture/627167.627167.gpuv2-llr40-oss120b-hip.n0.p2.w2/candidate_01_submission.txt |
| numba | tsvc_2_s152 | 32 | 74 | 4.865 | 3.469 | 2.548 | 54.059 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s152/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p15.w15/candidate_01_submission.txt |
| numba | tsvc_2_s252 | 32 | 67 | 4.446 | 3.703 | 1.000 | 64.663 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s252/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p21.w21/candidate_01_submission.txt |
| numba | scan_affine_decay | 17 | 23 | 4.147 | 6.239 | 1.474 | 10.572 | llr40v11-qwen38-c | sources/llr40v11-qwen38-c/scan_affine_decay/625532.625532.llr40v11-qwen38-c.n0.p8.w8/candidate_02_submission.txt |
| numba | tsvc_2_s3112 | 21 | 34 | 3.705 | 6.302 | 1.000 | 112.889 | gpuv2-llr40-oss120b-hip-skills | sources/gpuv2-llr40-oss120b-hip-skills/tsvc_2_s3112/627168.627168.gpuv2-llr40-oss120b-hip-skills.n0.p28.w28/candidate_01_submission.txt |
| numba | ext_war_unit | 29 | 44 | 3.250 | 4.865 | 1.062 | 24.877 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/ext_war_unit/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p3.w3/candidate_01_submission.txt |
| numba | tsvc_2_s119 | 27 | 40 | 2.752 | 2.261 | 1.000 | 53.524 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s119/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p12.w12/candidate_01_submission.txt |
| numba | wf_diff_skew | 25 | 31 | 2.654 | 3.367 | 1.051 | 6.623 | llr40v10-qwen38-c-skills | sources/llr40v10-qwen38-c-skills/wf_diff_skew/621024.621024.llr40v10-qwen38-c-skills.n0.p38.w38/candidate_02_submission.txt |
| numba | tsvc_2_s115 | 29 | 47 | 2.390 | 1.781 | 1.000 | 9.571 | v11w2-qwen38-fortran-skills | sources/v11w2-qwen38-fortran-skills/tsvc_2_s115/627014.627014.v11w2-qwen38-fortran-skills.n0.p0.w0/candidate_02_submission.txt |
| numba | tsvc_2_s231 | 28 | 46 | 2.178 | 1.999 | 1.000 | 26.672 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s231/626645.626645.gpuv2-llr40-qwen38-hip.n0.p18.w18/candidate_01_submission.txt |
| numba | tsvc_2_s2710 | 33 | 65 | 1.983 | 1.489 | 1.000 | 14.536 | gpuv2-llr40-oss120b-hip-skills | sources/gpuv2-llr40-oss120b-hip-skills/tsvc_2_s2710/627168.627168.gpuv2-llr40-oss120b-hip-skills.n0.p23.w23/candidate_01_submission.txt |
| numba | tsvc_2_vag | 27 | 44 | 1.983 | 1.000 | 1.000 | 11.678 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_vag/627228.627228.gpuv2-llr40-qwen38-hip.n0.p34.w34/candidate_01_submission.txt |
| numba | tsvc_2_s235 | 23 | 44 | 1.981 | 2.007 | 1.000 | 18.642 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/tsvc_2_s235/627167.627167.gpuv2-llr40-oss120b-hip.n0.p20.w20/candidate_01_submission.txt |
| numba | tsvc_2_s4112 | 24 | 57 | 1.797 | 1.010 | 1.000 | 11.111 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s4112/627017.627017.gpuv2-llr40-qwen38-hip.n0.p33.w33/candidate_03_submission.txt |
| numba | tsvc_2_s275 | 23 | 40 | 1.792 | 1.321 | 1.000 | 8.326 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s275/627017.627017.gpuv2-llr40-qwen38-hip.n0.p24.w24/candidate_01_submission.txt |
| numba | tsvc_2_vtvtv | 29 | 60 | 1.422 | 1.010 | 1.000 | 10.160 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_vtvtv/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p25.w25/candidate_01_submission.txt |
| numba | tsvc_2_vpvts | 26 | 66 | 1.318 | 1.020 | 1.000 | 9.289 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_vpvts/627228.627228.gpuv2-llr40-qwen38-hip.n0.p35.w35/candidate_04_submission.txt |

