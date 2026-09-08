# llr40 speed-up by kernel

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

## Per-kernel summary, geomean over one value per arm

| benchmark | arms | submissions | geomean_su | median_su | min_su | max_su | best_arm | best_source_path |
|---|---|---|---|---|---|---|---|---|
| tsvc_2_s319 | 35 | 90 | 132.033 | 147.683 | 8.162 | 1886.259 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s319/627017.627017.gpuv2-llr40-qwen38-hip.n0.p31.w31/candidate_01_submission.txt |
| tsvc_2_s1232 | 33 | 86 | 85.464 | 87.156 | 1.000 | 1254.368 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s1232/626645.626645.gpuv2-llr40-qwen38-hip.n0.p13.w13/candidate_02_submission.txt |
| tsvc_2_s255 | 33 | 85 | 70.105 | 209.208 | 3.140 | 3228.163 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s255/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p22.w22/candidate_01_submission.txt |
| tsvc_2_s2233 | 15 | 41 | 54.455 | 56.254 | 15.430 | 281.980 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s2233/627228.627228.gpuv2-llr40-qwen38-hip.n0.p16.w16/candidate_01_submission.txt |
| tsvc_2_s233 | 33 | 79 | 41.724 | 55.146 | 1.375 | 650.451 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s233/627017.627017.gpuv2-llr40-qwen38-hip.n0.p19.w19/candidate_01_submission.txt |
| fuse_diamond | 35 | 74 | 35.736 | 22.746 | 4.191 | 284.800 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/fuse_diamond/627167.627167.gpuv2-llr40-oss120b-hip.n0.p4.w4/candidate_01_submission.txt |
| tsvc_2_s3111 | 36 | 83 | 25.419 | 29.171 | 3.140 | 380.067 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s3111/626645.626645.gpuv2-llr40-qwen38-hip.n0.p27.w27/candidate_01_submission.txt |
| tsvc_2_s311 | 33 | 66 | 19.575 | 20.186 | 3.018 | 250.243 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s311/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p25.w25/candidate_01_submission.txt |
| argmax_with_index | 43 | 129 | 19.349 | 18.274 | 2.929 | 250.243 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/argmax_with_index/627017.627017.gpuv2-llr40-qwen38-hip.n0.p0.w0/candidate_01_submission.txt |
| tsvc_2_s316 | 34 | 83 | 14.321 | 13.694 | 1.460 | 182.003 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s316/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p29.w29/candidate_02_submission.txt |
| tsvc_2_s323 | 26 | 37 | 13.699 | 15.819 | 1.000 | 98.209 | gpuv2-llr40-qwen38-omp-skills | sources/gpuv2-llr40-qwen38-omp-skills/tsvc_2_s323/627078.627078.gpuv2-llr40-qwen38-omp-skills.n0.p32.w32/candidate_01_submission.txt |
| scatter_accum_dup | 31 | 67 | 13.411 | 10.364 | 1.000 | 228.807 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/scatter_accum_dup/627167.627167.gpuv2-llr40-oss120b-hip.n0.p9.w9/candidate_01_submission.txt |
| tsvc_2_s3110 | 32 | 51 | 13.233 | 14.902 | 1.138 | 238.098 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s3110/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p26.w26/candidate_01_submission.txt |
| fuse_stencil_through_transient | 36 | 82 | 12.339 | 8.536 | 1.321 | 108.484 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/fuse_stencil_through_transient/627017.627017.gpuv2-llr40-qwen38-hip.n0.p6.w6/candidate_01_submission.txt |
| segment_reduce_ragged | 28 | 68 | 12.165 | 12.836 | 1.149 | 111.772 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/segment_reduce_ragged/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p10.w10/candidate_01_submission.txt |
| tsvc_2_s318 | 31 | 52 | 11.841 | 14.682 | 1.000 | 224.299 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/tsvc_2_s318/627017.627017.gpuv2-llr40-qwen38-hip.n0.p30.w30/candidate_02_submission.txt |
| compact_threshold_pack | 27 | 103 | 10.487 | 13.969 | 1.000 | 17.914 | llr40v9-qwen38-c | sources/llr40v9-qwen38-c/compact_threshold_pack/618217.618217.llr40v9-qwen38-c.n0.p1.w1/candidate_07_submission.txt |
| quasi_affine_reduce_odd | 33 | 65 | 9.643 | 10.364 | 1.000 | 136.383 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/quasi_affine_reduce_odd/627017.627017.gpuv2-llr40-qwen38-hip.n0.p7.w7/candidate_01_submission.txt |
| tsvc_2_s2275 | 35 | 83 | 8.841 | 4.318 | 3.203 | 125.947 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s2275/621016.621016.llr40v10-qwen38-c.n0.p17.w17/candidate_01_submission.txt |
| fuse_move_ifs | 31 | 71 | 6.896 | 4.963 | 2.988 | 67.961 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/fuse_move_ifs/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p5.w5/candidate_01_submission.txt |
| tsvc_2_s1244 | 34 | 68 | 6.350 | 4.629 | 1.000 | 83.755 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s1244/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p14.w14/candidate_01_submission.txt |
| versioned_distance_update | 36 | 60 | 6.059 | 3.775 | 1.000 | 41.323 | gpuv2-llr40-qwen38-hip | sources/gpuv2-llr40-qwen38-hip/versioned_distance_update/626645.626645.gpuv2-llr40-qwen38-hip.n0.p37.w37/candidate_01_submission.txt |
| ext_break_capture | 26 | 42 | 5.746 | 4.629 | 1.000 | 54.059 | gpuv2-llr40-oss120b-hip | sources/gpuv2-llr40-oss120b-hip/ext_break_capture/627167.627167.gpuv2-llr40-oss120b-hip.n0.p2.w2/candidate_01_submission.txt |
| wf_triangular | 29 | 41 | 5.389 | 7.101 | 1.000 | 28.032 | llr40v11-qwen38-fortran-skills | sources/llr40v11-qwen38-fortran-skills/wf_triangular/625544.625544.llr40v11-qwen38-fortran-skills.n0.p39.w39/candidate_01_submission.txt |
| tsvc_2_s152 | 34 | 85 | 5.113 | 3.574 | 2.548 | 54.059 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s152/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p15.w15/candidate_01_submission.txt |
| tsvc_2_s252 | 34 | 77 | 4.598 | 4.088 | 1.000 | 65.310 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s252/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p21.w21/candidate_01_submission.txt |
| scan_affine_decay | 28 | 61 | 4.389 | 5.537 | 1.000 | 10.572 | llr40v11-qwen38-c | sources/llr40v11-qwen38-c/scan_affine_decay/625532.625532.llr40v11-qwen38-c.n0.p8.w8/candidate_02_submission.txt |
| tsvc_2_s231 | 30 | 62 | 4.011 | 2.250 | 1.000 | 96.274 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_s231/621016.621016.llr40v10-qwen38-c.n0.p18.w18/candidate_02_submission.txt |
| tsvc_2_s3112 | 23 | 39 | 3.306 | 1.072 | 1.000 | 112.889 | gpuv2-llr40-oss120b-hip-skills | sources/gpuv2-llr40-oss120b-hip-skills/tsvc_2_s3112/627168.627168.gpuv2-llr40-oss120b-hip-skills.n0.p28.w28/candidate_01_submission.txt |
| ext_war_unit | 31 | 57 | 3.139 | 2.787 | 1.062 | 24.877 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/ext_war_unit/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p3.w3/candidate_01_submission.txt |
| tsvc_2_vag | 30 | 59 | 3.132 | 1.156 | 1.000 | 25.126 | llr40v10-qwen38-c | sources/llr40v10-qwen38-c/tsvc_2_vag/621383.621383.llr40v10-qwen38-c.n0.p34.w34/candidate_02_submission.txt |
| tsvc_2_s4112 | 27 | 57 | 2.992 | 1.196 | 1.000 | 22.746 | llr40v10-oss120b-c | sources/llr40v10-oss120b-c/tsvc_2_s4112/621382.621382.llr40v10-oss120b-c.n0.p33.w33/candidate_01_submission.txt |
| tsvc_2_s2710 | 36 | 77 | 2.977 | 1.604 | 1.000 | 40.509 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s2710/621022.621022.llr40v10-kimi27sglang-fortran.n0.p3.w3/candidate_06_submission.txt |
| tsvc_2_s235 | 26 | 53 | 2.922 | 2.141 | 1.000 | 32.223 | llr40v10-kimi27sglang-fortran | sources/llr40v10-kimi27sglang-fortran/tsvc_2_s235/621022.621022.llr40v10-kimi27sglang-fortran.n0.p0.w0/candidate_02_submission.txt |
| tsvc_2_s275 | 27 | 48 | 2.738 | 1.661 | 1.000 | 52.994 | llr40v10-qwen38-fortran | sources/llr40v10-qwen38-fortran/tsvc_2_s275/621385.621385.llr40v10-qwen38-fortran.n0.p24.w24/candidate_02_submission.txt |
| tsvc_2_s119 | 30 | 46 | 2.655 | 2.367 | 1.000 | 53.524 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_s119/626646.626646.gpuv2-llr40-qwen38-hip-skills.n0.p12.w12/candidate_01_submission.txt |
| wf_diff_skew | 28 | 39 | 2.642 | 3.401 | 1.000 | 6.623 | llr40v10-qwen38-c-skills | sources/llr40v10-qwen38-c-skills/wf_diff_skew/621024.621024.llr40v10-qwen38-c-skills.n0.p38.w38/candidate_02_submission.txt |
| tsvc_2_s115 | 31 | 56 | 2.406 | 2.088 | 1.000 | 9.571 | v11w2-qwen38-fortran-skills | sources/v11w2-qwen38-fortran-skills/tsvc_2_s115/627014.627014.v11w2-qwen38-fortran-skills.n0.p0.w0/candidate_02_submission.txt |
| tsvc_2_vtvtv | 32 | 77 | 1.751 | 1.010 | 1.000 | 10.364 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_vtvtv/627229.627229.gpuv2-llr40-qwen38-hip-skills.n0.p25.w25/candidate_01_submission.txt |
| tsvc_2_vpvts | 31 | 76 | 1.642 | 1.030 | 1.000 | 9.289 | gpuv2-llr40-qwen38-hip-skills | sources/gpuv2-llr40-qwen38-hip-skills/tsvc_2_vpvts/627018.627018.gpuv2-llr40-qwen38-hip-skills.n0.p35.w35/candidate_03_submission.txt |

