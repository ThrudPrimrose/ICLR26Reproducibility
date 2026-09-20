#!/usr/bin/env bash
# llr-focus40 on the GPU: HIP, Triton and C + OpenMP offload, with and without the Language Skill Packet.
#   ./reproduce.sh              data/llr-gpu.db -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/llr-gpu.db
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
# 631274-631277: Triton graded while the judge refused Python, so every row is a C submission.
[[ " $* " == *" --extract "* ]] && extract "$db" gpu-llr-focus40 "$runs"/gpu-llr-focus40-2026* \
    -- 631274 631275 631276 631277

require_data "$db"
"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment gpu-llr-focus40 \
    --label "GPU: HIP, Triton, OpenMP Offload" --out figures/gpu.pdf --table tables/gpu.csv
"$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$db" --experiment gpu-llr-focus40 --treatment skills \
    --label "GPU: Language Skill Packet" --out figures/paired_skills.pdf --table tables/paired_skills.csv
[[ " $* " == *" --record "* ]] && check --record || check
