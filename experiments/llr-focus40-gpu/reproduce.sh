#!/usr/bin/env bash
# llr-focus40 on the GPU: HIP, Triton and C + OpenMP offload, with and without the Language Skill Packet.
#   ./reproduce.sh              data/llr-focus40-gpu.db -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/llr-focus40-gpu.db
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
# 631274-631277: Triton graded while the judge refused Python, so every row is a C submission.
# 638935 638940: still in the queue on 2026-09-15 when this snapshot was extracted; a live job's
# tasks are half-run, so the arm is read at its last finished wave.
[[ " $* " == *" --extract "* ]] && extract "$db" gpu-llr-focus40 "$runs"/gpu-llr-focus40-2026* \
    -- 631274 631275 631276 631277 638935 638940

mkdir -p tables figures
# The 40 loop-level kernels, read from the corpus manifest tag the jobs were selected by (spec E1).
roster_file=$(mktemp)
trap 'rm -f "$roster_file"' EXIT
roster --tag llr-focus40 > "$roster_file"

"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment gpu-llr-focus40 \
    --label "GPU: HIP, Triton, OpenMP Offload" --out figures/gpu.pdf --table tables/gpu.csv
"$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$db" --experiment gpu-llr-focus40 --treatment skills \
    --label "GPU: Language Skill Packet" --out figures/paired_skills.pdf --table tables/paired_skills.csv

# Intervention impact table (HPCAgent-Bench docs/DESIGN_data_collection_and_scoring.md section 10):
# Language Skill Packet vs no packet, every model and GPU language, one family.
pairs=()
for model in qwen38 oss120b kimi27sglang; do
    for lang in c-openmp hip triton; do
        pairs+=(--pair "gpu-llr-focus40-$model-$lang-skills,gpu-llr-focus40-$model-$lang")
    done
done
"$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" --family lang_skills_gpu "${pairs[@]}" \
    --roster-file "$roster_file" --out tables/impact_lang_skills_gpu_pairs.csv \
    --arms-out tables/impact_lang_skills_gpu_arms.csv --impact-out tables/impact_lang_skills_gpu.csv

# Per kernel (spec A7): each arm's speed-up and task token total, no paired ratios and no tests.
for lang in c-openmp hip triton; do
    "$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$db" --roster-file "$roster_file" \
        --arm-pattern "^gpu-llr-focus40-(?P<model>[a-z0-9]+)-$lang(?:-(?P<condition>skills))?\$" \
        --condition-order ',skills' --repeats latest --label "Language Skill Packet per Kernel, $lang" \
        --out "figures/lang_skills_${lang}_kernels.pdf" --table "tables/lang_skills_${lang}_kernels.csv"
done

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
