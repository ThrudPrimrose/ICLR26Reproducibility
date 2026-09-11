#!/usr/bin/env bash
# data/observations.csv -> figures/ and data/<figure>.csv.
set -euo pipefail
cd "$(dirname "$0")"
opt=${OPTARENA:-${SCRATCH:?}/optarena}
export PYTHONPATH="$opt:$opt/hpcagent_bench/numpy_translators/src"
py=${PYTHON:-python3}; obs=data/observations.csv
"$py" "$opt/scripts/plot_arm_summary.py" "$obs" --experiment gpu-llr-focus40 --label "GPU: HIP, Triton, OpenMP Offload" \
    --out figures/gpu.pdf --table data/gpu.csv
"$py" "$opt/scripts/plot_score_change.py" "$obs" --experiment gpu-llr-focus40 --treatment skills \
    --label "GPU: Language Skill Packet" --out figures/paired_skills.pdf --table data/paired_skills.csv
