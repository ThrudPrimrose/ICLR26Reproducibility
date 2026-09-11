#!/usr/bin/env bash
# data/observations.csv -> figures/ and data/<figure>.csv.
set -euo pipefail
cd "$(dirname "$0")"
opt=${OPTARENA:-${SCRATCH:?}/optarena}
export PYTHONPATH="$opt:$opt/hpcagent_bench/numpy_translators/src"
py=${PYTHON:-python3}; obs=data/observations.csv; m='(qwen38|oss120b|kimi27sglang)'
arm() { "$py" "$opt/scripts/plot_arm_summary.py" "$obs" --experiment cpf-llr-focus40 "$@"; }
pair() { "$py" "$opt/scripts/plot_score_change.py" "$obs" --experiment cpf-llr-focus40 "$@"; }
arm --arms "cpf-llr-focus40-$m-(c|fortran)(-skills)?" --label "Language Skill Packet" \
    --out figures/lang_skills.pdf --table data/lang_skills.csv
arm --arms "cpf-llr-focus40-$m-c(-cpfsrc)?" --label "Canonical Parallel Form" \
    --out figures/cpfsrc.pdf --table data/cpfsrc.csv
arm --arms "cpf-llr-focus40-$m-c(-cpf|-cpfsrc)?" --label "Canonical Parallel Form: Page vs Source" \
    --out figures/cpf_page_vs_source.pdf --table data/cpf_page_vs_source.csv
pair --treatment skills --label "Language Skill Packet" --out figures/paired_skills.pdf --table data/paired_skills.csv
pair --treatment cpfsrc --label "Canonical Parallel Form" --out figures/paired_cpfsrc.pdf --table data/paired_cpfsrc.csv
