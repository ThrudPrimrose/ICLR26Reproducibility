#!/usr/bin/env bash
# llr-focus40 on the CPU: no packet, Language Skill Packet, CPF page, CPF as source, perf playbook.
#   ./reproduce.sh              data/llr-focus40-cpu.db -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/llr-focus40-cpu.db
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
# 631231: cancelled while the CPF forms directory was being overwritten.
[[ " $* " == *" --extract "* ]] && extract "$db" cpf-llr-focus40 "$runs"/cpf-llr-focus40-2026* -- 631231

m='(qwen38|oss120b|kimi27sglang|glm53)'
arm() { "$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment cpf-llr-focus40 "$@"; }
pair() { "$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$db" --experiment cpf-llr-focus40 "$@"; }
arm --arms "cpf-llr-focus40-$m-(c|fortran)(-skills)?" --label "Language Skill Packet" \
    --out figures/lang_skills.pdf --table tables/lang_skills.csv
arm --arms "cpf-llr-focus40-$m-c(-cpfsrc)?" --label "Canonical Parallel Form as Source" \
    --out figures/cpfsrc.pdf --table tables/cpfsrc.csv
arm --arms "cpf-llr-focus40-$m-c(-cpf)?" --label "Canonical Parallel Form Page" \
    --out figures/cpf.pdf --table tables/cpf.csv
pair --treatment skills --label "Language Skill Packet" --out figures/paired_skills.pdf --table tables/paired_skills.csv
pair --treatment cpfsrc --label "Canonical Parallel Form as Source" --out figures/paired_cpfsrc.pdf \
    --table tables/paired_cpfsrc.csv
[[ " $* " == *" --record "* ]] && check --record || check
