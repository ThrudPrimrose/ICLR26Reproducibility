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
# 639060 639061 639207 639209-639221: still in the queue on 2026-09-15 when this snapshot was
# extracted. A live job's tasks are half-run, and the -clean arms among them would supersede the
# arms this snapshot reports (X9); both are read only once the wave has ended.
live="639060 639061 639207 639209 639210 639211 639212 639213 639214 639215 639216 639217 639218 639219 639220 639221"
[[ " $* " == *" --extract "* ]] && extract "$db" cpf-llr-focus40 "$runs"/cpf-llr-focus40-2026* -- 631231 $live

mkdir -p tables figures
# The 40 loop-level kernels, read from the corpus manifest tag the jobs were selected by (spec E1).
roster_file=$(mktemp)
trap 'rm -f "$roster_file"' EXIT
roster --tag llr-focus40 > "$roster_file"

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
pair --treatment cpf --label "Canonical Parallel Form Page" --out figures/paired_cpf.pdf --table tables/paired_cpf.csv
pair --treatment cpfsrc --label "Canonical Parallel Form as Source" --out figures/paired_cpfsrc.pdf \
    --table tables/paired_cpfsrc.csv

# Intervention impact tables (HPCAgent-Bench docs/DESIGN_data_collection_and_scoring.md section 10):
# one paired_arms.py invocation per table, every pair TREATMENT,CONTROL, the pair list is the family.
impact() {  # impact <name> <pair>...
    local name=$1 pairs=()
    shift
    for pair in "$@"; do pairs+=(--pair "$pair"); done
    "$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" --family "$name" "${pairs[@]}" \
        --roster-file "$roster_file" --out "tables/impact_${name}_pairs.csv" \
        --arms-out "tables/impact_${name}_arms.csv" --impact-out "tables/impact_${name}.csv"
}
p=cpf-llr-focus40
impact cpf "$p-qwen38-c-cpf,$p-qwen38-c" "$p-oss120b-c-cpf,$p-oss120b-c" \
    "$p-qwen38-c-cpfsrc,$p-qwen38-c" "$p-oss120b-c-cpfsrc,$p-oss120b-c" "$p-kimi27sglang-c-cpfsrc,$p-kimi27sglang-c"
skills=()
for model in qwen38 oss120b kimi27sglang; do
    for lang in c fortran; do skills+=("$p-$model-$lang-skills,$p-$model-$lang"); done
done
impact lang_skills_cpu "${skills[@]}"

# Per kernel (spec A7): each arm's speed-up and task token total, no paired ratios and no tests. The
# pattern is both the arm selector and the (model, condition) parser; a control matches with no
# condition group, which is how this figure spells "no packet".
kernels() {  # kernels <name> <pattern> <order> <label>
    "$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$db" --roster-file "$roster_file" \
        --arm-pattern "$2" --condition-order "$3" --repeats latest --label "$4" \
        --out "figures/${1}_kernels.pdf" --table "tables/${1}_kernels.csv"
}
kernels lang_skills_c '^cpf-llr-focus40-(?P<model>[a-z0-9]+)-c(?:-(?P<condition>skills))?$' ',skills' \
    "Language Skill Packet per Kernel, C"
kernels lang_skills_fortran '^cpf-llr-focus40-(?P<model>[a-z0-9]+)-fortran(?:-(?P<condition>skills))?$' ',skills' \
    "Language Skill Packet per Kernel, Fortran"

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
