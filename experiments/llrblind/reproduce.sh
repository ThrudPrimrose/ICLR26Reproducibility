#!/usr/bin/env bash
# llrblind: llr-focus40 on the CPU with ONE submission and no score tool, C and Fortran, +/- skills.
#   ./reproduce.sh              data/llrblind.db -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/llrblind.db
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
[[ " $* " == *" --extract "* ]] && extract "$db" llrblind "$runs"/llrblind-2026*

"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment llrblind \
    --label "No Score Tool: Language Skill Packet" --out figures/llrblind.pdf --table tables/llrblind.csv
pairs=()
for model in oss120b qwen38 kimi27sglang; do
    a=llrblind-$model
    pairs+=(--pair "$a-c,$a-fortran" --pair "$a-c-skills,$a-fortran-skills")
    pairs+=(--pair "$a-c-skills,$a-c" --pair "$a-fortran-skills,$a-fortran")
done
mkdir -p tables
"$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" --family llrblind-within "${pairs[@]}" \
    --out tables/paired_arms.csv --arms-out tables/arms.csv
[[ " $* " == *" --record "* ]] && check --record || check
