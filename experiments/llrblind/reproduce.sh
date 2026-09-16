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

mkdir -p tables figures
# The 40 loop-level kernels, read from the corpus manifest tag the jobs were selected by (spec E1).
roster_file=$(mktemp)
trap 'rm -f "$roster_file"' EXIT
roster --tag llr-focus40 > "$roster_file"

"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment llrblind \
    --label "No Score Tool: Language Skill Packet" --out figures/llrblind.pdf --table tables/llrblind.csv

# The Language Skill Packet with no score tool, one pair per model and language; the pair list is
# the family (spec M1). C against Fortran is not a pair: a pair holds model and language fixed
# (P1), and a speed-up on one language is not the same quantity as on the other.
pairs=()
for model in oss120b qwen38 kimi27sglang; do
    for lang in c fortran; do pairs+=(--pair "llrblind-$model-$lang-skills,llrblind-$model-$lang"); done
done
"$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" --family llrblind-skills "${pairs[@]}" \
    --roster-file "$roster_file" --out tables/paired_arms.csv --arms-out tables/arms.csv \
    --impact-out tables/impact_llrblind_skills.csv

# The same two square slope panels every intervention is drawn with; the pairs CSV supplies both
# the pairing and the corrected verdicts.
"$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$db" \
    --pairs-csv tables/paired_arms.csv --intervention lang-skills \
    --label "Language Skill Packet, No Score Tool" --out figures/paired_skills.pdf \
    --table tables/paired_skills_points.csv

# Per kernel (spec A7): each arm's speed-up and task token total, no paired ratios and no tests.
for lang in c fortran; do
    "$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$db" --roster-file "$roster_file" \
        --arm-pattern "^llrblind-(?P<model>[a-z0-9]+)-$lang(?:-(?P<condition>skills))?\$" \
        --condition-order ',skills' --repeats latest --label "No Score Tool per Kernel, $lang" \
        --out "figures/llrblind_${lang}_kernels.pdf" --table "tables/llrblind_${lang}_kernels.csv"
done

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
