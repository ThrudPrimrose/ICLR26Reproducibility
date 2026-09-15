#!/usr/bin/env bash
# llrblind-vs-scored: cpf-llr-focus40 (score tool, no submission limit) against llrblind (one
# submission, no score tool), on the same 40 llr-focus40 CPU kernels, matched by
# (model, language, skills). Reads its two sibling experiments' own databases directly -- no
# local data/, no copy of either.
#   ./reproduce.sh              tables/ + figures/ from ../llr-focus40-cpu/data and ../llrblind/data
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
#
# The two sibling databases come from their own reproduce.sh --extract; run those first if
# ../llr-focus40-cpu/data or ../llrblind/data is missing.
cd "$(dirname "$0")"
. ../common.sh
scored=../llr-focus40-cpu/data/llr-focus40-cpu.db
blind=../llrblind/data/llrblind.db

pairs=()
for model in oss120b qwen38 kimi27sglang; do
    for lang in c fortran; do
        for suffix in "" "-skills"; do
            pairs+=(--pair "cpf-llr-focus40-$model-$lang$suffix,llrblind-$model-$lang$suffix")
        done
    done
done

mkdir -p tables figures
# The 40 loop-level kernels, read from the corpus manifest tag the jobs were selected by (spec E1).
roster_file=$(mktemp)
trap 'rm -f "$roster_file"' EXIT
roster --tag llr-focus40 > "$roster_file"

"$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$scored" --observations "$blind" \
    --family blind-vs-scored "${pairs[@]}" --roster-file "$roster_file" --out tables/paired_arms.csv \
    --arms-out tables/arms.csv --impact-out tables/impact_blind_vs_scored.csv

"$PY" "$HPCAGENT_BENCH/scripts/plot_paired_arms.py" tables/paired_arms.csv \
    --label "What the Score Tool and Unlimited Submissions Buy" --ratio-label "Scored / Blind" \
    --out figures/blind_vs_scored.pdf

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
