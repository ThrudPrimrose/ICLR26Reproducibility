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
            # TREATMENT,CONTROL: withdrawing the score tool is the intervention, so the blind arm
            # is the treated side and the scored arm the control. Every ratio in this experiment is
            # therefore no-score / scored, and a value below 1 means the agent did worse without it.
            pairs+=(--pair "llrblind-$model-$lang$suffix,cpf-llr-focus40-$model-$lang$suffix")
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

# The same two square slope panels every other intervention is drawn with: X = Control | Treated,
# left Y = geomean speed-up, right Y = median tokens per task. The pairs CSV supplies both the
# pairing and the corrected verdicts, so the figure cannot star a pair this table calls not
# significant.
"$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$scored" "$blind" \
    --pairs-csv tables/paired_arms.csv --intervention no-score --label "No Score Tool" \
    --control-label "Score Tool and Unlimited Submissions" \
    --out figures/blind_vs_scored.pdf --table tables/blind_vs_scored_points.csv

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
