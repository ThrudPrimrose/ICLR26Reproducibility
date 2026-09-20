#!/usr/bin/env bash
# git-scicomp: the agent gets the kernel alone, or the repository the kernel lives in.
#   ./reproduce.sh              data/ -> tables/ + figures/, then check checksums
#   ./reproduce.sh --extract    first rebuild data/ from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
#
# The campaign ran each kernel three times, so every reduction here takes the MEDIAN of the
# repeats (--repeats median), and the speed-up denominator is auto-parallelised C, not Numba.
cd "$(dirname "$0")"
. ../common.sh

db=data/git-scicomp.db
family=repo-vs-kernel

if [[ " $* " == *" --extract "* ]]; then
    # One command: judge databases + the frozen rows of jobs whose directories are gone, scoped to
    # this experiment by hpcagent_bench.campaigns and written as both .db and .csv.
    # REGRADES re-times the submissions graded before the mwd-v2 timing rule; without it the
    # extractor refuses them rather than mixing two rules in one table.
    regrades=()
    [[ -n ${REGRADES:-} ]] && regrades=(--regrades "$REGRADES")
    "$PY" -m hpcagent_bench.dataset --experiment git-scicomp \
        --out "$db" --csv data/git-scicomp.csv "${regrades[@]}" \
        ${RUNS:+--runs-root "$RUNS"}
fi

require_data "$db"

# The pairs: one per model, the repository arm against the bare-kernel arm of the same model.
# Benjamini-Hochberg is declared over this whole family, both legs of all three pairs.
pairs=()
for model in qwen38 oss120b kimi27sglang; do
    pairs+=(--pair "git-scicomp-$model-repo,git-scicomp-$model-kernel")
done
"$PY" "$HPCAGENT_BENCH/statistics/paired_arms.py" --observations "$db" "${pairs[@]}" \
    --family "gitscicomp-$family" --repeats median --cost-model billed --baseline c-autopar \
    --out "tables/${family}_billed.csv" --arms-out "tables/${family}_billed_arms.csv"

# The figure reads those pairs: their arm_a,arm_b rows ARE the pairs it draws and their corrected
# verdicts ARE its stars, so the panel and the table cannot disagree.
"$PY" "$HPCAGENT_BENCH/statistics/plot_score_change.py" "$db" \
    --pairs-csv "tables/${family}_billed.csv" --intervention repo \
    --control-label "Kernel Formulation" --repeats median --cost-model billed \
    --mark-size 55 --legend-pt 6.0 \
    --out "figures/${family}.pdf" --table "tables/${family}_verdicts.csv"

[[ " $* " == *" --record "* ]] && check --record || check
