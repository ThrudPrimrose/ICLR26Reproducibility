#!/usr/bin/env bash
# git-scicomp: the agent gets the kernel alone, or the repository the kernel lives in.
#   ./reproduce.sh              data/git-scicomp.db -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/git-scicomp.db
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
[[ " $* " == *" --extract "* ]] && extract "$db" git-scicomp "$runs"/git-scicomp-2026*

mkdir -p tables figures
# The 10 scientific-computing kernels, read from the file the jobs were launched with (spec E1).
roster_file=$(mktemp)
trap 'rm -f "$roster_file"' EXIT
roster --kernels-file "$HPCAGENT_BENCH/experiments/kernels-git-scicomp.txt" > "$roster_file"

"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment git-scicomp \
    --label "Repository vs Kernel, Scientific Computing" --out figures/git.pdf --table tables/git.csv

# Repository vs kernel, paired by kernel, one pair per model: the geomean speed-up ratio and the
# geomean token ratio with their log-t intervals and paired t tests (spec P3), the total-token ratio
# with a paired bootstrap beside the second, Benjamini-Hochberg over the whole family.
pairs=()
for model in qwen38 oss120b kimi27sglang; do
    pairs+=(--pair "git-scicomp-$model-repo,git-scicomp-$model-kernel")
done
# git-scicomp runs 3 agents per kernel by design: kernels are the median over them (spec R5).
"$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" "${pairs[@]}" --repeats median \
    --roster-file "$roster_file" --family git-scicomp-repo-vs-kernel --out tables/paired.csv \
    --arms-out tables/arms.csv --impact-out tables/impact_git.csv

# The same two square slope panels every other intervention is drawn with: X = Control | Treated,
# left Y = geomean speed-up, right Y = median tokens per task. The pairs CSV supplies both the
# pairing and the corrected verdicts.
"$PY" "$HPCAGENT_BENCH/scripts/plot_score_change.py" "$db" \
    --pairs-csv tables/paired.csv --intervention repo --label "Whole Repository" \
    --control-label "Bare Kernel" --out figures/paired_forest.pdf \
    --table tables/paired_forest_points.csv

# Condition comes from the arm's own kernel|repo suffix, not the language/packet columns: a
# git-scicomp episode does not always stamp them, and the arm name is the one thing every row of
# an arm agrees on. No --canon-db: git-scicomp has no deterministic compiler reference.
"$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$db" --roster-file "$roster_file" \
    --arm-pattern '^git-scicomp-(?P<model>[a-z0-9]+)-(?P<condition>kernel|repo)$' \
    --condition-order kernel,repo --repeats median \
    --label "Repository vs Kernel, per Kernel" --out figures/kernel_comparison.pdf \
    --table tables/kernel_comparison.csv

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
