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

require_data "$db"
"$PY" "$HPCAGENT_BENCH/scripts/plot_arm_summary.py" "$db" --experiment git-scicomp \
    --label "Repository vs Kernel, Scientific Computing" --out figures/git.pdf --table tables/git.csv
[[ " $* " == *" --record "* ]] && check --record || check
