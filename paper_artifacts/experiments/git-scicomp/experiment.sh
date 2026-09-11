# Sourced by collect.sh and submit.sh. Run roots and the jobs that do not count.
runs=${RUNS:-${SCRATCH:?}/hpcagent-bench-runs}
roots=("$runs/git-scicomp-20260909" "$runs/git-scicomp-20260910")
excluded=()
