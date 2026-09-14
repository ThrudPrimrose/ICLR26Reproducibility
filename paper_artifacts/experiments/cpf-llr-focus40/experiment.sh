# Sourced by collect.sh and submit.sh. Run roots and the jobs that do not count.
runs=${RUNS:-${SCRATCH:?}/hpcagent-bench-runs}
roots=("$runs/cpf-llr-focus40-20260909" "$runs/cpf-llr-focus40-20260910")
# 631231: cancelled while the forms directory was being overwritten.
excluded=(631231)
