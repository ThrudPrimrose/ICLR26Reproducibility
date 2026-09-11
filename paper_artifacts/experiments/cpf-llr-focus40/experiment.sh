# Sourced by collect.sh and submit.sh. Run roots and the jobs that do not count.
runs=${RUNS:-${SCRATCH:?}/hpcagent-bench-runs}
roots=("$runs/cpf-llr-focus40-20260909" "$runs/cpf-llr-focus40-20260910")
# 630711 630716 630753: CPF arms served the pre-drop-in form (re-rendered 2026-09-10 00:44).
# 631231 631232 631237 631241: cancelled while the forms directory was being overwritten.
excluded=(630711 630716 630753 631231 631232 631237 631241)
