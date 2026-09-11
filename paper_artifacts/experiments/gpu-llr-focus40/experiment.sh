# Sourced by collect.sh and submit.sh. Run roots and the jobs that do not count.
runs=${RUNS:-${SCRATCH:?}/hpcagent-bench-runs}
roots=("$runs/gpu-llr-focus40-20260910")
# 631274-631277: Triton graded with JUDGE_INPUT_MODE=source, which refuses Python; every row is C.
excluded=(631274 631275 631276 631277)
