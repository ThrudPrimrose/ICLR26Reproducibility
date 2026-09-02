#!/usr/bin/env bash
# Every figure in the paper, from the committed CSVs. This is the path that works from a clone: it
# needs no cluster access and none of the judge databases.
#
#   ./reproduce.sh              aggregate each experiment's waves, then draw its figures
#   ./reproduce.sh --collect    re-read the judge databases first (cluster only, see README)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
# The scripts annotate with builtin generics (list[...], dict[...]) evaluated at import, so an
# older interpreter fails with "'type' object is not subscriptable" rather than anything readable.
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "need Python >= 3.10; set PYTHON=/path/to/python3" >&2
  exit 1
}
"$PY" -c 'import matplotlib' 2>/dev/null || "$PY" -m pip install -q -r requirements.txt

for exp in llr8 llr9; do
  if [ "${1:-}" = "--collect" ]; then
    echo "== $exp collect   judge databases -> experiments/$exp/data/<wave>/*.csv + artifacts/sources/"
    "$PY" "experiments/$exp/collect_$exp.py" ${RUN_ROOT:+--run-root "$RUN_ROOT"}
  fi
  echo "== $exp aggregate experiments/$exp/data/<wave>/ -> experiments/$exp/data/kernels.csv"
  "$PY" "experiments/$exp/aggregate_$exp.py"
  echo "== $exp figures   experiments/$exp/data/kernels.csv -> experiments/$exp/figures/"
  "$PY" "experiments/$exp/plot_${exp}_kernels.py"
  "$PY" "experiments/$exp/plot_${exp}_before_after.py"
done

if [ "${1:-}" = "--collect" ]; then
  echo "== git  collect   judge databases -> experiments/git/data/*.csv + artifacts/"
  "$PY" experiments/git/collect_git.py
fi
echo "== git  aggregate experiments/git/data/git_experiment_all.csv -> experiments/git/data/kernels.csv"
"$PY" experiments/git/aggregate_git.py
echo "== git  figures   experiments/git/data/ -> experiments/git/figures/"
"$PY" experiments/git/plot_git.py
"$PY" experiments/git/plot_git_kernels.py
