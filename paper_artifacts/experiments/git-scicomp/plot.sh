#!/usr/bin/env bash
# data/observations.csv -> figures/ and data/<figure>.csv.
set -euo pipefail
cd "$(dirname "$0")"
opt=${OPTARENA:-${SCRATCH:?}/optarena}
export PYTHONPATH="$opt:$opt/hpcagent_bench/numpy_translators/src"
"${PYTHON:-python3}" "$opt/scripts/plot_arm_summary.py" data/observations.csv --experiment git-scicomp \
    --label "Repository Framing, Scientific Computing" --out figures/git.pdf --table data/git.csv
