#!/usr/bin/env bash
# canon: deterministic compiler baselines over the 40 llr-focus40 kernels, no agents (sweep 631260).
#   ./reproduce.sh              data/canon.db -> figures/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the sweep's per-rank CSVs (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
db=data/canon.db
sweep=${CANON_SWEEP:-/capstor/scratch/cscs/ybudanaz/x86_64/canon-llr40-20260910}
[[ " $* " == *" --extract "* ]] && mkdir -p data \
    && "$PY" "$HPCAGENT_BENCH/scripts/collect_canon.py" --run-dir "$sweep" --db "$db" --label canon-llr40-631260

require_data "$db"
"$PY" "$HPCAGENT_BENCH/scripts/plot_canon_speedup.py" --db "$db" --out figures
[[ " $* " == *" --record "* ]] && check --record || check
