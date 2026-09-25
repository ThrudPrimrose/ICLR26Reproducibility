#!/usr/bin/env bash
# Steps 2-3 after ./download.sh: statistics, every figure and table, then the checksum check.
#   ./run_all.sh            check the outputs against SHA256SUMS
#   ./run_all.sh --record   rewrite SHA256SUMS
. "$(dirname "$0")/lib.sh"
"$ROOT/stats.sh"
for step in fig2 fig3 fig4 fig5 fig8 tab3; do "$ROOT/$step.sh"; done
check "${1:-}"
