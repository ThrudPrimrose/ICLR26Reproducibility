#!/usr/bin/env bash
# Figure 8 (appendix): per-kernel speedup of the loop-level answers the pair tables use.
. "$(dirname "$0")/lib.sh"
require "$W/llr-focus40.db"
"$PY" "$L/plot_cheating_per_kernel.py" --db "$W/llr-focus40.db" --out "$F/cheating_per_kernel.pdf"
