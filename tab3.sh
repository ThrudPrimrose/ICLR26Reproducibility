#!/usr/bin/env bash
# Table 3: GH200 against MI300A, from the joined regrade of the final loop-level answers.
. "$(dirname "$0")/lib.sh"
require "$D/transfer.csv"
"$PY" "$STATS/plot_transfer.py" --paired-csv "$D/transfer.csv" --out "$W/gh200-transfer" --table "$T/gh200-transfer.csv"
"$PY" "$L/gh200_table.py" "$T/gh200-transfer.csv" "$T/gh200-transfer-table.tex"
