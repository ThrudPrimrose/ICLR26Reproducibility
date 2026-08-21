#!/usr/bin/env bash
# Figures from the CSVs in data/. This is the path that works from a clone: it needs no cluster
# access and no 3 GB of judge shards.
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
"$PY" plot.py
echo
echo "figures/ regenerated from data/. To rebuild data/ from the raw judge shards instead:"
echo "  $PY collect.py --run-root <RUN_ROOT>   # see README.md, 'Raw data'"
