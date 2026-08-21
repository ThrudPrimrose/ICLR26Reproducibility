#!/usr/bin/env bash
# Figures from the CSVs in data/. This is the path that works from a clone: it needs no cluster
# access and no 3 GB of judge shards.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 plot.py
echo
echo "figures/ regenerated from data/. To rebuild data/ from the raw judge shards instead:"
echo "  python3 collect.py --run-root <RUN_ROOT>   # see README.md, 'Raw data'"
