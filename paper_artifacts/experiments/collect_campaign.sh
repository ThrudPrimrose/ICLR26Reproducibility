#!/usr/bin/env bash
# collect_campaign.sh <out.csv> <run-root>... [-- <excluded job id>...]
# Judge databases of every job under the run roots -> one observations CSV. Cluster only.
# Excluded jobs measured a superseded treatment; they are listed in each experiment's collect.sh
# rather than applied by hand, so the exclusion is part of the record.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$1; shift
roots=(); excluded=()
while (($#)); do [[ $1 == -- ]] && { shift; excluded=("$@"); break; }; roots+=("$1"); shift; done
args=()
for root in "${roots[@]}"; do
    for job in "$root"/*/; do
        id=$(basename "$job"); [[ $id =~ ^[0-9]+$ ]] || continue
        [[ " ${excluded[*]:-} " == *" $id "* ]] && continue
        args+=(--runs "${job%/}")
    done
done
tmp=$(mktemp -d)
"${PYTHON:-python3}" "$here/llr40/extract_llr40.py" "${args[@]}" \
    --benchmarks "${OPTARENA:-${SCRATCH:?}/optarena}/hpcagent_bench/benchmarks" --out "$tmp" --no-sources
mkdir -p "$(dirname "$out")"; mv "$tmp/llr40_observations.csv" "$out"; rm -rf "$tmp"
echo "$(($(wc -l < "$out") - 1)) observations -> $out"
