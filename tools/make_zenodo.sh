#!/usr/bin/env bash
# Build the anonymized Zenodo package: hpcagent-bench/ (the benchmark at <ref>) beside reproducibility/
# (this repository at HEAD with its data/), every file and path rewritten with the terms of the
# untracked .anonymize-terms.txt. Fails on any leftover term; re-check later with check_anonymous.sh.
#   tools/make_zenodo.sh <hpcagent-bench checkout> [ref=HEAD] [out.zip]
set -euo pipefail
ulimit -c 0
root=$(cd "$(dirname "$0")/.." && pwd)
bench=$(cd "${1:?usage: tools/make_zenodo.sh <hpcagent-bench checkout> [ref] [out.zip]}" && pwd)
ref=${2:-HEAD}
out=$(realpath -m "${3:-$root/work/hpcagent-bench-anonymous.zip}")
terms=$root/.anonymize-terms.txt
[[ -f $terms ]] || { echo "missing $terms" >&2; exit 2; }
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/src/hpcagent-bench" "$stage/src/reproducibility"
git -C "$bench" archive "$ref" | tar -x -C "$stage/src/hpcagent-bench"
git -C "$root" archive HEAD | tar -x -C "$stage/src/reproducibility"
cp -a "$root/data" "$stage/src/reproducibility/data"
find "$stage/src" \( -name '*.db-shm' -o -name '*.db-wal' \) -delete
if ! (cd "$stage/src" && python3 "$root/tools/anonymize.py" --terms "$terms" --out "$stage/out" \
    hpcagent-bench reproducibility >"$stage/anonymize.log"); then
    grep -v '^ok' "$stage/anonymize.log" >&2
    exit 1
fi
(cd "$stage/out/reproducibility/data" && find . -type f | LC_ALL=C sort | xargs sha256sum) \
    >"$stage/out/reproducibility/DATA_SHA256SUMS"
rm -f "$out"
(cd "$stage/out" && zip -qr "$out" hpcagent-bench reproducibility)
echo "zenodo package: $out ($(du -h "$out" | cut -f1)), $(find "$stage/out" -type f | wc -l) files, bench $(git -C "$bench" rev-parse --short "$ref")"
