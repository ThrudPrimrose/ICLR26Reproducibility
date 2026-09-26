#!/usr/bin/env bash
# Build the anonymized Zenodo package: hpcagent-bench/ (the benchmark at <ref>) beside reproducibility/
# (this repository at HEAD, its data/ as data.tar.zst with DATA_SHA256SUMS), every file, member and path
# rewritten with the terms of the untracked .anonymize-terms.txt and every commit id of both repositories
# pseudonymized. The DaCe dependency URL of the benchmark's pyproject.toml stays as it is, so the
# package still installs. Then three independent scans must all be clean: the terms (anonymize.py
# --check), the identifier patterns of the untracked .identifier-patterns.txt plus every name of the
# git histories (scan_identifiers.py), and a byte-level grep of the same. The one command:
#   tools/make_zenodo.sh <hpcagent-bench checkout> [ref=HEAD] [out.zip]
# EXTRA_PEOPLE="<repo> ..." adds more git histories (e.g. the DaCe and paper checkouts) to the name scans.
set -euo pipefail
ulimit -c 0
root=$(cd "$(dirname "$0")/.." && pwd)
bench=$(cd "${1:?usage: tools/make_zenodo.sh <hpcagent-bench checkout> [ref] [out.zip]}" && pwd)
ref=${2:-HEAD}
out=$(realpath -m "${3:-$root/work/hpcagent-bench-anonymous.zip}")
terms=$root/.anonymize-terms.txt
[[ -f $terms ]] || { echo "missing $terms" >&2; exit 2; }
# Reproducible archives: one fixed time for every file, no owner names.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1767225600} TZ=UTC
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
t0=$SECONDS
lap() { echo "[$((SECONDS - t0))s] $*"; }
mkdir -p "$stage/src/hpcagent-bench" "$stage/src/reproducibility"
git -C "$bench" archive "$ref" | tar -x -C "$stage/src/hpcagent-bench"
git -C "$root" archive HEAD | tar -x -C "$stage/src/reproducibility"
cp -a "$root/data" "$stage/src/reproducibility/data"
find "$stage/src" \( -name '*.db-shm' -o -name '*.db-wal' \) -delete
keep=$(grep -oE 'git\+https://github\.com/[^"]*/dace\.git@[0-9a-f]{40}' "$stage/src/hpcagent-bench/pyproject.toml" || true)
anon=(python3 "$root/tools/anonymize.py" --terms "$terms" --commits "$bench" --commits "$root" ${keep:+--keep "$keep"})
lap "staged $(git -C "$bench" rev-parse --short "$ref") and $(git -C "$root" rev-parse --short HEAD)"
if ! (cd "$stage/src" && "${anon[@]}" --out "$stage/out" hpcagent-bench reproducibility >"$stage/anonymize.log"); then
    grep -v '^ok' "$stage/anonymize.log" >&2
    exit 1
fi
lap "anonymized"
repro=$stage/out/reproducibility
(cd "$repro/data" && find . -type f | LC_ALL=C sort | xargs sha256sum) >"$repro/DATA_SHA256SUMS"
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
    --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime -C "$repro/data" -c . | zstd -q -19 -o "$repro/data.tar.zst"
mkdir -p "$stage/bytes"
mv "$repro/data" "$stage/bytes/data"
lap "data archive packed"
"${anon[@]}" --check "$stage/out/hpcagent-bench" "$repro" >"$stage/check.log" || { grep -v '^ok' "$stage/check.log" >&2; exit 1; }
lap "scan 1 (terms, every format and archive member): $(tail -1 "$stage/check.log")"
people=(--people "$bench" --people "$root")
for repo in ${EXTRA_PEOPLE:-}; do people+=(--people "$repo"); done
scan=(python3 "$root/tools/scan_identifiers.py" --patterns "$root/.identifier-patterns.txt" "${people[@]}" ${keep:+--keep "$keep"})
"${scan[@]}" "$stage/out" >"$stage/scan2.log" || { cat "$stage/scan2.log" >&2; exit 1; }
lap "scan 2 (identifier patterns and git-history names): $(tail -1 "$stage/scan2.log")"
cp -a "$stage/out/." "$stage/bytes/"
mv "$stage/bytes/data" "$stage/bytes/reproducibility/data"
"${scan[@]}" --bytes "$stage/bytes" >"$stage/scan3.log" || { cat "$stage/scan3.log" >&2; exit 1; }
lap "scan 3 (byte-level grep, data unpacked): $(tail -1 "$stage/scan3.log")"
find "$stage/out" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +
rm -f "$out"
(cd "$stage/out" && find hpcagent-bench reproducibility -type f | LC_ALL=C sort | zip -X -D -q "$out" -@)
lap "zipped"
echo "zenodo package: $out ($(du -h "$out" | cut -f1)), $(find "$stage/out" -type f | wc -l) files, sha256 $(sha256sum "$out" | cut -c1-64)"
