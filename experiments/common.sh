# Sourced by every experiments/<name>/reproduce.sh. Sets PY and PYTHONPATH, and defines require_data
# (the committed database must exist), extract (cluster only) and check (checksums). Use the latest
# hpcagent-bench main; HPCAGENT_BENCH_COMMIT records the commit the committed figures were made at.
# This file is kept IDENTICAL in ICLR26Reproducibility and mpr-artifacts.
set -euo pipefail

here_common=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${HPCAGENT_BENCH:?set HPCAGENT_BENCH to a hpcagent-bench checkout (latest main)}"
PY=${PYTHON:-python3}
export PYTHONPATH="$HPCAGENT_BENCH:$HPCAGENT_BENCH/hpcagent_bench/numpy_translators/src"
export MPLBACKEND=Agg
# dace hashes iteration order into generated code, and the plotting path shares the rule.
export PYTHONHASHSEED=0

made_with=$(cat "$here_common/../HPCAGENT_BENCH_COMMIT")
using=$(git -C "$HPCAGENT_BENCH" rev-parse HEAD)
[[ "$using" == "$made_with"* ]] || echo "note: figures were made at $made_with; you are at $using" >&2

# require_data <db> [why]: the committed database, or one line saying why it is absent and exit 2.
require_data() {
    [[ -f "$1" ]] && return 0
    local why=${2:-"the re-grade of rows graded before 2026-09-13 is still running"}
    local where="see Status in the top-level README and STATUS.md; rebuild it on the cluster with --extract"
    echo "$(basename "$PWD"): $1 is not committed yet -- $why, $where" >&2
    exit 2
}

# extract <out.db> <arm-prefix> <run-root>... [-- <excluded job id>...]
# Every judge database under the run roots, minus the excluded jobs -> one observations table.
# REGRADES: glob of the regrade databases experiments/regrade.sbatch wrote. Without it
# extract_llr40.py refuses rows graded before the current timing reduction; see the top-level
# README, "Rebuild the data".
extract() {
    local out=$1 prefix=$2 roots=() excluded=() args=() regrades=() root job id
    shift 2
    while (($#)); do [[ $1 == -- ]] && { shift; excluded=("$@"); break; }; roots+=("$1"); shift; done
    for root in "${roots[@]}"; do
        for job in "$root"/*/; do
            id=$(basename "$job")
            [[ $id =~ ^[0-9]+$ ]] || continue
            [[ " ${excluded[*]:-} " == *" $id "* ]] && continue
            args+=(--runs "${job%/}")
        done
    done
    [[ -n "${REGRADES:-}" ]] && regrades=(--regrades "$REGRADES")
    local tmp
    tmp=$(mktemp -d)
    "$PY" "$HPCAGENT_BENCH/reproducibility/llr40/extract_llr40.py" "${args[@]}" "${regrades[@]}" \
        --arm-prefix "$prefix" --benchmarks "$HPCAGENT_BENCH/hpcagent_bench/benchmarks" \
        --out "$tmp" --no-sources --db "$out"
    rm -rf "$tmp"
}

# check: every figure and table matches SHA256SUMS. check --record rewrites SHA256SUMS instead.
check() {
    local dirs=()
    for dir in figures tables; do [[ -d "$dir" ]] && dirs+=("$dir"); done
    if ((${#dirs[@]} == 0)); then
        echo "nothing to record"
        exit 0
    fi
    if [[ "${1:-}" == --record ]]; then
        find "${dirs[@]}" -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
        echo "recorded $(wc -l < SHA256SUMS) checksums"
    else
        sha256sum --quiet -c SHA256SUMS && echo "OK: every figure and table matches SHA256SUMS"
    fi
}
