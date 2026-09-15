# Sourced by every experiments/<name>/reproduce.sh. Sets PY and PYTHONPATH, and defines extract
# (cluster only) and check (checksums). Use the latest hpcagent-bench main; HPCAGENT_BENCH_COMMIT
# records the commit the committed figures were made with.
set -euo pipefail

here_common=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${HPCAGENT_BENCH:?set HPCAGENT_BENCH to a hpcagent-bench checkout (latest main)}"
PY=${PYTHON:-python3}
export PYTHONPATH="$HPCAGENT_BENCH:$HPCAGENT_BENCH/hpcagent_bench/numpy_translators/src"
export MPLBACKEND=Agg

made_with=$(cat "$here_common/../HPCAGENT_BENCH_COMMIT")
using=$(git -C "$HPCAGENT_BENCH" rev-parse HEAD)
[[ "$using" == "$made_with"* ]] || echo "note: figures were made with hpcagent-bench $made_with; using $using" >&2

# Re-timed rows (scripts/regrade.py) for submissions graded before the timing-reduction stamp; extract drops such
# a submission when it has no re-timed row, so every speed-up in a table comes from one rule.
regrades=${REGRADES:-${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}/regrade-20260915/full/regrade-*.db}

# extract <out.db> <arm-prefix> <run-root>... [-- <excluded job id>...]
# Every judge database under the run roots, minus the excluded jobs -> one observations table.
extract() {
    local out=$1 prefix=$2 roots=() excluded=() args=() root job id
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
    local tmp
    tmp=$(mktemp -d)
    "$PY" "$HPCAGENT_BENCH/reproducibility/llr40/extract_llr40.py" "${args[@]}" --arm-prefix "$prefix" \
        --benchmarks "$HPCAGENT_BENCH/hpcagent_bench/benchmarks" --out "$tmp" --no-sources --db "$out" \
        --regrades "$regrades"
    rm -rf "$tmp"
}

# roster --tag <manifest tag> | --kernels-file <launcher kernels file>
# The kernel set a table is read over (spec E1), printed one per line. DECLARED, never derived from
# the rows: an experiment that lost a kernel everywhere would otherwise report full coverage over
# the survivors, and one stray kernel served by a single wave would empty a family.
roster() {
    "$PY" "$HPCAGENT_BENCH/reproducibility/llr40/focus_roster.py" \
        --benchmarks "$HPCAGENT_BENCH/hpcagent_bench/benchmarks" "$@"
}

# check: every figure and table matches SHA256SUMS. check --record rewrites SHA256SUMS instead.
check() {
    local dirs=()
    for dir in figures tables; do [[ -d "$dir" ]] && dirs+=("$dir"); done
    if [[ "${1:-}" == --record ]]; then
        find "${dirs[@]}" -type f | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
        echo "recorded $(wc -l < SHA256SUMS) checksums"
    else
        sha256sum --quiet -c SHA256SUMS && echo "OK: every figure and table matches SHA256SUMS"
    fi
}
