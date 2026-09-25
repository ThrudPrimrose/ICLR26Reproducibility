#!/usr/bin/env bash
# Build the OWED final-grade worklist on the cluster: every episode's final submission that has no
# mw4x5-final(-v2) row yet. Runs on the login node over ssh; writes only to the node-local /tmp
# (never to scratch), then copies the worklist and a per-arm count back. Submits nothing.
#   tools/owed_worklist.sh    -> $OWED_DIR/owed.jsonl, $OWED_DIR/summary.txt
# Env: tools/cluster.env; OWED_DIR (default <dir of $MIRROR>/owed), REMOTE_PY (the cluster's
# hpcagent-bench python), EXPERIMENTS (space-separated).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/cluster.sh"
case ${1:-} in
    -h | --help) usage; exit 0 ;;
    '') ;;
    *) echo "$(basename "$0"): unknown argument '$1' (see --help)" >&2; exit 2 ;;
esac
cluster_init
REMOTE_PY=${REMOTE_PY:-$CLUSTER_SCRATCH/venv-hpcagent-bench-314/bin/python}
EXPERIMENTS=${EXPERIMENTS:-llr-focus40 llr-focus40-blind git-scicomp scicomp-focus40 harness20}
OWED_DIR=${OWED_DIR:-$(dirname "$MIRROR")/owed}
mkdir -p "$OWED_DIR"

# Not retried: a failure here is the job's, not the connection's; re-run by hand.
ssh "${SSH_OPTS[@]}" "$CLUSTER_HOST" "$(printf 'S=%q PY=%q EXPERIMENTS=%q bash -s' \
    "$CLUSTER_SCRATCH" "$REMOTE_PY" "$EXPERIMENTS")" <<'REMOTE'
set -euo pipefail
T=/tmp/owed-$USER; mkdir -p "$T"
cd "$S/hpcagent-bench"
export PYTHONHASHSEED=0 PYTHONPATH=$PWD:$PWD/hpcagent_bench/numpy_translators/src
# Final-grade regrade waves, oldest first (later waves win ties).
regrades=()
while IFS= read -r wave; do regrades+=(--regrades "$wave/*"); done \
    < <(find experiments -maxdepth 1 -type d -name 'mwd-final-regrades-v*' | sort -V)
for exp in $EXPERIMENTS; do
    "$PY" -m hpcagent_bench.dataset --experiment "$exp" --out "$T/$exp.db" \
        --runs-root "$S/hpcagent-bench-runs" --frozen-observations "$S/audit-20260918/frozen-observations-0919" \
        "${regrades[@]}" 2>&1 | grep -E '^(final grade|live rows|wrote)' >&2 \
        || { echo "dataset $exp failed (rerun without the grep to see why)" >&2; exit 1; }
    "$PY" -m hpcagent_bench.harness.regrade worklist --observations "$T/$exp.db" --env-dir experiments \
        --scope all --final-only --out "$T/$exp.all.jsonl" 2>&1 | tail -2 >&2 \
        || { echo "worklist $exp failed" >&2; exit 1; }
done
"$PY" - "$T" $EXPERIMENTS <<'EOF'
import collections, json, sqlite3, sys
from hpcagent_bench.harness import timing
from hpcagent_bench.observations_extract import run_path
tmp, exps = sys.argv[1], sys.argv[2:]
out, count = open(f"{tmp}/owed.jsonl", "w"), collections.Counter()
for exp in exps:
    conn = sqlite3.connect(f"{tmp}/{exp}.db")
    done = {(run_path(db), str(run), str(bench), int(ts)) for db, run, bench, ts in conn.execute(
        "SELECT db, run_id, benchmark, ts_ms FROM observations WHERE timing_reduction IN (?, ?)",
        timing.FINAL_GRADE_REDUCTIONS)}
    # Only the ANSWER is plotted: the latest submission per (arm, kernel). git-scicomp keeps every
    # repeat, since its three designed repeats are reduced by their median.
    latest: dict[tuple[str, ...], tuple[int, str, dict]] = {}
    for line in open(f"{tmp}/{exp}.all.jsonl"):
        item = json.loads(line)
        key = (item.get("arm", "?").removesuffix("-clean"), str(item["benchmark"]))
        if exp == "git-scicomp":
            key += (str(item["run_id"]),)
        if key not in latest or int(item["ts_ms"]) > latest[key][0]:
            latest[key] = (int(item["ts_ms"]), line, item)
    for _, line, item in latest.values():
        if (run_path(item["db"]), str(item["run_id"]), str(item["benchmark"]), int(item["ts_ms"])) in done:
            continue
        out.write(line)
        count[(exp, item.get("arm", "?"), item.get("language", "?"))] += 1
out.close()
with open(f"{tmp}/summary.txt", "w") as f:
    for (exp, arm, lang), n in sorted(count.items()):
        f.write(f"{n:5d}  {exp:18s} {arm} [{lang}]\n")
    f.write(f"{sum(count.values()):5d}  TOTAL owed final submissions\n")
EOF
REMOTE
for f in owed.jsonl summary.txt; do fetch "cat /tmp/owed-\$USER/$f" "$OWED_DIR/$f"; done
cat "$OWED_DIR/summary.txt"
