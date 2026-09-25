#!/usr/bin/env bash
# Snapshot HPCAgent-Bench data (fresh runs, DBs, frozen observations, artifact) from the cluster into a
# dated folder and zip it with SHA256SUMS: a frozen, shareable copy, unlike the incremental tools/pull.sh.
# Read-only on the cluster (see pull.sh); nothing is deleted locally, so a re-run resumes.
#   tools/collect.sh                  -> <dir of $MIRROR>/hb-data-<date>/ and .zip
#   DEST=/path/snap tools/collect.sh  -> /path/snap/ and /path/snap.zip
# Env: tools/cluster.env (see cluster.env.example); TRIES, RETRY_SLEEP tune the retries.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/cluster.sh"
case ${1:-} in
    -h | --help) usage; exit 0 ;;
    '') ;;
    *) echo "$(basename "$0"): unknown argument '$1' (see --help)" >&2; exit 2 ;;
esac
cluster_init
DEST=${DEST:-$(dirname "$MIRROR")/hb-data-$(date +%Y%m%d)}
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)

DBS=(--include='*.db' --include='*.db-wal' --include='*.db-shm' --include='*.sqlite')
pull hpcagent-bench-runs runs --prune-empty-dirs \
    --exclude='home/' --exclude='.cache/' --exclude='vllm/' --exclude='rocprof_out*/' --exclude='dacecache/' \
    --include='*/' --include='.env' --include='*.resolved' --include='prompt.txt' --include='observations/**' \
    --include='*.json' --include='*.jsonl' --include='*.csv' "${DBS[@]}" --exclude='*'
pull hpcagent-bench/experiments hb/experiments --exclude='__pycache__/' --exclude='*.err' --exclude='*.out'
pull audit-20260918/frozen-observations-0919 frozen-observations
pull ICLR26Reproducibility ICLR26Reproducibility --exclude='__pycache__/'
pull canon-sweep-0923 canon-sweep --prune-empty-dirs --exclude='dacecache/' --exclude='dbg*/' \
    --include='*/' --include='*.csv' --include='*.db' --exclude='*'

fetch "echo hpcagent-bench \$(git -C '$CLUSTER_SCRATCH/hpcagent-bench' rev-parse HEAD); echo frozen figures built with \
\$(cat '$CLUSTER_SCRATCH/ICLR26Reproducibility/HPCAGENT_BENCH_COMMIT' 2>/dev/null || echo unknown)" "$DEST/COMMITS"
# The snapshot carries the scripts that made it.
cp "$0" "$(dirname "$0")/cluster.sh" "$DEST/"
checksum_dir "$DEST"
rm -f "$DEST.zip"
(cd "$(dirname "$DEST")" && zip -qr "$DEST.zip" "$(basename "$DEST")")
du -sh "$DEST" "$DEST.zip"
echo "DONE $DEST.zip"
