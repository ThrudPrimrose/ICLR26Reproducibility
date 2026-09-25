#!/bin/bash
# Resubmit regrade_debug.sbatch until all 40 ranks have a DB task row for every worklist item (max 8 jobs).
cd "$(dirname "${BASH_SOURCE[0]}")"
R=results_debug/c
complete() {
    python3 - "$R" <<'PY'
import sqlite3, sys, pathlib
root = pathlib.Path(sys.argv[1]); left = 0; ranks = 0
for r in range(40):
    d = root / f"rank-{r}"; wl = d / "worklist.jsonl"; db = d / "regrade-cells-0.db"
    if not wl.exists(): left += 1; continue
    ranks += 1
    n_want = sum(1 for l in wl.read_text().splitlines() if l.strip())
    n_have = sqlite3.connect(db).execute("select count(*) from regrade_tasks").fetchone()[0] if db.exists() else 0
    left += max(0, n_want - n_have)
print(f"ranks with worklist {ranks}/40, items left {left}", file=sys.stderr)
sys.exit(0 if ranks == 40 and left == 0 else 1)
PY
}
for i in 1 2 3 4 5 6 7 8; do
    complete && { echo "ALL DONE after $((i-1)) jobs $(date +%H:%M)"; exit 0; }
    j=$(sbatch --parsable -A g34 regrade_debug.sbatch) || { echo "sbatch failed"; exit 1; }
    echo "job $i: $j submitted $(date +%H:%M)"
    while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 30; done
    echo "job $i: $j $(sacct -j "$j" -X -n -o State,Elapsed | xargs) $(date +%H:%M)"
done
complete && echo "ALL DONE" || echo "STOPPED after 8 jobs, not complete"
