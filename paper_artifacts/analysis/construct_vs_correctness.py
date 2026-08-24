"""Does using an unchecked-assertion construct predict incorrect calls, within an arm?

Joins each agent to its OWN judge rows via run_id (identity_env writes
"<arm>.n<node>.p<problem_index>.w<worker_index>", and the worker directory is named
problem-<problem_index>-worker-<worker_index>), so this is a within-arm comparison
between agents, not a between-arm one.
"""
import collections
import glob
import json
import pathlib
import re
import sqlite3

ARMS = {"qwen30b/C no-skills": 605458, "qwen30b/C skills": 605459,
        "oss120b/C no-skills": 605460, "oss120b/C skills": 605461}
PATTERNS = {
    "omp simd": re.compile(r"omp\s+(parallel\s+for\s+)?simd"),
    "collapse": re.compile(r"collapse\s*\("),
    "reduction": re.compile(r"reduction\s*\("),
}
WORKER_RE = re.compile(r"problem-(\d+)-worker-(\d+)$")


def calls_by_run(job: int) -> dict:
    out = collections.defaultdict(lambda: [0, 0])
    for db in glob.glob(f"/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs/{job}/judge/rank-*/*.db"):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for run_id, status in con.execute("select run_id, status from calls"):
            rec = out[run_id or ""]
            rec[0] += 1
            if status == "incorrect":
                rec[1] += 1
        con.close()
    return out


def written(log: pathlib.Path) -> str:
    parts = []
    for line in log.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for b in ((rec.get("message") or {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_use" and (b.get("name") or "") in (
                    "Write", "Edit", "MultiEdit"):
                inp = b.get("input") or {}
                for k in ("content", "new_string"):
                    if isinstance(inp.get(k), str):
                        parts.append(inp[k])
                for ed in (inp.get("edits") or []):
                    if isinstance(ed, dict) and isinstance(ed.get("new_string"), str):
                        parts.append(ed["new_string"])
    return "\n".join(parts)


for label, job in ARMS.items():
    runs = calls_by_run(job)
    # run_id suffix -> (calls, incorrect); match on the p<..>.w<..> tail
    tail = {}
    for rid, rec in runs.items():
        m = re.search(r"\.p(\d+)\.w(\d+)$", rid)
        if m:
            tail[(m.group(1), m.group(2))] = rec
    groups = {k: [[0, 0], [0, 0]] for k in PATTERNS}  # [without],[with] -> [calls, incorrect]
    for log in pathlib.Path(f"/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs/{job}/agents").glob(
            "node-*/problem-*-worker-*/claude.log"):
        m = WORKER_RE.search(log.parent.name)
        if not m:
            continue
        rec = tail.get((m.group(1), m.group(2)))
        if not rec or rec[0] == 0:
            continue
        text = written(log)
        for key, rx in PATTERNS.items():
            slot = groups[key][1 if rx.search(text) else 0]
            slot[0] += rec[0]
            slot[1] += rec[1]
    print(f"=== {label} ===")
    for key, (without, with_) in groups.items():
        f = lambda r: (100 * r[1] / r[0]) if r[0] else float("nan")
        print(f"  {key:10s} without: {f(without):5.1f}% incorrect (n={without[0]:5d})   "
              f"with: {f(with_):5.1f}% incorrect (n={with_[0]:5d})")
    print()
