"""Pair each SUBMISSION row to the source blob that produced it, by (run_id, benchmark, ts).

harvest_sources.py keyed sources by (run_id, kernel) into a dict, so the LAST source row won and
the header speedup came from a different round. This walks the rows in time order instead.
"""
import pathlib, sqlite3, sys, re

RUNS = pathlib.Path("hpcagent-bench-runs")
KERNEL = sys.argv[1] if len(sys.argv) > 1 else "ext_break_capture"
GLOB = sys.argv[2] if len(sys.argv) > 2 else "*llr-focus40*"

rows = []
for root in sorted(RUNS.glob(GLOB)):
    for db in sorted(root.glob("*/judge/rank-*/hpcagent_bench*.db")):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
            if not {"submissions", "sources", "runs"} <= tabs:
                continue
            arms = {r[0]: (r[1] or "") for r in c.execute("select run_id, arm from runs")}
            srcs = {}
            for rid, ts, bench, lang, path in c.execute(
                    "select run_id, ts, benchmark, language, path from sources order by ts"):
                if str(bench).split("/")[-1] != KERNEL:
                    continue
                srcs.setdefault(rid, []).append((ts, lang, path))
            for rid, ts, bench, sp, susp in c.execute(
                    "select run_id, ts, benchmark, speedup, suspect from submissions "
                    "where speedup is not null order by ts"):
                if str(bench).split("/")[-1] != KERNEL:
                    continue
                cand = [s for s in srcs.get(rid, []) if s[0] <= ts]
                if not cand:
                    continue
                t, lang, path = cand[-1]
                rows.append((arms.get(rid, "?"), rid, ts, float(sp), int(susp or 0), lang,
                             str(db.with_name(db.stem + "_prompts") / path)))
        except sqlite3.Error:
            continue

for arm, rid, ts, sp, susp, lang, blob in sorted(rows, key=lambda r: (r[0], -r[3])):
    print(f"{arm}\t{rid}\t{ts}\t{sp:.3f}\tsuspect={susp}\t{blob}")
