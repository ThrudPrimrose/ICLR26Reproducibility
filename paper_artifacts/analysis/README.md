# analysis/

Session analyses that are derived from the judge databases but are not part of the paper's
`data/ -> plot.py -> figures/` pipeline. Each document states the jobs it was collected from and
the caveats that govern its numbers.

| document | question it answers |
|---|---|
| `SKILLS_ABLATION_LLR6V10.md` | Does the v10 skills packet change speedup? (no: pooled p = 1.000, n = 102) |
| `FORTRAN_ABI_DEFECT.md` | Why does Fortran fail ~40% of judged calls? (one ABI defect, 10 of 11 dead kernels) |

## Files

- `extract_arms.py` -- judge SQLite DBs -> `arms.json`. Login-node safe: `sqlite3` only, no numpy
  and no `hpcagent_bench` import (neither is installed for python3.11 there).
- `analyze_arms.py` -- `arms.json` -> the paired sign tests and per-arm quality tables.
- `build_payload.py` -- `arms.json` -> `payload.json`, the shape the HTML figure consumes.
- `figures/skills_ablation.html` -- the published figure, self-contained. `*.head.html`,
  `*.body.html` and `*.js` are its sources; the JSON is inlined into the built file.
- `verify/probe_ref.py` -- emits the NumpyToX reference for a kernel in C and Fortran and prints
  the loop nest. Needs the container plus `PYTHONPATH=<repo>:<repo>/hpcagent_bench/numpy_translators/src`
  (the emitter spawns a subprocess, so `sys.path` alone does not reach it).
- `verify/focus40.txt` -- the 40 `llr-focus40` kernel keys, comma-joined.
- `verify/stub-*.json` -- stub-agent correctness runs over the emitted references.

## Standing caveats for anything computed here

1. Recorded `speedup` is quantized onto a two-decimal reciprocal grid; means are unusable and
   anything above ~5x is resolution, not signal.
2. `speedup` disagrees with the stored `baseline_ns` / `native_ns`.
3. `submissions` contains only correctness-passing rows, so a kernel that never passes is absent
   from every statistic rather than scoring zero.

Full statements of all three are in `SKILLS_ABLATION_LLR6V10.md`.
