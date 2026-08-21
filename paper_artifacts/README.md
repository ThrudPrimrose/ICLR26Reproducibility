# Loop-level reasoning: skills on vs off

Reproducibility artifact. Every number and figure in the paper is derived from `data/*.csv` by
`plot.py`; `data/` itself is derived from the judge databases by `collect.py`.

```bash
./reproduce.sh          # figures/ from data/ -- needs only Python + matplotlib
```

## What was run

Each **arm** is one Slurm job: a model, a language, and the skills packet either present in the
prompt or absent. All arms draw from the same 242 kernels of the `loop_level_reasoning` track and
run 80 agents against one vLLM endpoint, with a judge that compiles, checks and times every
submission against a serial baseline. Per-arm configuration, the exact submit command and the job
id are in `experiments/<arm>/`.

| model | C | C skills | Fortran | Fortran skills |
|---|---|---|---|---|
| Qwen3-Coder-30B | 601850 | 601851 | 601852 | *incomplete* |
| gpt-oss-120b | 602070 | 602071 | 602072 | 602073 |
| Kimi-K2.7-Code | -- | -- | -- | -- |

Kimi arms are absent: the engine died repeatedly under agent load on this hardware and produced no
gradeable results. Qwen3-Coder Fortran-skills had not finished at the time of collection.

## The headline result, and the caveat that governs it

**On matched problems, no skills effect reaches significance.** `figures/matched.png`:

| pair | n | off | skills | delta | McNemar exact p |
|---|---|---|---|---|---|
| Qwen3-Coder-30B - C | 217 | 73.3% | 74.2% | +0.9 pp | 0.89 |
| gpt-oss-120b - C | 105 | 26.7% | 37.1% | +10.5 pp | 0.14 |
| gpt-oss-120b - Fortran | 84 | 26.2% | 21.4% | -4.8 pp | 0.57 |

Three denominators are reported because they disagree, and the disagreement is itself a finding:

- **`solved / 242`** (`figures/success.png`) -- yield at a fixed budget. Reads as if skills *hurt*
  gpt-oss on C (21.5% -> 19.4%).
- **`solved / attempted`** (`figures/success_attempted.png`) -- capability on problems reached.
  Reverses the sign (27.1% -> 36.1%).
- **matched subset** (`figures/matched.png`) -- the only fair comparison, scoring both arms on the
  kernels *both* reached.

The cause is coverage, not capability. The skills packet is inlined into the prompt, making the
`task` field **23,249 characters against 93** -- about 250x larger -- which costs ~33% more tokens
per call. On the same token spend, gpt-oss with skills reached **130** kernels where its pair
reached **192** (`figures/coverage.png`). Reporting only `solved / 242` would have published a
throughput artifact as a capability claim.

`figures/speedup_ecdf.png` shows performance as a distribution rather than a level: the median
speedup is 1.00x for six of seven arms while the tail reaches 12.5x, so any summary statistic of
the centre reports "no effect" and hides the result.

## Limitations

- **One run per cell.** No repeats, so no confidence intervals on any single arm; the McNemar test
  is available only because the paired comparison shares problems.
- **Underpowered.** No pair reaches p<0.05. Fill runs re-running each arm on only the kernels it
  never reached are in flight to close coverage and raise n; `make_fill_problems.py` generates them.
- Speedups are wall-clock against a serial baseline on the same node type; submissions the judge
  flagged `suspect` are excluded from the speedup figures.

## Layout

```
collect.py               judge shards  -> data/*.csv
plot.py                  data/*.csv    -> figures/*.{png,svg}
make_fill_problems.py    per-arm problem sets for the unreached kernels
reproduce.sh             figures from the committed CSVs
data/calls.csv           every judge call: route, status, tokens, speedup
data/submissions.csv     final submissions: baseline_ns, native_ns, speedup, suspect
data/summary.csv         per-arm aggregates
data/matched.csv         paired comparison on the common subset, with McNemar p
problems/                the exact problem sets, skills packet inlined
experiments/<arm>/       arm.env, submit.sh, README.md, per-arm results
```

## Raw data

The judge shards are **~3 GB** and are not in this repository. `data/*.csv` is the complete
reduction and is sufficient for every figure. To rebuild from raw:

```bash
python3 collect.py --run-root <RUN_ROOT>   # <RUN_ROOT>/<jobid>/judge/rank-*/*.db
```

`RUN_ROOT` is the `RUN_ROOT` set in each `experiments/<arm>/arm.env`.
