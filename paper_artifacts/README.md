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

## Performance

`figures/fastp.png` is the headline performance figure: fast_p, the fraction of matched problems
that are both correct and at least p times faster. At p=1 it is the success rate, so one curve
carries correctness and speed together and cannot be gamed by returning a correct but unoptimised
kernel. `figures/speedup_ecdf.png` shows the raw distribution, faceted by language.

| pair | fast_1 | fast_2 | fast_4 |
|---|---|---|---|
| Qwen3-Coder-30B - C | 73.3 -> 74.2 | 8.8 -> 6.0 | 5.1 -> 2.3 |
| gpt-oss-120b - C | 26.7 -> 37.1 | 6.7 -> 8.6 | 3.8 -> 1.9 |
| gpt-oss-120b - Fortran | 26.2 -> 21.4 | 3.6 -> 2.4 | 0.0 -> 1.2 |

Skills raise fast_1 on gpt-oss C but lower fast_4 in two of three pairs: more answers are correct,
fewer are aggressively optimised. At fast_4 that is 11 problems against 5 out of 217, so the
direction is suggestive and the magnitude is not.

**Caveat on the speedup column.** 512 of 801 submissions report exactly 1.000000, and the
`baseline_ns` / `native_ns` recorded beside them do not reproduce it (one row is 1.29x by its own
timings, another 0.75x, both stored as 1.00). Only 90 distinct speedup values occur across 801
submissions, on the grid 1/0.99, 1/0.98, 1/0.97, so the ratio is rounded to two decimals before
inversion. The likely explanation is that `/submit` re-times on a second seed and `speedup` comes
from that measurement rather than from the stored ns columns, but this has not been confirmed
against the harness. If the value is floored at 1.0, the mass at x=1 in both performance figures is
a property of the metric rather than of the generated code. Confirm before publishing a speedup
claim.

## Does the skills packet cause compile failures? No -- it prevents them

| pair | build_error, off -> on |
|---|---|
| Qwen3-Coder-30B - C | 3.6% -> 2.7% |
| gpt-oss-120b - C | 7.6% -> 7.2% |
| gpt-oss-120b - Fortran | 22.4% -> 17.6% |

Build errors fall in every pair, most on Fortran. What rises instead is timeouts and wrong
answers: on Qwen C, timeouts go 3.4% -> 5.2% and incorrect 7.5% -> 9.6%. That is consistent with
prompt length rather than bad advice -- the packet is 23,249 characters against 93, so agents spend
more of a fixed budget reading and less iterating. The skills teach OpenMP (13 `#pragma omp` and 16
`parallel for` mentions in the C packet; `openacc` and, for Fortran, `doconcurrent-fortran` are
inlined too), and the effect shows up as a build-error reduction rather than as more speedup.

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
