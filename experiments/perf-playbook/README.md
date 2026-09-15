# perf-playbook

## Question

Does a performance-engineering playbook packet, a page of profiling-and-optimization method with
the profiler tools to follow it, change what an agent delivers?

## Conditions

No packet against the playbook packet, on three campaigns:

| campaign | arms | roster | repeats |
|---|---|---|---|
| scicomp-focus40 | `scicomp-perf-playbook-<model>-plain` vs `-perf-playbook-cpu` | 40 scientific-computing kernels | 3 agents per kernel, kernel = their median |
| llr-focus40 CPU, C | `cpf-llr-focus40-<model>-c` vs `-c-perf-playbook-cpu` | 40 loop-level kernels | one task per kernel, a rerun replacing what it repeats |
| llr-focus40 GPU, HIP | `gpu-llr-focus40-<model>-hip` vs `-hip-perf-playbook-amd` | same 40 kernels | same |

The CPU packet names the `gcc` report flags and the AMD packet the ROCm profiler. They are the same
method on different tools, so each is compared against its own control and never against the other.

## Models and kernels

Qwen3.8-27B and GPT-OSS-120B, driven by Claude Code. Kimi-K2.7-Code ran the scicomp campaign but
enters no pair here: its playbook arm reached 2 of the 40 kernels in the waves that had finished
when this snapshot was extracted.

## Scoring

Every number obeys `docs/DESIGN_data_collection_and_scoring.md` in HPCAgent-Bench. Three rules
decide what the tables say, and a reader needs all three.

**The last agent ran the task from nothing to its end.** A crashed agent is relaunched from an
empty context and an empty workspace (T5), so nothing an earlier attempt built survived into what
was graded. A task is therefore scored and priced as its FINAL attempt alone: judge rows stamped
before that attempt started are dropped (X7), the token total is the final attempt's (T2), and what
the earlier attempts spent is reported beside it as `tokens_crashed`, never added in. Cancelled
tasks, where the job ended under a working agent, are dropped whole (X8); this snapshot has none.
Because the rule bites hardest where agents crash most, every table carries `attempts_per_task` and
the share of tasks that relaunched next to its token ratio.

**Tokens are counted once (fold 2).** `output` is every token the model generated, reasoning
included, as both serving engines report it; the client's thinking estimate is never added on top.
The count comes from the first source that has it: the server's per-request count, else the
server's episode count on the `result` record, else the model's own tokenizer over the transcript
(`output_source = retokenized`, which is 2-4% low by construction, T11). Input is counted when it
first enters the context, so a cached prompt is not billed again on every turn.

**A failed episode scores 1x and still costs its tokens.** The population of every speed-up
aggregate is the kernels the arm was SERVED, not the kernels it verified. An episode that never
delivered a verified answer enters at 1.0, which is exactly what it left standing, and its tokens
enter the totals and the medians, because the agent was given the kernel and spent its budget.
Scoring only what an arm verified reports it on the subset it happened to succeed on, which flatters
the arms that failed most. Every table carries `n_solved` beside `n`, so how much of an arm's number
is delivery and how much is 1.0 is always visible, and the per-kernel figures draw a
never-delivered kernel with an x.

**Summary statistics.** Per arm, the speed-up is the geometric mean over the kernels it verified,
with a log-t interval (A1), and the token cost is the median task total with a percentile bootstrap
interval (A2). Per pair, both legs are paired by kernel: the geometric mean ratio with its log-t
interval and paired t test (P3), and beside the token leg the ratio of TOTAL tokens over the shared
kernels with a paired bootstrap interval (9999 resamples, seed 0). The two token numbers answer
different questions and the table carries both: the geomean is the typical kernel, the total is the
budget. Benjamini-Hochberg at q = 0.05 runs over one family, and the family is every leg of every
pair of one invocation (M1).

**Which interval, and why.** Hoefler and Belli (SC15), as encoded in
`hpcagent_bench/stats/rules.py`: Rule 4 says a ratio is summarized by the geometric mean and the
costs it was taken over stay in the table, so the artifact tables keep `baseline_ns`, `native_ns`
and `tokens` beside every ratio. Rule 5 requires an interval for nondeterministic data, so a geomean
is never reported bare. Rule 7 compares through intervals rather than through point estimates. The
geomean interval is taken in log space (log-t from `summary.geomean_ci`); token costs use the
nonparametric bootstrap of the median. Every figure and table states `n`, the kernels or pairs
behind the number. Rule 12 allows a connecting line only where it means something, so the segment
joining a control mark to its packet mark is a PAIR LINK and the legend says so.

Benjamini-Hochberg runs over one campaign at a time here, because the three campaigns share neither
roster nor repeat policy and correcting them together would pool tests that never shared a
population.

## Results

Speed-up ratio and token ratio are playbook / no packet, so above 1 means faster and more
expensive. `n` is the kernels behind the speed-up leg and behind the token leg. `q` is the
Benjamini-Hochberg adjusted p of that campaign's family; `*` marks q < 0.05.

Scientific computing:

<!--TABLE playbook_scicomp-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen38 | -- | 14/40 | 2.82 | 95% | 1.04 [0.83, 1.29] | 0.943 | 1.12 [0.94, 1.34] | 0.361 | 1.13 [0.95, 1.36] |
| oss120b | -- | 19/40 | 1.00 | 0% | 0.99 [0.77, 1.27] | 0.943 | 0.92 [0.84, 1.01] | 0.338 | 0.93 [0.87, 1.00] |

Loop-level C:

<!--TABLE playbook_llr_cpu-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen38 | c | 14/40 | 1.80 | 55% | 1.34 [0.91, 1.98] | 0.378 | 1.29 [0.88, 1.89] | 0.378 | 1.21 [0.91, 1.60] |
| oss120b | c | 38/40 | 1.00 | 0% | 1.01 [0.81, 1.26] | 0.924 | 1.07 [0.90, 1.26] | 0.598 | 1.02 [0.84, 1.26] |

Loop-level HIP:

<!--TABLE playbook_llr_gpu-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen38 | hip | 13/40 | 1.88 | 65% | 1.69 [0.65, 4.40] | 0.536 | 0.88 [0.60, 1.28] | 0.657 | 0.94 [0.71, 1.25] |
| oss120b | hip | 40/40 | 1.00 | 0% | 1.27 [0.83, 1.93] | 0.536 | 0.99 [0.86, 1.13] | 0.830 | 0.96 [0.82, 1.11] |

Nothing the packet did reaches significance on any campaign, on either leg. The speed-up ratios sit
between 0.99x and 1.69x with intervals that all contain 1, and the widest of them, Qwen3.8-27B on
HIP, rests on 13 shared kernels.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/perf-playbook/reproduce.sh"                     # data/ + the two llr-focus40 databases -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/perf-playbook/reproduce.sh" --extract           # CSCS only: rebuild data/scicomp-perf-playbook.db from $RUNS first
"$ARTIFACT_ROOT/experiments/perf-playbook/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/perf-playbook/reproduce.sh
```

Run `../llr-focus40-cpu/reproduce.sh --extract` and `../llr-focus40-gpu/reproduce.sh --extract`
first: the two loop-level campaigns are read from those experiments' databases.

## Outputs

| file | how to read it |
|---|---|
| `tables/playbook_<campaign>.csv` | the intervention impact table: one row per arm, the treatment row carrying both paired legs, the total-token ratio, and the usage and relaunch counts behind them |
| `tables/playbook_<campaign>_arms.csv` | per arm: geomean speed-up and median task tokens with intervals, solved and served counts, attempts, calls, crashed spend |
| `tables/playbook_<campaign>_pairs.csv`, `figures/playbook_<campaign>_forest.pdf` | the two legs of each pair with intervals, wins and losses, the family's corrected verdicts |
| `tables/playbook_<campaign>_kernels.csv`, `figures/playbook_<campaign>_kernels.pdf` | per kernel, each arm's speed-up and task token total, with a geomean and median summary row; no ratios and no tests |

## Data provenance

Run root `<RUNS>/scicomp-perf-playbook-2026*` for the scicomp campaign; `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. The two loop-level campaigns are read
from `../llr-focus40-cpu/data/` and `../llr-focus40-gpu/data/`, which their own `reproduce.sh
--extract` builds, so this experiment keeps no second copy of either database.

Excluded: jobs 637044 and 638964, still in the queue on 2026-09-15 when this snapshot was
extracted. The llr-focus40 exclusions are stated in those two experiments' own READMEs.

## Caveats

Both scicomp playbook arms cover 38 of the 40 kernels: `nussinov` and `quatrex_rgf` were never
dispatched to them, while both plain arms cover all 40. That family is therefore run with
`--include-incomplete`; the legs are paired, so the two missing kernels lower `n` and enter no
estimate.

The scicomp campaign grades most kernels against C -O3 + autopar and a few against numpy or a
vendored library, and one kernel, `cholesky`, has rows of both kinds. A speed-up divided by two
references is not one quantity, so the reported speed-up leg is restricted to the autopar-graded
kernels (`--baseline c-autopar`). Tokens carry no denominator and keep every kernel.

Qwen3.8-27B relaunched 95% of its scicomp tasks and 55-65% of its loop-level ones; GPT-OSS-120B
relaunched none. The Qwen token cost in these tables is the final attempt's, and the crashed
attempts spent 33.0M (playbook) and 36.5M (plain) more on scicomp, recorded in `tokens_crashed` and
in no ratio. A Qwen row compares two arms that each lost most of their first attempts; it is not a
statement about what one uninterrupted agent costs.

A handful of tasks carry no driver token record and were re-folded from their transcripts here; ten
of them across the databases report a non-positive total and drop out of the token population (R7).
