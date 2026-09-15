# llrblind

## Question

Does removing the score tool and allowing only one submission change the outcome, versus the
normal iterate-and-score loop?

## Conditions

CPU, `llr-focus40` roster, C and Fortran, Language Skill Packet on and off. One submission per
episode (`AGENT_SINGLE_SUBMISSION=1`); no score route at all (`AGENT_SCORE_TOOL=0` withholds the
tool, `HPCAGENT_BENCH_SERVICE_SCORE_ENABLED=0` closes the HTTP route an agent could otherwise
call itself).

## Models and kernels

GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code, driven by Claude Code. Same 40 `llr-focus40` kernels.

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

One task per kernel per arm. A kernel an arm ran more than once is charged its LATEST task, never
the best of them and never the sum (R4, R6).

## Results: the Language Skill Packet with no score tool

Ratios are with-packet / no-packet, so above 1 means faster and more expensive. `n` is the kernels
behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg adjusted p over this
experiment's family; `*` marks q < 0.05.

<!--TABLE impact_llrblind_skills-->
| model | language | packet | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| oss120b | c | lang-skills | 40/40 | 1.00 | 0% | 0.95 [0.67, 1.33] | 0.786 | 1.06 [0.92, 1.22] | 0.712 | 1.01 [0.87, 1.16] |
| oss120b | fortran | lang-skills | 40/40 | 1.00 | 0% | 0.86 [0.58, 1.27] | 0.712 | 1.05 [0.92, 1.19] | 0.712 | 1.03 [0.91, 1.17] |
| qwen38 | c | lang-skills | 40/40 | 1.05 | 5% | 1.28 [0.84, 1.95] | 0.593 | 0.98 [0.83, 1.15] | 0.786 | 0.98 [0.80, 1.17] |
| qwen38 | fortran | lang-skills | 40/40 | 1.02 | 2% | 1.37 [0.91, 2.05] | 0.504 | 0.79 [0.69, 0.90] | 0.010* | 0.83 [0.73, 0.93] |
| kimi27sglang | c | lang-skills | 40/40 | 1.00 | 0% | 1.46 [1.07, 1.99] | 0.103 | 0.97 [0.79, 1.20] | 0.786 | 0.98 [0.85, 1.12] |
| kimi27sglang | fortran | lang-skills | 40/40 | 1.00 | 0% | 1.31 [0.86, 2.01] | 0.593 | 0.95 [0.78, 1.16] | 0.786 | 1.01 [0.85, 1.19] |

What the score tool itself buys is the sibling experiment `llrblind-vs-scored`, which pairs these
arms against the scored arms of `llr-focus40-cpu` kernel by kernel.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llrblind/reproduce.sh"                     # data/llrblind.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llrblind/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/llrblind/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/llrblind/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/llrblind.pdf`, `tables/llrblind.csv` | geometric mean speed-up and total tokens per arm, no score tool, skills on/off |
| `tables/paired_arms.

C against Fortran is no longer reported as a pair: a paired comparison holds model and language
fixed (spec P1), and a speed-up over Numba in C is not the same quantity as one in Fortran.csv`, `figures/paired_skills.pdf` | per-kernel paired ratios, skills against no packet, per model and language, with intervals and corrected verdicts |
| `tables/impact_llrblind_skills.csv`, `tables/arms.csv` | the intervention impact table and the per-arm table behind it: geomean speed-up, median task tokens, attempts, calls, crashed spend |
| `tables/llrblind_<language>_kernels.csv`, `figures/llrblind_<language>_kernels.pdf` | per kernel, each arm's speed-up and task token total, with a geomean and median summary row; no ratios and no tests |

## Data provenance

Run root `<RUNS>/llrblind-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. No jobs excluded.

## Caveats

The recorded speed-up is the judge's significance-gated minimum gain: a verified submission that
is slower or within noise records at exactly 1.0. `llrblind-qwen38-c-skills` ran under a tighter
global token cap than the other arms and most of its agents never reached submit; treat its
numbers as answer quality only, not as a coverage or submission-rate comparison against the other
arms.
