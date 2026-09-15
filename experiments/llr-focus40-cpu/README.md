# llr-focus40-cpu

## Question

Does a packet or a compiler-derived form help an agent write a faster CPU kernel?

## Conditions

No packet (control), Language Skill Packet, Canonical Parallel Form page (the CPF tool and its
page), Canonical Parallel Form as the starting source (cpfsrc), perf playbook (perf-playbook-cpu).
CPF conditions (`cpf`, `cpfsrc`) are C only; Fortran has no CPF spelling.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, GLM-5.3, driven by Claude Code. 40 `llr-focus40`
kernels, C and Fortran, one agent per kernel per arm.

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

## Results: the Language Skill Packet

Ratios are with-packet / no-packet, so above 1 means faster and more expensive. `n` is the kernels
behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg adjusted p over the
family of twelve tests; `*` marks q < 0.05.

<!--TABLE impact_lang_skills_cpu-->
| model | language | packet | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen38 | c | lang-skills | 40/40 | 1.90 | 62% | 1.21 [0.73, 1.99] | 0.650 | 1.15 [0.76, 1.72] | 0.650 | 1.12 [0.82, 1.53] |
| qwen38 | fortran | lang-skills | 40/39 | 1.60 | 48% | 1.10 [0.71, 1.72] | 0.650 | 0.80 [0.52, 1.24] | 0.650 | 0.88 [0.64, 1.18] |
| oss120b | c | lang-skills | 40/40 | 1.00 | 0% | 0.93 [0.74, 1.15] | 0.650 | 1.05 [0.85, 1.30] | 0.650 | 1.05 [0.80, 1.39] |
| oss120b | fortran | lang-skills | 40/40 | 1.00 | 0% | 0.89 [0.67, 1.16] | 0.650 | 1.15 [0.98, 1.35] | 0.445 | 1.11 [0.94, 1.32] |
| kimi27sglang | c | lang-skills | 40/40 | 1.00 | 0% | 0.82 [0.64, 1.05] | 0.445 | 1.12 [0.96, 1.30] | 0.445 | 1.13 [0.98, 1.29] |
| kimi27sglang | fortran | lang-skills | 40/40 | 1.00 | 0% | 0.95 [0.75, 1.20] | 0.650 | 0.78 [0.63, 0.97] | 0.310 | 0.75 [0.63, 0.90] |

The packet moved nothing that survives the correction. Six pairs, twelve tests, no significant
verdict on either leg; point estimates run from 0.86x to 1.21x on speed-up and 0.78x to 1.15x on
tokens.

The CPF tables in `tables/impact_cpf.csv` come from the same database and the same rules and are
described in the CPF paper rather than here.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh"                     # data/llr-focus40-cpu.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/llr-focus40-cpu/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/lang_skills.pdf`, `tables/lang_skills.csv` | geometric mean speed-up and total tokens per arm, no packet vs Language Skill Packet |
| `figures/cpfsrc.pdf`, `tables/cpfsrc.csv` | geometric mean speed-up and total tokens per arm, no packet vs Canonical Parallel Form as source (C only) |
| `figures/cpf.pdf`, `tables/cpf.csv` | geometric mean speed-up and total tokens per arm, no packet vs Canonical Parallel Form page (C only) |
| `figures/paired_skills.pdf`, `tables/paired_skills.csv` | per-kernel paired speed-up/cost ratio, skills on vs off, with significance |
| `figures/paired_cpfsrc.pdf`, `tables/paired_cpfsrc.csv` | per-kernel paired speed-up/cost ratio, cpfsrc vs no packet, with significance |
| `tables/impact_lang_skills_cpu.csv`, `tables/impact_cpf.csv` | the intervention impact tables: one row per arm, the treatment row carrying both legs, the total-token ratio, and the usage and relaunch counts behind them |
| `tables/lang_skills_<language>_kernels.csv`, `figures/lang_skills_<language>_kernels.pdf` | per kernel, each arm's speed-up and task token total, with a geomean and median summary row; no ratios and no tests |

## Data provenance

Run root `<RUNS>/cpf-llr-focus40-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. Excluded: job 631231, cancelled while
the CPF forms directory was being overwritten, and jobs 639060, 639061, 639207 and 639209-639221,
still in the queue when this snapshot was extracted.

## Caveats

Qwen3.8-27B relaunched 30-62% of its tasks; GPT-OSS-120B and Kimi-K2.7-Code relaunched none. Its
reported cost is the final attempt's, and its crashed attempts spent a further 47.1M tokens across
this campaign, recorded in `tokens_crashed` and in no ratio. The same crashes cost it answers: 75
of the campaign's (arm, kernel) answers were graded before their task's final attempt began and are
dropped by X7, 72 of them Qwen's, with a median dropped speed-up of 6.5x. Qwen's coverage in these
tables, 21 to 30 kernels of 40, is what is left after that rule.

The `-clean` arms among the excluded jobs would otherwise supersede the arms reported here (X9),
and a live job's tasks are half-run, so each arm is read at its last finished wave.

GLM-5.3 ran only the C skills arm and has no control, so it enters no pair; the arm figures still
draw it. Significance per contrast is in the `paired_*.csv` and `impact_*.csv` tables (the
`*_verdict` columns).

## Caveats

Significance per contrast is in the `paired_*.csv` tables (the `*_verdict` columns). GLM-5.3 ran
only the C skills arm; the figures reflect the rows that exist.
