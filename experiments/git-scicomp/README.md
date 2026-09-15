# git-scicomp

## Question

Does handing the agent the repository a kernel lives in beat handing it the bare kernel?

## Conditions

Bare kernel vs whole repository. Three agent episodes per kernel.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B and Kimi-K2.7-Code, driven by Claude Code, each with both a kernel arm
and a repository arm. 10 scientific-computing kernels. Speedups are over C -O3 + autopar, not
Numba: Numba runs 16-165x slower than C on these kernels and does not finish the XL preset, which
would credit the agent for Numba's own failure.

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

**Summary statistics.** Per arm, the speed-up is the geometric mean over the kernels it verified,
with a log-t interval (A1), and the token cost is the median task total with a percentile bootstrap
interval (A2). Per pair, both legs are paired by kernel: the geometric mean ratio with its log-t
interval and paired t test (P3), and beside the token leg the ratio of TOTAL tokens over the shared
kernels with a paired bootstrap interval (9999 resamples, seed 0). The two token numbers answer
different questions and the table carries both: the geomean is the typical kernel, the total is the
budget. Benjamini-Hochberg at q = 0.05 runs over one family, and the family is every leg of every
pair of one invocation (M1).

Three agents run each kernel by design, so a kernel's speed-up is the median over its tasks and its
token total is the median of theirs, reported with their minimum and maximum (R5).

## Results

Ratios are repository / kernel, so below 1 means the repository arm was slower or cheaper. `n` is
the kernels behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg
adjusted p over the family of six tests; `*` marks q < 0.05.

<!--TABLE impact_git-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen38 | -- | 7/10 | 2.45 | 85% | 0.72 [0.34, 1.54] | 0.396 | 0.86 [0.63, 1.17] | 0.396 | 0.85 [0.69, 1.03] |
| oss120b | -- | 7/10 | 1.00 | 0% | 0.75 [0.58, 0.96] | 0.093 | 0.93 [0.66, 1.30] | 0.616 | 0.91 [0.67, 1.28] |
| kimi27sglang | -- | 8/10 | 1.00 | 0% | 0.62 [0.32, 1.21] | 0.272 | 0.64 [0.52, 0.77] | 0.003* | 0.68 [0.55, 0.84] |

Handing over the repository did not make any model faster. All three point estimates are below 1,
and no speed-up leg survives the correction. Kimi-K2.7-Code spent less with the repository (geomean
0.64x, total 0.68x, q = 0.003), which is the one effect in the family.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh"                     # data/git-scicomp.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/git-scicomp/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/git.pdf`, `tables/git.csv` | geometric mean speed-up and median task tokens per arm, repository framing vs kernel framing |
| `tables/paired.csv`, `figures/paired_forest.pdf` | per model, repository arm against kernel arm, PAIRED by kernel: geomean speed-up ratio and token ratio with log-t intervals and paired t tests, the total-token ratio with its paired bootstrap interval, Benjamini-Hochberg over the whole family; `underpowered` below 6 paired kernels |
| `tables/impact_git.csv`, `tables/arms.csv` | the intervention impact table and the per-arm table behind it: solved and served counts, attempts, calls, crashed spend |
| `tables/kernel_comparison.csv`, `figures/kernel_comparison.pdf` | per kernel, each arm's speed-up and task token total, with a geomean and median summary row; no ratios and no tests |

## Data provenance

Run root `<RUNS>/git-scicomp-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. No jobs excluded.

## Caveats

Every arm has a row for all 10 roster kernels, so the completeness check drops nothing; what varies
is how many of them each arm verified (7 to 10).

Qwen3.8-27B relaunched 76-85% of its tasks; the other two models relaunched none. Its reported cost
is the final attempt's, and the crashed attempts spent a further 9.5M tokens on the kernel arm and
8.4M on the repository arm, recorded in `tokens_crashed` and in no ratio. Its two arms also ran
unequal numbers of tasks (45 against 33) because relaunches and extra waves fell unevenly, so the
medians behind its row rest on different amounts of evidence.

Rows graded before 2026-09-13 used an older timing rule and were re-timed (`scripts/regrade.py`); a
submission with no re-timed row is dropped, so every speed-up here comes from one rule.
