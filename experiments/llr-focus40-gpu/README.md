# llr-focus40-gpu

## Question

Does the same packet help on the GPU, and how do the three delivery languages compare?

## Conditions

HIP, Triton (Python delivery), C + OpenMP offload, each with and without the Language Skill
Packet. Speedups are over the same Numba baseline as the CPU track.

## Models and kernels

Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, driven by Claude Code. Same 40 `llr-focus40` kernels,
on the MI300A GPU.

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

One task per kernel per arm. A kernel an arm ran more than once is charged its LATEST task, never
the best of them and never the sum (R4, R6).

## Results: the Language Skill Packet

Ratios are with-packet / no-packet, so above 1 means faster and more expensive. `n` is the kernels
behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg adjusted p over the
family of eighteen tests; `*` marks q < 0.05.

<!--TABLE impact_lang_skills_gpu-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| qwen38 | c | 24/40 | 1.25 | 22% | 1.45 [0.68, 3.08] | 0.597 | 1.18 [0.81, 1.72] | 0.597 | 1.02 [0.73, 1.46] |
| qwen38 | hip | 21/40 | 2.00 | 62% | 1.63 [0.93, 2.88] | 0.343 | 0.91 [0.66, 1.24] | 0.638 | 0.94 [0.70, 1.24] |
| qwen38 | triton | 13/39 | 1.55 | 50% | 0.78 [0.37, 1.63] | 0.627 | 1.04 [0.71, 1.55] | 0.822 | 1.07 [0.78, 1.48] |
| oss120b | c | 39/40 | 1.00 | 0% | 0.64 [0.42, 0.97] | 0.289 | 0.94 [0.79, 1.12] | 0.627 | 0.93 [0.80, 1.08] |
| oss120b | hip | 40/40 | 1.00 | 0% | 1.38 [0.94, 2.03] | 0.343 | 1.07 [0.93, 1.23] | 0.597 | 1.04 [0.90, 1.20] |
| oss120b | triton | 0/40 | 1.00 | 0% | 1.00 (no interval) | underpowered | 1.02 [0.92, 1.15] | 0.751 | 1.03 [0.93, 1.14] |
| kimi27sglang | c | 37/40 | 1.00 | 0% | 1.19 [0.84, 1.69] | 0.597 | 0.82 [0.67, 1.00] | 0.289 | 0.86 [0.73, 1.02] |
| kimi27sglang | hip | 32/40 | 1.00 | 0% | 1.02 [0.90, 1.15] | 0.822 | 0.86 [0.70, 1.06] | 0.425 | 0.92 [0.78, 1.07] |
| kimi27sglang | triton | 38/40 | 1.00 | 0% | 1.24 [0.83, 1.86] | 0.597 | 1.32 [1.11, 1.57] | 0.048* | 1.18 [1.03, 1.38] |

Nine pairs, eighteen tests, one significant verdict: Kimi-K2.7-Code with the packet spent 1.32x as
many tokens per kernel in Triton (total 1.18x, q = 0.048) and was no faster for it. No speed-up leg
is significant on any model or language.

GPT-OSS-120B in Triton solved 5 kernels without the packet and 6 with it, and the two sets do not
overlap, so its speed-up leg has no pairs at all and reports `underpowered` rather than a ratio.

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH`, `PYTHON` (and `RUNS` for `--extract`) as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh"                     # data/llr-focus40-gpu.db -> figures/ + tables/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh" --extract           # CSCS only: rebuild the .db from $RUNS first
"$ARTIFACT_ROOT/experiments/llr-focus40-gpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

Example on CSCS Beverin:

```sh
HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena \
PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python \
RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs \
    /capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility/experiments/llr-focus40-gpu/reproduce.sh
```

## Outputs

| file | how to read it |
|---|---|
| `figures/gpu.pdf`, `tables/gpu.csv` | geometric mean speed-up and total tokens per arm, across HIP, Triton, OpenMP offload |
| `figures/paired_skills.pdf`, `tables/paired_skills.csv` | per-kernel paired speed-up/cost ratio, skills on vs off, with significance |
| `tables/impact_lang_skills_gpu.csv` | the intervention impact table: one row per arm, the packet row carrying both legs, the total-token ratio, and the usage and relaunch counts behind them |
| `tables/lang_skills_<language>_kernels.csv`, `figures/lang_skills_<language>_kernels.pdf` | per kernel, each arm's speed-up and task token total, with a geomean and median summary row; no ratios and no tests |

## Data provenance

Run root `<RUNS>/gpu-llr-focus40-2026*`, `RUNS` defaults to
`/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs`. Excluded: jobs 631274-631277, graded
while the judge refused Python, so every row in them is actually a C submission mislabeled Triton.

## Caveats

Triton coverage is low on GPT-OSS-120B, 5 and 6 kernels of 40, so its Triton row carries a token leg
and no speed-up leg. Device residency (the kernel actually running on the GPU, not falling back to
host) is not enforced per submission.

Qwen3.8-27B relaunched 22-62% of its tasks depending on the language; the other two models
relaunched none. Its reported cost is the final attempt's, and its crashed attempts spent a further
41.6M tokens across this campaign, recorded in `tokens_crashed` and in no ratio. The same crashes
cost it answers: 71 of the campaign's (arm, kernel) answers were graded before their task's final
attempt began and are dropped by X7, all of them Qwen's, with a median dropped speed-up of 18x.

Jobs 638935 and 638940 (Kimi-K2.7-Code with the packet, OpenMP offload and HIP) were still in the
queue when this snapshot was extracted and are excluded, so those two arms are read at their last
finished wave.
