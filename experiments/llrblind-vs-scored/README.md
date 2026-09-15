# llrblind-vs-scored

## Question

What do the score tool and unlimited submissions buy? The agent in `llr-focus40-cpu` can score a
candidate against the visible tests as often as it likes and submit as often as it likes; the agent
in `llrblind` has neither, and gets one accepted submission.

## Conditions

`cpf-llr-focus40-<model>-<language>[-skills]` (scored) against
`llrblind-<model>-<language>[-skills]` (blind), matched by model, language and packet: twelve pairs
over the same 40 `llr-focus40` CPU kernels. This experiment has no database of its own; it reads
its two siblings' directly.

## Models and kernels

GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code, driven by Claude Code, in C and Fortran, with and
without the Language Skill Packet. Same 40 `llr-focus40` kernels, CPU, speed-up over Numba.

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

The two campaigns differ in one more way the reader has to hold: the blind arms run under
`AGENT_SINGLE_SUBMISSION=1`, so an accepted submission ends the task. A blind task that never
submits has no answer, and the pair then loses that kernel from its speed-up leg rather than
scoring it at 1.

## Results

Ratios are scored / blind, so above 1 means the scored arm was faster or spent more. `n` is the
kernels behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg adjusted p
over the family of all twelve pairs; `*` marks q < 0.05.

<!--TABLE impact_blind_vs_scored-->
| model | language | n | attempts/task | relaunched | speed-up ratio | q | token ratio | q | total tokens |
|---|---|---|---|---|---|---|---|---|---|
| oss120b | c | 37/40 | 1.00 | 0% | 1.23 [0.93, 1.65] | 0.321 | 1.55 [1.25, 1.93] | 0.003* | 1.72 [1.35, 2.15] |
| oss120b | c | 30/40 | 1.00 | 0% | 1.01 [0.80, 1.26] | 0.961 | 1.46 [1.18, 1.79] | 0.003* | 1.70 [1.37, 2.07] |
| oss120b | fortran | 33/40 | 1.00 | 0% | 1.38 [1.03, 1.85] | 0.100 | 1.55 [1.24, 1.95] | 0.003* | 1.75 [1.37, 2.22] |
| oss120b | fortran | 30/40 | 1.00 | 0% | 1.18 [0.83, 1.66] | 0.525 | 1.41 [1.19, 1.68] | 0.003* | 1.52 [1.26, 1.84] |
| qwen38 | c | 18/40 | 1.88 | 60% | 1.13 [0.74, 1.72] | 0.740 | 0.76 [0.55, 1.04] | 0.217 | 0.97 [0.75, 1.24] |
| qwen38 | c | 21/40 | 1.90 | 62% | 1.29 [0.90, 1.84] | 0.321 | 0.89 [0.66, 1.18] | 0.564 | 1.12 [0.88, 1.38] |
| qwen38 | fortran | 19/39 | 1.43 | 30% | 0.94 [0.67, 1.32] | 0.796 | 0.62 [0.46, 0.84] | 0.010* | 0.81 [0.65, 0.97] |
| qwen38 | fortran | 26/40 | 1.60 | 48% | 0.96 [0.75, 1.24] | 0.830 | 0.64 [0.46, 0.88] | 0.023* | 0.87 [0.68, 1.06] |
| kimi27sglang | c | 38/40 | 1.00 | 0% | 1.76 [1.30, 2.38] | 0.003* | 0.85 [0.66, 1.09] | 0.347 | 0.85 [0.72, 1.02] |
| kimi27sglang | c | 35/40 | 1.00 | 0% | 0.90 [0.72, 1.12] | 0.525 | 0.98 [0.76, 1.25] | 0.878 | 0.98 [0.83, 1.17] |
| kimi27sglang | fortran | 36/40 | 1.00 | 0% | 1.29 [0.92, 1.82] | 0.321 | 1.15 [0.92, 1.45] | 0.347 | 1.22 [1.03, 1.45] |
| kimi27sglang | fortran | 35/40 | 1.00 | 0% | 0.95 [0.77, 1.16] | 0.744 | 0.95 [0.75, 1.21] | 0.796 | 0.91 [0.77, 1.08] |

## Commands

Set `ARTIFACT_ROOT`, `HPCAGENT_BENCH` and `PYTHON` as in the top-level README, then:

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract   # CSCS only, once
"$ARTIFACT_ROOT/experiments/llrblind/reproduce.sh" --extract          # CSCS only, once
"$ARTIFACT_ROOT/experiments/llrblind-vs-scored/reproduce.sh"          # tables/ + figures/, check SHA256SUMS
"$ARTIFACT_ROOT/experiments/llrblind-vs-scored/reproduce.sh" --record # rewrite SHA256SUMS instead
```

## Outputs

| file | how to read it |
|---|---|
| `tables/paired_arms.csv`, `figures/blind_vs_scored.pdf` | the two legs of each pair with intervals, wins and losses, the family's corrected verdicts |
| `tables/impact_blind_vs_scored.csv`, `tables/arms.csv` | the intervention impact table and the per-arm table behind it: geomean speed-up, median task tokens, attempts, calls, crashed spend |

## Data provenance

`../llr-focus40-cpu/data/llr-focus40-cpu.db` and `../llrblind/data/llrblind.db`, each built by its
own `reproduce.sh --extract`. Excluded jobs are stated in those two experiments' READMEs.

## Caveats

The two campaigns ran on different days with different serving builds, so this is a comparison of
two campaigns and not of two conditions inside one. Everything else about the arms is matched:
model, language, packet, roster, baseline and harness.

`llrblind-qwen38-c-skills` ran under a tighter global token cap than the other arms and most of its
agents never reached submit, so its row reports answer quality on the kernels it did deliver and is
not a coverage comparison.

Qwen3.8-27B relaunched a large share of its tasks on both sides; the cost on both sides is the
final attempt's, and the crashed attempts are in `tokens_crashed` and in no ratio.
