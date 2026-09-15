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

**Summary statistics.** Per arm, the speed-up is the geometric mean over the kernels it verified,
with a log-t interval (A1), and the token cost is the median task total with a percentile bootstrap
interval (A2). Per pair, both legs are paired by kernel: the geometric mean ratio with its log-t
interval and paired t test (P3), and beside the token leg the ratio of TOTAL tokens over the shared
kernels with a paired bootstrap interval (9999 resamples, seed 0). The two token numbers answer
different questions and the table carries both: the geomean is the typical kernel, the total is the
budget. Benjamini-Hochberg at q = 0.05 runs over one family, and the family is every leg of every
pair of one invocation (M1).

The two campaigns differ in one more way the reader has to hold: the blind arms run under
`AGENT_SINGLE_SUBMISSION=1`, so an accepted submission ends the task. A blind task that never
submits has no answer, and the pair then loses that kernel from its speed-up leg rather than
scoring it at 1.

## Results

Ratios are scored / blind, so above 1 means the scored arm was faster or spent more. `n` is the
kernels behind the speed-up leg and behind the token leg. `q` is the Benjamini-Hochberg adjusted p
over the family of all twelve pairs; `*` marks q < 0.05.

<!--TABLE impact_blind_vs_scored-->

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
