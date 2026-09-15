# ICLR26 Reproducibility

Data and reproduction commands for the ICLR26 submission's agent-optimization experiments: LLM
agents rewriting numerical kernels for speed, scored against compiler and framework baselines on
CSCS Beverin (AMD MI300A nodes, 4 APUs per node, 24 Zen4 cores per APU). Each grade runs on one
APU's 24 physical cores. Scores, statistics and plots come from HPCAgent-Bench; this repository
holds only committed data and the commands that turn it into figures and tables.

## Folders

| folder | what it is |
|---|---|
| `experiments/<name>/` | one experiment: `data/<name>.db`, `figures/`, `tables/` (when it has tables), `reproduce.sh`, `SHA256SUMS` |
| `experiments/common.sh` | shared shell helpers every `reproduce.sh` sources (`extract`, `check`) |
| `skill_histories/` | every version of the **language packet** (Language Skill Packet) the agents read |
| `notes/` | earlier notes, kept for provenance, not part of the reproduction path |
| `requirements.txt` | Python dependencies for extraction and plotting |
| `HPCAGENT_BENCH_COMMIT` | the HPCAgent-Bench commit the committed figures were built with; a different commit prints a note |

## Experiments

| name | question | what varies | models | kernels |
|---|---|---|---|---|
| `llr-focus40-cpu` | does a packet or the Canonical Parallel Form help, on CPU? | no packet, Language Skill Packet, CPF page, CPF as source, perf playbook (perf-playbook-cpu); CPF conditions are C only | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, GLM-5.3 | 40 `llr-focus40` kernels, C and Fortran |
| `llr-focus40-gpu` | does the packet help on GPU, across delivery languages? | HIP, Triton (Python delivery), C + OpenMP offload, with/without Language Skill Packet | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | same 40 kernels, on the MI300A GPU |
| `llrblind` | does removing the score tool and capping submissions to one change the outcome? | Language Skill Packet on/off, C vs Fortran, one submission, no score route | GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code | same 40 kernels, CPU |
| `git-scicomp` | does handing the agent the whole repository beat handing it the bare kernel? | bare kernel vs repository, 3 agents per kernel | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | 10 scientific-computing kernels |
| `perf-playbook` | does a performance-engineering playbook packet change what the agent delivers? | no packet vs the playbook packet, on scicomp-focus40, loop-level C and loop-level HIP | Qwen3.8-27B, GPT-OSS-120B | 40 scientific-computing and 40 loop-level kernels |
| `canon` | what does DaCe canonicalization buy against plain compilers, with no agent? | toolchain column: numba, cc, cc_autopar, dace_cpu, dace_cpu_canonicalize (+ GPU columns) | none, deterministic compiler baselines | same 40 kernels |

## Reproduce

Needs `git`, Python 3.12 or newer and `sha256sum`. Every `reproduce.sh` can be called from any
directory.

### 1. Set the paths

```sh
export ARTIFACT_ROOT=/path/to/ICLR26Reproducibility   # this repository
export HPCAGENT_BENCH=/path/to/hpcagent-bench         # HPCAgent-Bench checkout, latest main
export PYTHON=/path/to/venv/bin/python                 # a Python with requirements.txt installed
export RUNS=/path/to/hpcagent-bench-runs               # --extract only: the campaign run directories
export CANON_SWEEP=/path/to/canon-llr40-sweep          # --extract of canon only: the compiler sweep
```

Example on CSCS Beverin:

```sh
export ARTIFACT_ROOT=/capstor/scratch/cscs/ybudanaz/x86_64/ICLR26Reproducibility
export HPCAGENT_BENCH=/capstor/scratch/cscs/ybudanaz/x86_64/optarena
export PYTHON=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python
export RUNS=/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs
export CANON_SWEEP=/capstor/scratch/cscs/ybudanaz/x86_64/canon-llr40-20260910
```

### 2. Install (once)

```sh
git clone https://github.com/ThrudPrimrose/ICLR26Reproducibility.git "$ARTIFACT_ROOT"
git clone https://github.com/spcl/HPCAgent-Bench.git "$HPCAGENT_BENCH"
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install -r "$ARTIFACT_ROOT/requirements.txt"
```

Use HPCAgent-Bench's latest `main`, not a pinned commit.

### 3. Reproduce the figures and tables

One experiment (any name from the table above):

```sh
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh"
```

All experiments:

```sh
for name in llr-focus40-cpu llr-focus40-gpu llrblind git-scicomp canon; do
    "$ARTIFACT_ROOT/experiments/$name/reproduce.sh"
done
```

Each run rebuilds `figures/` (and `tables/`, where the experiment has them) from the committed
`data/<name>.db` and prints `OK: every figure and table matches SHA256SUMS` on success.

### 4. Rebuild the data (CSCS cluster only)

```sh
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract           # $RUNS -> data/llr-focus40-cpu.db, then step 3
"$ARTIFACT_ROOT/experiments/canon/reproduce.sh" --extract                     # $CANON_SWEEP -> data/canon.db, then step 3
"$ARTIFACT_ROOT/experiments/llr-focus40-cpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

`--record` rewrites `SHA256SUMS` instead of checking it. Use it only when the data changed on purpose.

## What the numbers mean

The full specification is `docs/DESIGN_data_collection_and_scoring.md` in HPCAgent-Bench; this is
what a reader of the tables has to know.

A per-kernel speedup is `median(baseline run time) / median(candidate run time)` over 20 timed
repeats each, credited only when a one-sided Mann-Whitney U test confirms the direction (p = 0.1);
otherwise the speedup is exactly 1.0, and a confirmed slow-down is below 1. The baseline is Numba
for the loop-level-reasoning kernels (`llr-focus40-cpu`, `llr-focus40-gpu`, `llrblind`) and C -O3 +
autopar for `git-scicomp` and `perf-playbook`'s scicomp campaign. Rows graded before 2026-09-13
used an older timing rule and were re-timed; a submission with no re-timed row is dropped, so every
speedup here comes from one rule.

A TASK is one agent optimizing one kernel once. Within a task the LAST verified submission is the
answer; a suspect row and a non-positive speedup are not candidates. A kernel a campaign runs more
than once is reduced by that campaign's policy: the LATEST task where a rerun replaces what it
repeats, the MEDIAN over tasks where the repeats are by design (`git-scicomp` and `perf-playbook`'s
scicomp campaign run three agents per kernel). The maximum over an arm's tasks is never taken, and
tokens are never summed over tasks.

THE LAST AGENT RAN THE TASK FROM NOTHING TO ITS END. A crashed agent is relaunched from an empty
context and an empty workspace, so a task is scored and priced as its final attempt alone: judge
rows stamped before that attempt started are dropped, the token total is the final attempt's, and
what the earlier attempts spent is reported beside it as `tokens_crashed` and added to nothing.
Cancelled tasks are dropped whole. Every table carries `attempts_per_task` and the share of tasks
that relaunched, because that is where this rule applied.

TOKENS ARE COUNTED ONCE. `output` is every token the model generated, reasoning included, as both
serving engines report it; the client's thinking estimate is never added on top. Input is counted
when it first enters the context, so a cached prompt is not billed again on every turn. Where no
server count survived, the count comes from the model's own tokenizer over the transcript, which is
2-4% low by construction and is marked `output_source = retokenized` on the task row.

Every OVERALL speedup (per arm, per experiment, or any other summary) is the GEOMETRIC MEAN of the
per-kernel speedups, never the median: a speedup is a ratio, and a median over a mostly-flat
distribution reports "no effect" everywhere a geometric mean would not. A token cost per arm is the
MEDIAN task total. A paired comparison reports the geometric mean ratio with a log-t interval and a
paired t test, and on the token leg also the ratio of total tokens with a paired bootstrap interval
(9999 resamples, seed 0). Benjamini-Hochberg at q = 0.05 runs over one declared family, and only a
corrected verdict is called significant.

## Status (snapshot 2026-09-15)

Every experiment carries a committed database, figures, tables and checksums, re-extracted after
the re-grade and after the token records were re-folded, so no committed number mixes two timing
rules or two token folds.

A `-clean` arm is a re-run of one condition from scratch and is excluded by name: it carries the
same identity as the arm it supersedes, so one incomplete re-run would replace a finished campaign.
Arms whose jobs were still in the queue on 2026-09-15 are read at their last finished wave, and the
excluded job ids are named in each `reproduce.sh` and README. The re-runs launched that evening are
not in this snapshot.

## Full artifact

The earlier, more detailed artifact (per-wave provenance, campaign history, superseded readings)
is kept on branch `archive/paper-artifacts-20260915` and is not part of the reproduction path above.
