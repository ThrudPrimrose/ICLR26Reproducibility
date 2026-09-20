# ICLR26 Reproducibility

Data and reproduction commands for the ICLR26 submission's agent-optimization experiments: LLM
agents rewriting numerical kernels for speed, scored against compiler and framework baselines on
CSCS Beverin (AMD MI300A nodes, 4 APUs per node, 24 Zen4 cores per APU). Each grade runs on one
APU's 24 physical cores. Scores, statistics and plots come from HPCAgent-Bench; this repository
holds only committed data and the commands that turn it into figures and tables.

## Folders

| folder | what it is |
|---|---|
| `experiments/<name>/` | one experiment, everything it is made of: `data/` (its observations `.db` and `.csv`), `tables/` (the paired-arm CSVs a figure draws from and the verdicts it stars from), `figures/` (`.pdf` + `.png`), `reproduce.sh`, `SHA256SUMS` |
| `experiments/common.sh` | shared shell helpers every `reproduce.sh` sources (`require_data`, `extract`, `check`) |
| `experiments/reproduce_all.sh` | runs every experiment, one status line each, non-zero exit if any failed |
| `tests/test_common_sh.sh` | exercises the shared shell helpers and the run-all script against a throwaway tree |
| `skill_histories/` | every version of the **language packet** (Language Skill Packet) the agents read |
| `notes/` | earlier analysis, kept for provenance, not part of the reproduction path and superseded by the rebuilt numbers |
| `requirements.txt` | Python dependencies for extraction and plotting |
| `HPCAGENT_BENCH_COMMIT` | the HPCAgent-Bench commit the committed figures were built with; a different commit prints a note |

### Where a number lives

Everything a figure is made of is in this repository, in that figure's own experiment folder: the
observations, the paired-arm CSVs, the verdicts, the tables and the rendered figure. Nothing is
spread across repositories, because a figure whose inputs live somewhere else is a figure nobody
can check.

The other two repositories borrow from it rather than holding their own copies:

* **HPCAgent-Bench** owns the CODE -- the extractor, the statistics and the plotting API. Its
  `data/` directory is a scratch area for the extraction currently being run and is untracked;
  no result is kept there.
* **The paper** copies the `.pdf` it includes out of `experiments/<name>/figures/`. It is a
  consumer of this repository, never a source for it.

One experiment is one folder, split by device where the experiment ran on both: `llr-cpu` and
`llr-gpu` are the CPU and GPU halves of the forty loop-level kernels, and `scicomp-cpu` and
`scicomp-gpu` the same for the scientific-computing forty.

## Experiments

| name | question | what varies | models | kernels |
|---|---|---|---|---|
| `llr-cpu` | does a packet or the Canonical Parallel Form help, on CPU? | no packet, Language Skill Packet, CPF page, CPF as source, perf playbook (perf-playbook-cpu); CPF conditions are C only | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code, GLM-5.3 | 40 `llr-focus40` kernels, C and Fortran |
| `llr-gpu` | does the packet help on GPU, across delivery languages? | HIP, Triton (Python delivery), C + OpenMP offload, with/without Language Skill Packet | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | same 40 kernels, on the MI300A GPU |
| `llrblind` | does removing the score tool and capping submissions to one change the outcome? | Language Skill Packet on/off, C vs Fortran, one submission, no score route | GPT-OSS-120B, Qwen3.8-27B, Kimi-K2.7-Code | same 40 kernels, CPU |
| `git-scicomp` | does handing the agent the whole repository beat handing it the bare kernel? | bare kernel vs repository, 3 agents per kernel | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | 10 scientific-computing kernels |
| `scicomp-cpu` | does a perf playbook or divide-and-conquer framing help on real scientific kernels? | no packet, perf playbook, divide-and-conquer; C and Fortran | Qwen3.8-27B, GPT-OSS-120B, Kimi-K2.7-Code | 40 `scicomp40` kernels, CPU |
| `scicomp-gpu` | the same, delivered to the GPU | HIP and C + OpenMP offload, with and without the packet | Qwen3.8-27B, GPT-OSS-120B | same 40 kernels, on the MI300A GPU |
| `harness20` | does the agent HARNESS change the outcome, with the model held fixed? | Claude Code, mini-SWE, OpenHands, Optimas | Qwen3.8-27B, GPT-OSS-120B | 20 kernels (14 `scicomp40` lvl1/2 + 6 loop-level lvl2) |
| `canon` | what does DaCe canonicalization buy against plain compilers, with no agent? | toolchain column: cc, cc_autopar, numba, pluto, ppcg, dace_cpu_canonicalize, dace_gpu_canonicalize | none, deterministic compiler baselines | same 40 kernels |

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

Use HPCAgent-Bench's latest `main`, not a pinned commit. Install `requirements.txt` as pinned,
though: `SHA256SUMS` records the bytes matplotlib 3.11.1 and that exact dependency set draw, and a
different matplotlib changes a PDF without changing a number.

### 3. Reproduce the figures and tables

One experiment (any name from the table above):

```sh
"$ARTIFACT_ROOT/experiments/git-scicomp/reproduce.sh"
```

All experiments:

```sh
"$ARTIFACT_ROOT/experiments/reproduce_all.sh"
```

It runs every experiment whatever the earlier ones did, prints `ok <name>` or `FAIL <name> (exit N)`
for each, and exits non-zero if any failed.

Each run rebuilds `figures/` (and `tables/`, where the experiment has them) from the committed
`data/<name>.db` and prints `OK: every figure and table matches SHA256SUMS` on success. Only `canon`
carries its database today; the other four exit 2 with one line naming the missing file (see Status).

### 4. Rebuild the data (CSCS cluster only)

```sh
"$ARTIFACT_ROOT/experiments/llr-cpu/reproduce.sh" --extract           # $RUNS -> data/llr-cpu.db, then step 3
"$ARTIFACT_ROOT/experiments/canon/reproduce.sh" --extract                     # $CANON_SWEEP -> data/canon.db, then step 3
"$ARTIFACT_ROOT/experiments/llr-cpu/reproduce.sh" --extract --record  # also rewrite SHA256SUMS
```

`--record` rewrites `SHA256SUMS` instead of checking it. Use it only when the data changed on purpose.

`extract_llr40.py` refuses rows graded before the current timing reduction, so on the population this
repository describes `--extract` needs re-graded rows. The path is three steps, and `REGRADES` is the
glob `common.sh` forwards as `--regrades`:

```sh
"$PYTHON" "$HPCAGENT_BENCH/scripts/regrade.py" worklist --observations <db> \
    --env-dir "$HPCAGENT_BENCH/experiments" --out worklist.jsonl   # what still needs re-timing
cd "$HPCAGENT_BENCH/experiments" && sbatch --nodes=<N> regrade.sbatch worklist.jsonl <out-dir>
REGRADES='<out-dir>/regrade-*.db' "$ARTIFACT_ROOT/experiments/llr-cpu/reproduce.sh" --extract
```

Without `REGRADES` the extraction stops and names the count it refused. `--allow-unstamped`, passed
straight to `extract_llr40.py`, extracts them unmigrated and mixes two timing rules in one table;
nothing in this repository is built that way.

## What the numbers mean

A per-kernel speedup is `median(baseline run time) / median(candidate run time)` over 20 timed
repeats each, credited only when a one-sided Mann-Whitney U test confirms the direction (p = 0.1);
otherwise the speedup is exactly 1.0, and a confirmed slow-down is below 1. That per-kernel value
is the credited speedup of the kernel's final answer. The baseline is Numba for the
loop-level-reasoning kernels (`llr-cpu`, `llr-gpu`, `llrblind`) and C -O3 + autopar
for `git-scicomp`. Scoring: within one agent episode the LAST verified submission counts; across an
arm's episodes the maximum is kept; suspect rows are dropped before any aggregate. Every OVERALL
speedup (per arm, per experiment, or any other summary) is the GEOMETRIC MEAN of the per-kernel
speedups, never the median: a speedup is a ratio, and a median over a mostly-flat distribution
reports "no effect" everywhere a geometric mean would not. Tokens are the total per kernel, summed
over every episode that touched it, never averaged. Every row in this repository is measured under
this one rule: rows originally graded before 2026-09-13 used an older rule and were re-timed so no
two numbers here come from different definitions.

## Status (reset 2026-09-20)

**This repository was reset for a fresh set of results.** Every experiment here carries its
`reproduce.sh` and its README; none carries committed figures or a database yet. They are filled in
one experiment at a time as each finishes, against the rebuilt extraction pipeline
(`hpcagent_bench.dataset`), so that no committed number predates it.

The reset was needed because the figures that stood here were built under an earlier per-task score
rule (`s-v2`, which clamped a score to a ceiling) and against pair CSVs whose population is no
longer the one the experiments ran. The current rule is `s-v5`, which credits the raw geometric
mean with no ceiling and no floor. Numbers under the two rules are not comparable, and mixing them
in one table is the failure this reset exists to avoid.

## The archive branch

Everything this repository held before the reset is on **`archive-pre-iclr26-fresh-20260920`**,
branched from the last commit of the old `main`. Nothing was deleted -- that branch is the complete
prior state, including:

* `experiments/canon/` with its committed `canon.db`, its figures and its `SHA256SUMS` -- the
  deterministic compiler baselines as they stood on 2026-09-15;
* every earlier figure and table, under the score rule and the population of its own time.

```sh
git log --oneline archive-pre-iclr26-fresh-20260920      # what was there
git show archive-pre-iclr26-fresh-20260920:experiments/canon/README.md
git checkout archive-pre-iclr26-fresh-20260920 -- experiments/canon   # take a file back
```

Older history lives on **`archive`**, a single branch the earlier archive branches were folded
into (`archive/paper-artifacts-20260915` and the two `x86_64_old` branches). A slash-free name is
therefore required for any new archive branch: git cannot create `archive/<x>` while a branch named
`archive` exists. The unmerged `restructure` and `llr40-reduction` are kept as they are.

### The skill histories are NOT archived

`skill_histories/` stays on `main`. It is every version of the **Language Skill Packet** the agents
were given, `v2` through `v11-as-run`, and which version an arm read is part of what that arm
measured -- an arm run under `v9` is not comparable to one under `v11` even with everything else
held fixed. `HISTORY.md` says what changed between versions and `snapshot.py` is how a version was
captured. `-as-run` marks the version actually served to agents, which is not always the one
authored: `v11-as-run` is C and Fortran only, having dropped the C++ pages `v9` and `v10` carried.

`notes/` also stays, for the same reason -- it records how earlier readings were arrived at -- but
nothing in the reproduction path reads it, and where it disagrees with a current figure the figure
is right.
