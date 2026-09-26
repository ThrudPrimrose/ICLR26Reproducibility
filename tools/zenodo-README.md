# HPCAgent-Bench: anonymized artifact

This package is the anonymized artifact of a conference submission under double-blind review. It holds the benchmark and
everything needed to regenerate the paper's figures and tables from its measurements. Names,
sites and commit ids are anonymized throughout.

## Contents

| Folder | Holds |
|---|---|
| `hpcagent-bench/` | the benchmark: kernels, agent harnesses, scoring, statistics and plotting code, docs |
| `reproducibility/` | the paper's pipeline (`run_all.sh`, `stats.sh`, one script per figure and table) and all its inputs and outputs |

Inside `reproducibility/`:

| Path | Holds |
|---|---|
| `data/` | the measurements, extracted and ready to use: one SQLite database per experiment, the compiler sweep, the GH200 regrade |
| `data.tar.zst`, `DATA_SHA256SUMS` | the same data as one archive, with a checksum per file |
| `work/` | the pooled answers the figures read (`*.db`, `*-grade-sources.csv`, `graded-arms.txt`) and the GH200 transfer plots |
| `tables/` | every CSV and LaTeX table of the pipeline (paired tests per figure panel, comparators, GH200 transfer) |
| `figures/` | every figure, PDF and PNG |
| `SHA256SUMS` | checksums of `figures/` and `tables/` |
| `case-studies/` | full submission sources behind the case studies (`INDEX.md`) |
| `skill_histories/` | every version of the skill packets the agents read |
| `lib/`, `tools/` | the pipeline's helpers and the anonymization tools |

## Quick start: regenerate every figure and table (no GPU, no model access)

Python 3.12 or newer and pip 25.1 or newer.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e "hpcagent-bench[cpu]" --group hpcagent-bench/pyproject.toml:dace
cd reproducibility
./run_all.sh
```

`run_all.sh` finds `../hpcagent-bench` by itself and reads the extracted `data/`, so no download
step is needed. It pools the answers (`work/`), writes the tables and figures, and ends with

```
OK: @CHECKSUMS@ files match SHA256SUMS
```

or names every file that differs. Each figure has its own script (`fig2.sh` ... `tab3.sh`); the
table in `reproducibility/README.md` maps scripts to paper figures.

To check the data against the archive instead of the extracted copy:

```sh
cd reproducibility/data && sha256sum --quiet -c ../DATA_SHA256SUMS && echo data OK
```

## Rerun the experiments

The experiments themselves need GPUs or CPU nodes and model endpoints. Start from
`hpcagent-bench/README.md` (install, a first local run, running a campaign), then
`hpcagent-bench/docs/README.md` (index of the docs: launch, scoring, timing protocol,
statistics, token accounting) and `hpcagent-bench/SUBMITTING.md` (campaign scripts under
`hpcagent-bench/experiments/`). `reproducibility/README.md` describes how the measurements of a
campaign become `data/`.
