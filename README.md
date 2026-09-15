# ICLR26 Reproducibility

| folder | what it is |
|---|---|
| `experiments/<name>/` | one experiment: `data/<name>.db`, `figures/`, `tables/`, `reproduce.sh`, `SHA256SUMS` |
| `skill_histories/` | every version of the **language packet** (Language Skill Packet) the agents read |
| `notes/` | earlier notes |

Experiments: `llr-focus40-cpu`, `llr-focus40-gpu`, `llrblind`, `git-scicomp`, `canon`.
Scores, statistics and plots come from HPCAgent-Bench; this repository holds only data and commands.

## Reproduce an experiment

Needs `git`, Python 3.12 or newer, and `sha256sum`. Run from this folder:

```sh
git clone https://github.com/spcl/HPCAgent-Bench.git hpcagent-bench
git -C hpcagent-bench checkout "$(cat HPCAGENT_BENCH_COMMIT)"
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
HPCAGENT_BENCH="$PWD/hpcagent-bench" PYTHON="$PWD/.venv/bin/python" experiments/git-scicomp/reproduce.sh
```

It rebuilds `figures/` and `tables/` from `data/git-scicomp.db` and prints
`OK: every figure and table matches SHA256SUMS`. Replace `git-scicomp` with any experiment name.

`reproduce.sh --extract` also rebuilds the `.db` from the judge databases. That works only on the
CSCS cluster, where the run directories are (`RUNS=` overrides their location).

The full earlier artifact is on branch `archive/paper-artifacts-20260915`.
