#!/bin/bash
# Figures 3 and 4 at 50M from the one-limit reruns, drawn by `hpcagent-bench plot efficacy` through the v1 driver
# figures.py (presentation-only overrides). Run on Ault: bash run.sh (from anywhere).
4] draws one figure only.|4] draws one figure only.
# Figure 4 panel (a) is "Git vs. Kernel": the API names its per-panel tables after the title, and "Git / Kernel" fails
# (ValueError: invalid file name with a slash), with or without --table.
# Inputs copied from the laptop to $W/in: figures.py, comparators.csv, CPF v1 pairs + sidecar, llr-focus40.db.
set -u
X=${X:-$HOME/rv/ana1118}
W=$X/out/fig34-50m
P=$X/out/pairs50-1125
OBS="$(ls $X/out/out-ault-1130-abi/obs/obs-*.db | tr '\n' ' ') $X/out/xfinal-1125/obs-final-all.db"
OBSC=$(echo $OBS | tr ' ' ',')
C=$W/in/comparators.csv
mkdir -p $W/out
cd $X/code-den
sif() {
  singularity exec -B /scratch ${JUDGE_SIF:-judge-cpu.sif} bash -c "unset PYTHONSAFEPATH; export PYTHONPATH=\$PWD:\$PWD/agent PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 HPCAGENT_BENCH_BUDGETS_MATCHED_LIMITS=flag MPLBACKEND=Agg; nice ${PY:-python3} $*"
}
# Figure 3: CPF keeps its own panel. A merged "intervention=packets" panel was tried: the v1 CPF setups end in
# "cpfsrc-v2", which the packet registry does not know, so the API labels them "C-No Skill Packet" with no CPF shape and
# no CPF legend entry. The CPF pairs are the v1 family rows (no rerun) with their v1 sidecar.
[ "${1:-}" = 4 ] || sif -W ignore $W/in/figures.py $W/out/figure3_packets_and_scope.pdf -- $OBS $W/in/llr-focus40.db \
  --mode absolute --cost-model billed --include-incomplete --dots-row-height ${ROW3:-0.7077} --row-width iclr \
  --comparison "\"title=LLR CPU;intervention=lang-skills;pairs=$P/llr-cpu.csv;comparators=$C;comparator-set=pluto:C\"" \
  --comparison "\"title=CPF;intervention=cpf-cpu-src;pairs=$W/in/llr-cpu-cpfsrc-v2.csv;observations=$W/in/llr-focus40.db\"" \
  --comparison "\"title=LLR GPU;intervention=lang-skills;pairs=$P/llr-gpu.csv;comparators=$C;comparator-set=ppcg_hip:HIP\"" \
  --comparison "\"title=Toolkit;intervention=perf-playbook-cpu;pairs=$P/sc40-toolkit.csv\"" \
  --table $W/out/figure3.csv
[ "${1:-}" = 3 ] || sif -W ignore $W/in/figures.py $W/out/figure4_scope_row.pdf -- $OBS \
  --mode absolute --cost-model billed --include-incomplete --dots-row-height ${ROW4:-0.7258} --row-width iclr \
  --comparison "\"title=Git vs. Kernel;intervention=repo;pairs=$P/git.csv;control-label=Kernel\"" \
  --comparison "\"title=Harness20;intervention=harness;pairs=$P/harness20.csv;control-label=Claude Code\"" \
  --comparison "\"title=LLR Blind;intervention=no-score-tool;pairs=$P/blind-vs-open.csv;control-label=Open\""
