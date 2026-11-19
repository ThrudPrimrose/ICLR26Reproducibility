#!/bin/bash
# Appendix C per-kernel figure and Figure 5 (cost weighting) from the one-limit reruns, on Ault (login node, nice).
#   ./run.sh            # from the laptop: copy the scripts, run on Ault, fetch PDFs/CSVs into out/, install in paper
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=${PAPER:-$HOME/.cache/wt/paper-revision/figures}
R=rv/ana1118/out/fig-appendix-50m
ssh ault "mkdir -p ~/$R" 2>&1 | grep -vi quantum || true
V1=$HOME/Work/rebuttal-review/analysis/migrate-320a45c61
scp -q "$HERE/cheating_per_kernel.py" "$HERE/figure5_numbers.py" "$HERE/figure5_rerun.py" "$V1/figure5.py" "$V1/analysis.py" ault:$R/
# Figure 5 inputs: the CPF comparison keeps its v1 data (never rerun).
ssh ault "mkdir -p ~/$R/db" 2>&1 | grep -vi quantum || true
scp -q "$V1/db/llr-focus40.db" ault:$R/db/llr-focus40.db
ssh ault bash -s <<'REMOTE' 2>&1 | grep -vi quantum
set -e
X=${X:-$HOME/rv/ana1118} O=$X/out/fig-appendix-50m
A=$X/out/out-ault-1130-abi
cd $X/code-den
py() { singularity exec -B /scratch ${JUDGE_SIF:-judge-cpu.sif} bash -c "unset PYTHONSAFEPATH; export PYTHONPATH=\$PWD:\$PWD/agent PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 HPCAGENT_BENCH_BUDGETS_MATCHED_LIMITS=flag MPLBACKEND=Agg; nice ${PY:-python3} $*"; }
py $O/cheating_per_kernel.py --tag-file $A/tags/llr40.txt --height-pt 145.984 --panel-height-in ${PANEL:-1.12} \
    --out $O/cheating_per_kernel.pdf $A/obs/obs-*.db | tee $O/cheating_per_kernel.log
py $O/figure5_numbers.py --harness20 $X/out/xfinal-1125/obs-final-all.db --llr40 $A/obs/obs-*.db --out $O/figure5_numbers.csv | tee $O/figure5.log
ln -sf $X/out/xfinal-1125/obs-final-all.db $O/db/harness20-rerun.db
ln -sf $A/obs/obs-main-0.db $O/db/rev-llr40.db
py $O/figure5_rerun.py --wt $X/code-den --db $O/db --out $O/figure5_cost_weighting.pdf | tee $O/figure5_rerun.log
REMOTE
mkdir -p "$HERE/out"
scp -q "ault:$R/*.pdf" "ault:$R/*.png" "ault:$R/*.csv" "ault:$R/*.log" "$HERE/out/"
# Figure 5: v1's figure5.py (hand-written matplotlib, the paper's own script) over the rerun inputs.
for f in cheating_per_kernel figure5_cost_weighting; do
    [[ -e $PAPER/v1_$f.pdf ]] || cp "$PAPER/$f.pdf" "$PAPER/v1_$f.pdf"
    cp "$HERE/out/$f.pdf" "$PAPER/$f.pdf"
    pdfinfo "$PAPER/v1_$f.pdf" | grep "Page size"; pdfinfo "$PAPER/$f.pdf" | grep "Page size"
done
