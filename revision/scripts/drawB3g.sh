#!/bin/bash
X=${X:-$HOME/rv/ana1118}; O=${O:-$X/fig6/ault-1005}
cd $X/code-den
singularity exec -B /scratch ${JUDGE_SIF:-judge-cpu.sif} bash -c '
unset PYTHONSAFEPATH; export LEGEND_NCOL=3 PYTHONPATH=$PWD:$PWD/agent PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
${PY:-python3} -m hpcagent_bench.stats.commands.scaling '"$O"'/all-obs.db --figure mode-grid --quantity speedup --print-width 5.5 --setups "^(torch_dist|rev-mlscale20-.*)$" --kernels dist_moe_router dist_sdpa dist_all_to_all_transpose dist_softmax --out '"$O"'/ngB3g --table '"$O"'/ngB3g.csv 2>&1 | grep -v VIRTUAL | tail -1'
