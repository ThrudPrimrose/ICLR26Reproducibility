#!/bin/bash
# A/A figure with gelu through `hpcagent-bench plot kernel-runs`: temperature3 (3 kernels) + repeat3 gelu (promoted).
X=${X:-$HOME/rv/ana1118}; D=$X/out/aa; A=$X/out/${STAMP:-out-ault-1130-abi}
cd $X/code-den
C="singularity exec -B /scratch ${JUDGE_SIF:-judge-cpu.sif}"
ENV='unset PYTHONSAFEPATH; export PYTHONPATH=$PWD:$PWD/agent PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg HPCAGENT_BENCH_BUDGETS_MATCHED_LIMITS=flag'
PY=${PY:-python3}
rm -rf /dev/shm/aa-$USER; $C bash -c "$ENV; nice $PY -m hpcagent_bench extract --runs $D/temperature3.db --benchmarks \$PWD/hpcagent_bench/benchmarks --out /dev/shm/aa-$USER --no-sources --threads 4 --db $D/t3-obs.db" > $D/extract.log 2>&1; echo extract rc=$?
rm -rf /dev/shm/aa-$USER
rm -f $D/aa4-obs.db; cp $D/t3-obs.db $D/aa4-obs.db
for o in $A/obs/obs-*.db; do
  sqlite3 $D/aa4-obs.db "attach '$o' as r; insert into observations select * from r.observations where kernel='gelu' and setup in ('rev-repeat3-ml-feedback-qwen38-hip','rev-repeat3-ml-feedback-qwen38-hip-t0','rev-repeat3-ml-feedback-oss120b-hip','rev-repeat3-ml-feedback-oss120b-hip-t0'); insert or ignore into setups select * from r.setups where name like 'rev-repeat3-ml-feedback-%';" 2>&1 | head -2
done
sqlite3 $D/aa4-obs.db "select setup, count(*) from observations group by setup"
printf 'scan_affine_decay\nkmp\naddusxx_g\ngelu\n' > $D/aa4.txt
$C bash -c "$ENV; nice $PY -m hpcagent_bench plot kernel-runs $D/aa4-obs.db --tag $D/aa4.txt --conditions t0,default --out $D/aa4.pdf" 2>&1 | tail -5
