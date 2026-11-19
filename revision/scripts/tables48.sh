#!/bin/bash
# Tables 18/19 with the toolkit's Fortran and HIP pairs and Beverin's Harness20 dataset (dataset-harness20-final.db).
X=${X:-$HOME/rv/ana1118}; A=$X/out/${STAMP:-out-ault-1130-abi}; H=$X/out/h20/dataset-harness20-final.db
D=$X/out/pairs48; mkdir -p $D
cp $X/out/xfinal-1125/obs-final-all.db $D/xfinal-noh20.db && sqlite3 $D/xfinal-noh20.db "delete from observations where setup like 'harness20-%'"
M=""; for o in $A/obs/obs-*.db; do
  if [ "$(sqlite3 -readonly $o "select count(*) from observations where setup like 'harness20-%' and setup not like '%caveman%'")" != "0" ]; then
    b=$D/$(basename $o); cp $o $b; sqlite3 $b "delete from observations where setup like 'harness20-%' and setup not like '%caveman%'"; M="$M,$b"
  else M="$M,$o"; fi; done
OBS=${M#,},$D/xfinal-noh20.db,$H
OA=""; for o in ${OBS//,/ }; do OA="$OA --observations $o"; done
cd $X/code-den
C="singularity exec -B /scratch ${JUDGE_SIF:-judge-cpu.sif}"
ENV='unset PYTHONSAFEPATH; export PYTHONPATH=$PWD:$PWD/agent PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 HPCAGENT_BENCH_BUDGETS_MATCHED_LIMITS=flag'
PY=${PY:-python3}
grep -v '^#' hpcagent_bench/tags/harness20.txt | awk 'NF{print $1}' > $D/harness20.txt
run() { local name=$1 tag=$2; shift 2; local pa=""; for p in "$@"; do pa="$pa --pair $p"; done
  $C bash -c "$ENV; nice $PY statistics/paired_setups.py $OA --tag-file $tag --family $name --episodes $pa --out $D/$name.csv --setups-out $D/$name-setups.csv" > $D/$name.log 2>&1; echo "$name rc=$?"; }
S=rev-scicomp40; Hh=harness20
run sc40-toolkit $A/tags/scicomp40.txt $S-qwen38-c-perf-playbook-cpu,$S-qwen38-c $S-oss120b-c-perf-playbook-cpu,$S-oss120b-c $S-kimi27sglang-c-perf-playbook-cpu,$S-kimi27sglang-c \
  $S-qwen38-fortran-perf-playbook-cpu,$S-qwen38-fortran $S-oss120b-fortran-perf-playbook-cpu,$S-oss120b-fortran $S-qwen38-hip-perf-playbook-amd,$S-qwen38-hip $S-oss120b-hip-perf-playbook-amd,$S-oss120b-hip
P=(); for m in qwen38 oss120b; do for a in miniswe openhands autokernel; do P+=($Hh-$m-c-$a,$Hh-$m-c); done; done
run harness20 $D/harness20.txt "${P[@]}"
run terse-harness20 $D/harness20.txt $Hh-qwen38-c-caveman,$Hh-qwen38-c
$C bash -c "$ENV; nice $PY $X/ana/lib/all_rows.py $OBS $D" > $D/all-rows.log 2>&1; echo all-rows rc=$?
