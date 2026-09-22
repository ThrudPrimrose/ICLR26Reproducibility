#!/usr/bin/env bash
# llrblind: llr-focus40 on the CPU in the blind submission mode (no score tool, one submission),
# Language Skill Packet on and off. The paper's panel pairs C +skills against C within the blind mode.
#   ./reproduce.sh              data/llrblind.db -> tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild the .db from the judge databases (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
# --extract passes --regrades: the 447 unstamped 2026-09-12 rows have no stored source, cannot be
# re-timed, and are dropped (their kernels are owed reruns); the glob also picks up promotions.
cd "$(dirname "$0")"
. ../common.sh
db=data/llrblind.db
regrades=${REGRADES:-$here_common/../../audit-20260918/promote-0921/regrades/regrade-*.db}
if [[ " $* " == *" --extract "* ]]; then
    "$PY" -m hpcagent_bench.dataset --experiment llr-focus40-blind --regrades "$regrades" --out "$db"
fi

require_data "$db"
pairs=()
for model in oss120b qwen38 kimi27sglang; do
    pairs+=(--pair "llrblind-cmp-$model-c-skills,llrblind-cmp-$model-c")
done
mkdir -p tables
"$PY" "$HPCAGENT_BENCH/statistics/paired_arms.py" --observations "$db" --family llrblind-skills \
    --policy solved --repeats median --cost-model billed "${pairs[@]}" \
    --out tables/skills_billed.csv --arms-out tables/skills_billed_arms.csv
[[ " $* " == *" --record "* ]] && check --record || check
