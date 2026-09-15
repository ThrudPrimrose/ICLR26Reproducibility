#!/usr/bin/env bash
# perf-playbook: what the performance-engineering playbook packet did, on three campaigns.
#   ./reproduce.sh              data/ + the two llr-focus40 databases -> figures/ + tables/, then check checksums
#   ./reproduce.sh --extract    first rebuild data/scicomp-perf-playbook.db (CSCS cluster only)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
cd "$(dirname "$0")"
. ../common.sh
runs=${RUNS:-/capstor/scratch/cscs/ybudanaz/x86_64/hpcagent-bench-runs}
scicomp=data/scicomp-perf-playbook.db
# The two loop-level playbook arms live in the llr-focus40 campaigns, so this reads those
# experiments' own databases rather than keeping a second copy of each. Run their reproduce.sh
# --extract first if ../llr-focus40-cpu/data or ../llr-focus40-gpu/data is missing.
llr_cpu=../llr-focus40-cpu/data/llr-focus40-cpu.db
llr_gpu=../llr-focus40-gpu/data/llr-focus40-gpu.db

# 637044, 638964: still in the queue on 2026-09-15 when this snapshot was extracted; a live job's
# tasks are half-run, so each arm is read at its last finished wave. The llr-focus40 exclusions are
# stated in those two experiments' own reproduce.sh.
[[ " $* " == *" --extract "* ]] && extract "$scicomp" scicomp-perf-playbook "$runs"/scicomp-perf-playbook-2026* \
    -- 637044 638964

mkdir -p tables figures
# The roster a per-kernel figure is drawn over is NAMED, not derived from the rows (spec E1, A7).
llr_roster=$(mktemp)
scicomp_roster=$(mktemp)
trap 'rm -f "$llr_roster" "$scicomp_roster"' EXIT
roster --tag llr-focus40 > "$llr_roster"
roster --kernels-file "$HPCAGENT_BENCH/experiments/kernels-scicomp40.txt" > "$scicomp_roster"

# One paired_arms.py invocation per campaign, and so one Benjamini-Hochberg family per campaign
# (spec M1): the three campaigns run different rosters and different repeat policies, and
# correcting them together would pool tests that never shared a population.
impact() {  # impact <name> <db> <roster> <repeats> <baseline> [--include-incomplete] <pair>...
    local name=$1 db=$2 file=$3 repeats=$4 baseline=$5 pairs=()
    shift 5
    if [[ ${1:-} == --include-incomplete ]]; then pairs+=("$1"); shift; fi
    for pair in "$@"; do pairs+=(--pair "$pair"); done
    "$PY" "$HPCAGENT_BENCH/experiments/paired_arms.py" --observations "$db" --family "$name" --repeats "$repeats" \
        --baseline "$baseline" --roster-file "$file" "${pairs[@]}" --out "tables/${name}_pairs.csv" \
        --arms-out "tables/${name}_arms.csv" --impact-out "tables/${name}.csv"
}

# scicomp-focus40, 3 agents per kernel by design, so a kernel is the median over them (spec R5).
# --baseline c-autopar: this campaign grades most kernels against C -O3 + autopar but a few against
# numpy or a vendored library, and one kernel has rows of both kinds. A speed-up divided by two
# references is not one quantity (spec P1), so the reported speed-up leg is the autopar-graded
# kernels; the token leg carries no denominator and keeps every kernel.
# --include-incomplete, and the caption has to say so (spec E1): both playbook arms cover 38 of the
# 40 kernels -- nussinov and quatrex_rgf were never dispatched to them -- while both plain arms
# cover all 40. Every leg is paired, so the missing two lower n and enter no estimate.
# Kimi-K2.7-Code has no pair here: its playbook arm reached 2 of the 40 in the finished waves, and
# the wave that would have carried the rest (job 637044) was still running when this was extracted.
s=scicomp-perf-playbook
impact playbook_scicomp "$scicomp" "$scicomp_roster" median c-autopar --include-incomplete \
    "$s-qwen38-perf-playbook-cpu,$s-qwen38-plain" \
    "$s-oss120b-perf-playbook-cpu,$s-oss120b-plain"
# llr-focus40, one task per kernel, a rerun replacing the run it repeats (spec R4).
c=cpf-llr-focus40
impact playbook_llr_cpu "$llr_cpu" "$llr_roster" latest numba \
    "$c-qwen38-c-perf-playbook-cpu,$c-qwen38-c" "$c-oss120b-c-perf-playbook-cpu,$c-oss120b-c"
g=gpu-llr-focus40
impact playbook_llr_gpu "$llr_gpu" "$llr_roster" latest numba \
    "$g-qwen38-hip-perf-playbook-amd,$g-qwen38-hip" "$g-oss120b-hip-perf-playbook-amd,$g-oss120b-hip"

forest() {  # forest <name> <label>
    "$PY" "$HPCAGENT_BENCH/scripts/plot_paired_arms.py" "tables/${1}_pairs.csv" --label "$2" \
        --ratio-label "Playbook / No Packet" --out "figures/${1}_forest.pdf"
}
forest playbook_scicomp "Performance Playbook, Scientific Computing"
forest playbook_llr_cpu "Performance Playbook, Loop-Level C"
forest playbook_llr_gpu "Performance Playbook, Loop-Level HIP"

# Per kernel (spec A7): each arm's speed-up and task token total, no paired ratios and no tests.
# The pattern is both the arm selector and the (model, condition) parser; a control matches with no
# condition group, which is how this figure spells "no packet".
kernels() {  # kernels <name> <db> <roster> <pattern> <order> <repeats> <label>
    "$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$2" --roster-file "$3" \
        --arm-pattern "$4" --condition-order "$5" --repeats "$6" --label "$7" \
        --out "figures/${1}_kernels.pdf" --table "tables/${1}_kernels.csv"
}
# --include-incomplete for the same two kernels, stated in the README; kimi27sglang is excluded by
# the pattern rather than drawn as a 2-of-40 row.
"$PY" "$HPCAGENT_BENCH/scripts/plot_kernel_comparison.py" --observations "$scicomp" \
    --roster-file "$scicomp_roster" --include-incomplete --repeats median \
    --arm-pattern '^scicomp-perf-playbook-(?P<model>qwen38|oss120b)-(?:plain|(?P<condition>perf-playbook-cpu))$' \
    --condition-order ',perf-playbook-cpu' --label "Performance Playbook per Kernel, Scientific Computing" \
    --out figures/playbook_scicomp_kernels.pdf --table tables/playbook_scicomp_kernels.csv
kernels playbook_llr_cpu "$llr_cpu" "$llr_roster" \
    '^cpf-llr-focus40-(?P<model>qwen38|oss120b)-c(?:-(?P<condition>perf-playbook-cpu))?$' \
    ',perf-playbook-cpu' latest "Performance Playbook per Kernel, Loop-Level C"
kernels playbook_llr_gpu "$llr_gpu" "$llr_roster" \
    '^gpu-llr-focus40-(?P<model>qwen38|oss120b)-hip(?:-(?P<condition>perf-playbook-amd))?$' \
    ',perf-playbook-amd' latest "Performance Playbook per Kernel, Loop-Level HIP"

# The README quotes its own tables; it is regenerated from them, never edited (spec N4:
# rounding happens in printed text only).
"$PY" "$HPCAGENT_BENCH/scripts/impact_markdown.py" --readme README.md

[[ " $* " == *" --record "* ]] && check --record || check
