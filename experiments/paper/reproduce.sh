#!/usr/bin/env bash
# paper: every figure of the paper (Figs 2-5 and appendix Fig 7), from the experiments' observations.
#   ./reproduce.sh              data/ -> work/ (pooled) -> tables/ + figures/, then check checksums
#   ./reproduce.sh --extract    first rebuild data/ from a mirror of the cluster runs (MIRROR=...)
#   ./reproduce.sh --record     rewrite SHA256SUMS instead of checking it
#   PAPER=<paper checkout> ./reproduce.sh   also copy the figures into the paper and check their widths
#
# Fig 2 also draws the polyhedral compilers (Pluto on the CPU, PPCG on the GPU) from the canon sweep's
# data/canon.db (comparators.py).
#
# Data rules (2026-09-25): every latest answer counts, at its final 4x5 grade where it has one and
# at its live grade otherwise (pool.py); SciComp counts the 35 kernels of kernels-scicomp35.txt;
# git vs. kernel answers the final grade could not time, or failed on the warpx tolerance, count at
# their live grade until regraded (pool.py --git-correct).
cd "$(dirname "$0")"
. ../common.sh

EXPERIMENTS=(llr-focus40 llr-focus40-blind git-scicomp scicomp-focus40 harness20 mlscale)
M3=(qwen38 oss120b kimi27sglang)
W=work T=tables F=figures
mkdir -p data "$W" "$T" "$F"

if [[ " $* " == *" --extract "* ]]; then
    # MIRROR holds hpcagent-bench-runs/, frozen-observations/ (rows of deleted jobs) and the
    # final-grade regrade waves hb/experiments/mwd-final-regrades-v*/, oldest first (later wins).
    : "${MIRROR:?set MIRROR to the local mirror of the cluster runs}"
    runs=$MIRROR/hpcagent-bench-runs
    cp "$MIRROR/canon-central/canon.db" data/canon.db
    regrades=()
    while IFS= read -r wave; do regrades+=(--regrades "$wave/*"); done \
        < <(find "$MIRROR/hb/experiments" -maxdepth 1 -type d -name 'mwd-final-regrades-v*' | sort -V)
    for exp in "${EXPERIMENTS[@]}"; do
        if [[ $exp == mlscale ]]; then
            # The distributed track is not a dataset.py experiment: extract its jobs directly.
            jobs=()
            for job in "$runs"/mlscale-2026*/[0-9]*; do [[ -d $job ]] && jobs+=(--runs "$job"); done
            tmp=$(mktemp -d)
            "$PY" "$HPCAGENT_BENCH/reproducibility/llr40/extract_llr40.py" "${jobs[@]}" \
                --benchmarks "$HPCAGENT_BENCH/hpcagent_bench/benchmarks" --out "$tmp" --no-sources --db data/mlscale.db
            rm -rf "$tmp"
            continue
        fi
        (cd "$HPCAGENT_BENCH" && "$PY" -m hpcagent_bench.dataset --experiment "$exp" --out "$OLDPWD/data/$exp.db" \
            --runs-root "$runs" --frozen-observations "$MIRROR/frozen-observations" "${regrades[@]}")
    done
fi

for exp in "${EXPERIMENTS[@]}"; do require_data "data/$exp.db"; done
require_data data/canon.db "the canon sweep is copied from the mirror by --extract"

for db in llr-focus40 llr-focus40-blind harness20; do "$PY" pool.py "data/$db.db" "$W/$db.db"; done
"$PY" pool.py data/scicomp-focus40.db "$W/scicomp-focus40.db" --roster kernels-scicomp35.txt
"$PY" pool.py data/git-scicomp.db "$W/git-scicomp.db" --git-correct
# The GEMM-based operators are left out until the setups told to tile or call a BLAS finish.
GEMM_OPS=(dist_gemm_add_relu dist_gemm_gn_swish dist_matmul_gelu_softmax dist_matmul_large_k dist_sdpa)
# Each model's no-packet arm reports the better of its two prompts per operator (the GEMM-hint
# prompt adds that the score compares against tuned PyTorch).
"$PY" torch_anchor.py data/mlscale.db "$W/mlscale-torch.db" --drop "${GEMM_OPS[@]}" \
    --best-of mlscale-oss120b-hip-gemmhint=mlscale-oss120b-hip mlscale-qwen38-hip-gemmhint=mlscale-qwen38-hip
"$PY" comparators.py data/canon.db "$W/llr-focus40.db" --out "$T/comparators.csv"
"$PY" comparator_ratios.py "$W/llr-focus40.db" "$T/comparators.csv" --out "$T/comparator_ratios.csv"

# Arms with at least one graded answer: a pair whose arm has none yet is skipped, not drawn empty.
"$PY" - "$W" <<'EOF'
import pathlib, sqlite3, sys
work = pathlib.Path(sys.argv[1])
arms = set()
for db in ("llr-focus40", "llr-focus40-blind", "git-scicomp", "scicomp-focus40", "harness20"):
    rows = sqlite3.connect(work / f"{db}.db").execute(
        "select distinct arm from observations where record = 'submission' and speedup > 0")
    arms |= {r[0].removesuffix("-clean") for r in rows}
(work / "graded-arms.txt").write_text("\n".join(sorted(arms)) + "\n")
EOF

# family <name> <db...> -- <TREATED,CONTROL pair...> [-- extra paired_arms args]
# Every family is corrected (Benjamini-Hochberg) over its own pairs. One family = one panel: exactly
# the tests that panel draws, never a test it does not draw and never two panels.
family() {
    local name=$1 dbs=() pairs=() extra=(); shift
    while (($#)) && [[ $1 != -- ]]; do dbs+=(--observations "$1"); shift; done; shift
    while (($#)) && [[ $1 != -- ]]; do
        if grep -qx "${1%%,*}" "$W/graded-arms.txt" && grep -qx "${1#*,}" "$W/graded-arms.txt"; then
            pairs+=(--pair "$1")
        else
            echo "skip $1 (no graded answer yet)"
        fi
        shift
    done
    (($#)) && { shift; extra=("$@"); }
    ((${#pairs[@]})) || { echo "skip family $name (no pair)"; return 0; }
    "$PY" "$HPCAGENT_BENCH/statistics/paired_arms.py" "${dbs[@]}" "${pairs[@]}" --family "$name" \
        --cost-model billed --include-incomplete --out "$T/$name.csv" --arms-out "$T/${name}_arms.csv" "${extra[@]}"
}

L=$W/llr-focus40.db B=$W/llr-focus40-blind.db G=$W/git-scicomp.db S=$W/scicomp-focus40.db H=$W/harness20.db
# Fig 2's LLR CPU panel: the skill packets and CPF as source against plain C.
p=(); for m in "${M3[@]}"; do p+=("cpf-llr-focus40-$m-c-skills,cpf-llr-focus40-$m-c"); done
for m in "${M3[@]}"; do p+=("cpf-llr-focus40-$m-c-cpfsrc-v2,cpf-llr-focus40-$m-c"); done
family llr-cpu-packets "$L" -- "${p[@]}"
p=(); for m in "${M3[@]}"; do for d in hip c-openmp-device triton-device; do
    p+=("gpu-llr-focus40-$m-$d-skills,gpu-llr-focus40-$m-$d"); done; done
family llr-gpu-skills "$L" -- "${p[@]}"
p=(); for m in "${M3[@]}"; do for d in c fortran hip; do
    [[ $m == kimi27sglang && $d == hip ]] && continue
    p+=("llrblind-cmp-$m-$d-skills,llrblind-cmp-$m-$d"); done; done
family blind-skills "$B" -- "${p[@]}"
# Fig 2 (iii): blind submission against the scored run, the intervention itself, on both devices.
p=(); for m in "${M3[@]}"; do for d in c fortran; do p+=("llrblind-cmp-$m-$d,cpf-llr-focus40-$m-$d"); done; done
for m in qwen38 oss120b; do p+=("llrblind-cmp-$m-hip,gpu-llr-focus40-$m-hip"); done
family blind-vs-scored "$L" "$B" -- "${p[@]}"
p=(); for m in "${M3[@]}"; do p+=("git-scicomp-$m-repo,git-scicomp-$m-kernel"); done
family repo-vs-kernel "$G" -- "${p[@]}" -- --repeats median
p=(); for m in "${M3[@]}"; do p+=("scicomp-perf-playbook-$m-perf-playbook-cpu,scicomp-perf-playbook-$m-plain"); done
family scicomp-toolkit "$S" -- "${p[@]}"
p=(); for m in qwen38 oss120b; do for h in miniswe openhands; do p+=("harness20-$m-$h,harness20-$m-claude"); done; done
for m in qwen38 oss120b; do p+=("harness20-$m-claude-autokernel,harness20-$m-claude"); done
family harness20 "$H" -- "${p[@]}"

plot=$HPCAGENT_BENCH/statistics
dots=(--mode dots --cost-model billed --include-incomplete --dots-row-height 0.8)
# Compilers and frameworks beside the models on Fig 2's speed-up and solved rows.
cpu_cmp="comparators=$T/comparators.csv;comparator-set=pluto:C"
gpu_cmp="comparators=$T/comparators.csv;comparator-set=ppcg_hip:HIP"
# Fig 2: skill packets on the loop-level track, three panels at full text width.
"$PY" "$plot/plot_score_change.py" "$L" "$B" "${dots[@]}" --row-width iclr \
    --comparison "title=LLR CPU;intervention=packets;pairs=$T/llr-cpu-packets.csv;$cpu_cmp" \
    --comparison "title=LLR GPU;intervention=lang-skills;pairs=$T/llr-gpu-skills.csv;$gpu_cmp" \
    --comparison "title=LLR Blind;intervention=no-score-tool;pairs=$T/blind-vs-scored.csv;control-label=Scored" \
    --out "$F/efficacy_packets_and_scope" --table "$T/fig2.csv"
# Fig 3: the task's form, the harness and the performance toolkit, three panels at full text width.
"$PY" "$plot/plot_score_change.py" "$G" "$H" "$S" "${dots[@]}" --row-width iclr \
    --comparison "title=Git vs. Kernel;intervention=repo;pairs=$T/repo-vs-kernel.csv;control-label=Kernel" \
    --comparison "title=Harness20;intervention=harness;pairs=$T/harness20.csv;control-label=Claude Code" \
    --comparison "title=Perf. Toolkit;intervention=perf-playbook-cpu;pairs=$T/scicomp-toolkit.csv" \
    --out "$F/scope_row" --table "$T/fig3.csv"
# Fig 4: what each treatment costs under each token weighting, over the same pooled answers as Fig 2-3.
"$PY" "$plot/plot_cost_weighting.py" "$H" "$L" \
    --pair "harness20-qwen38-openhands,harness20-qwen38-claude,OpenHands" \
    --pair "cpf-llr-focus40-qwen38-c-cpfsrc-v2,cpf-llr-focus40-qwen38-c,CPF" \
    --pair "gpu-llr-focus40-oss120b-hip-skills,gpu-llr-focus40-oss120b-hip,HIP Skills" \
    --width 1.75 --out "$F/cost_weighting" --table "$T/fig4.csv"
# Fig 5: distributed ML scaling, speed-up over PyTorch on one GPU.
"$PY" "$plot/plot_scaling.py" "$W/mlscale-torch.db" --experiment mlscale- --arm '^mlscale-(qwen38|oss120b)-hip(-dist-rccl-amd)?(-clean)?$' --figure mode-grid --quantity speedup \
    --kernels dist_layer_norm dist_cross_entropy dist_softmax --print-width 5.5 \
    --out "$F/ml_scaling" --table "$T/fig5.csv"
mv -f "$F/ml_scaling-mode-grid.pdf" "$F/ml_scaling.pdf"
mv -f "$F/ml_scaling-mode-grid.png" "$F/ml_scaling.png"
# Fig 8 (appendix): per-kernel speed-up of the LLR answers, the final answers the pair tables use.
"$PY" ../llr-cheating/plot_cheating_per_kernel.py --db "$L" --out "$F/cheating_per_kernel.pdf"

if [[ -n ${PAPER:-} ]]; then
    for name in efficacy_packets_and_scope scope_row cost_weighting ml_scaling cheating_per_kernel; do
        cp "$F/$name.pdf" "$PAPER/figures/$name.pdf"
    done
    "$PY" "$plot/check_paper_figures.py" "$PAPER" --ignore agentbench_v11_is.pdf
fi

[[ " $* " == *" --record "* ]] && check --record || check
