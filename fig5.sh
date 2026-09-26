#!/usr/bin/env bash
# Figure 6: distributed ML scaling on 1-16 GPUs, speedup over PyTorch on one GPU; four operators and
# the geomean over every operator a series solved (MLP-TP left out: its grade is under review).
. "$(dirname "$0")/lib.sh"
ulimit -c 0
require "$W/mlscale16-torch.db"
"$PY" "$STATS/plot_scaling.py" "$W/mlscale16-torch.db" --experiment mlscale \
    --arm '^(mlscale-(part2-)?(qwen38|oss120b|kimi27sglang)-hip(-gemmhint|-dist-rccl-amd)?|torch_dist)$' \
    --figure mode-grid --quantity speedup \
    --kernels dist_moe_router dist_sdpa dist_all_to_all_transpose dist_softmax --print-width 5.5 \
    --out "$F/ml_scaling" --table "$T/fig5.csv"
mv -f "$F/ml_scaling-mode-grid.pdf" "$F/ml_scaling.pdf"
mv -f "$F/ml_scaling-mode-grid.png" "$F/ml_scaling.png"
