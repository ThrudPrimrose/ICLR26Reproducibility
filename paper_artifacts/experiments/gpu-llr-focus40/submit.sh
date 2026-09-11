#!/usr/bin/env bash
# Launch every arm: HIP, Triton and C + OpenMP offload, with and without the packet. Cluster only.
set -euo pipefail
opt=${OPTARENA:-${SCRATCH:?}/optarena}
cd "$opt/experiments"
export BEGIN=${BEGIN:-now} MODELS=${MODELS:-"oss120b qwen38 kimi27sglang"}
LANGUAGES=hip ./submit-gpu-llr40.sh
LANGUAGES=triton ./submit-gpu-llr40.sh
LANGUAGES=c OFFLOAD=openmp ./submit-gpu-llr40.sh
