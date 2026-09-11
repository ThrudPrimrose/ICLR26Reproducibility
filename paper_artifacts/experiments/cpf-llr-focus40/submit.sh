#!/usr/bin/env bash
# Launch every arm: 4 models x {no packet, language packet, CPF page, CPF source} on C, and
# {no packet, language packet} on Fortran. Cluster only. Needs the drop-in forms rendered first:
#   CPF_DROPIN=1 $OPTARENA/experiments/prerender_cpf.sh outer $CPF_FORMS_DIR <kernels> $OPTARENA cpu
set -euo pipefail
opt=${OPTARENA:-${SCRATCH:?}/optarena}
cd "$opt/experiments"
export BEGIN=${BEGIN:-now} MODELS=${MODELS:-"oss120b qwen38 kimi27sglang glm53"}
export ARMS=${ARMS:-"c:plain c:skills c:cpf c:cpfsrc fortran:plain fortran:skills"}
export CPF_FORMS_DIR=${CPF_FORMS_DIR:-$SCRATCH/cpf-dropin-cpu-llr-focus40}
export CPF_DROPIN_DIR=${CPF_DROPIN_DIR:-$SCRATCH/cpf-dropin-cpu-llr-focus40-frozen}
./submit-cpf-llr40.sh
