#!/usr/bin/env bash
# Exact submission for llr4-oss120b-fortran (Slurm job 602072).
# Run from containers/cluster/example-script in the HPCAgent-Bench tree.
# submit-llr4.sh reads .env.llr4-oss120b-fortran and derives --nodes from
# INFERENCE_NODES + AGENT_NODES + JUDGE_NODES; never pass --nodes yourself.
set -euo pipefail
LANGS=fortran ARMS=off MODEL=oss120b ./submit-llr4.sh
