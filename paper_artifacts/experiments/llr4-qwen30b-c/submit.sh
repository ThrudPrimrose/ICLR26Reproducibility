#!/usr/bin/env bash
# Exact submission for llr4-qwen30b-c (Slurm job 601850).
# Run from containers/cluster/example-script in the HPCAgent-Bench tree.
# submit-llr4.sh reads .env.llr4-qwen30b-c and derives --nodes from
# INFERENCE_NODES + AGENT_NODES + JUDGE_NODES; never pass --nodes yourself.
set -euo pipefail
LANGS=c ARMS=off MODEL=qwen30b ./submit-llr4.sh
