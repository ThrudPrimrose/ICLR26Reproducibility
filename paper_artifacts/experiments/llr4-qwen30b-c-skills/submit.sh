#!/usr/bin/env bash
# Exact submission for llr4-qwen30b-c-skills (Slurm job 601851).
# Run from containers/cluster/example-script in the HPCAgent-Bench tree.
# submit-llr4.sh reads .env.llr4-qwen30b-c-skills and derives --nodes from
# INFERENCE_NODES + AGENT_NODES + JUDGE_NODES; never pass --nodes yourself.
set -euo pipefail
LANGS=c ARMS=skills MODEL=qwen30b ./submit-llr4.sh
