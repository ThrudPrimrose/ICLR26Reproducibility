#!/usr/bin/env bash
# Exact submission for llr8-oss120b-c.
# Run from containers/cluster/example-script in the HPCAgent-Bench tree.
# arm_nodes reads the same env file the launcher does, so the two can never disagree;
# never pass --nodes yourself. No --account: beverin schedules every account alike.
set -euo pipefail
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.llr8-oss120b-c)" --time=08:00:00 --job-name=llr8-oss120b-c \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.llr8-oss120b-c" beverin.sbatch
