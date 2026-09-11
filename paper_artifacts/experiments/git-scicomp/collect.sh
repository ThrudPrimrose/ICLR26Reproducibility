#!/usr/bin/env bash
# Judge databases -> data/observations.csv. Cluster only.
set -euo pipefail
cd "$(dirname "$0")"
. ./experiment.sh
../collect_campaign.sh data/observations.csv "${roots[@]}" -- "${excluded[@]}"
