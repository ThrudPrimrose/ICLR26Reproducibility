#!/usr/bin/env bash
# Every experiment under this directory, one status line each; non-zero exit if any failed.
#   ./reproduce_all.sh [--extract] [--record]     arguments are passed to each reproduce.sh
# This file is kept IDENTICAL in ICLR26Reproducibility and mpr-artifacts.
set -uo pipefail
cd "$(dirname "$0")"

failed=0
for script in */reproduce.sh; do
    name=$(dirname "$script")
    "./$script" "$@" && status=0 || status=$?
    if ((status == 0)); then
        echo "ok    $name"
    else
        echo "FAIL  $name (exit $status)"
        failed=$((failed + 1))
    fi
done

((failed == 0)) && echo "all experiments reproduced" || echo "$failed experiment(s) failed"
((failed == 0))
