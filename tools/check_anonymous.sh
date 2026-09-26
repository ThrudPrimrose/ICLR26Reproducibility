#!/usr/bin/env bash
# Re-check a package (zip or directory) against .anonymize-terms.txt and the commit ids of this repository
# (and of the benchmark checkout, when given); prints "clean" or every leak by file and term line.
#   tools/check_anonymous.sh <package.zip | dir> [hpcagent-bench checkout]
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:?usage: tools/check_anonymous.sh <package.zip | dir> [hpcagent-bench checkout]}
if [[ -f $target ]]; then
    dir=$(mktemp -d)
    trap 'rm -rf "$dir"' EXIT
    unzip -q "$target" -d "$dir"
    target=$dir
fi
if find "$target" -name .git -print -quit | grep -q .; then echo "LEAK: a .git in the package" >&2; exit 1; fi
# The one exempt literal: the DaCe dependency URL make_zenodo.sh keeps so the package installs.
keep=$(grep -ohE 'git\+https://github\.com/[^"]*/dace\.git@[0-9a-f]{40}' "$target"/*/pyproject.toml 2>/dev/null | head -1 || true)
python3 "$root/tools/anonymize.py" --terms "$root/.anonymize-terms.txt" --commits "$root" \
    ${2:+--commits "$2"} ${keep:+--keep "$keep"} --check "$target"
