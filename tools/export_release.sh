#!/usr/bin/env bash
# Copy this repository's tracked files into a benchmark checkout as reproducibility/paper/, the
# folder the paper's Appendix G names. Local cluster settings (tools/cluster.env) never leave.
#   tools/export_release.sh <hpcagent-bench checkout>
set -euo pipefail
bench=${1:?usage: tools/export_release.sh <hpcagent-bench checkout>}
root=$(cd "$(dirname "$0")/.." && pwd)
dest=$bench/reproducibility/paper
mkdir -p "$dest"
(cd "$root" && git ls-files -z | grep -zv '^tools/cluster.env$' | rsync -a --from0 --files-from=- ./ "$dest/")
echo "exported $(cd "$root" && git ls-files | wc -l) files to $dest"
