#!/usr/bin/env bash
# Figure 3 (what the agent is given): skill packets and the CPF on the CPU (with Pluto), skills on the GPU (with PPCG), the performance toolkit.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "$0")/lib.sh"
require "$W/llr-focus40.db" "$W/llr-focus40-blind.db" "$T/llr-cpu-packets.csv" "$T/llr-gpu-skills.csv" "$T/blind-vs-scored.csv"
# Terse joins the packet panels (its pairs are computed by stats.sh as terse-*.csv).
"$PY" - "$T" <<'PYEOF'
import sys, pandas as pd
t = sys.argv[1]
terse = pd.read_csv(f"{t}/terse-llr.csv")
pd.concat([pd.read_csv(f"{t}/llr-cpu-packets.csv"), terse[terse.arm_a.str.startswith("cpf-")]]).to_csv(f"{t}/llr-cpu-packets-terse.csv", index=False)
pd.concat([pd.read_csv(f"{t}/llr-gpu-skills.csv"), terse[terse.arm_a.str.startswith("gpu-")]]).to_csv(f"{t}/llr-gpu-skills-terse.csv", index=False)
pd.concat([pd.read_csv(f"{t}/harness20.csv"), pd.read_csv(f"{t}/terse-harness20.csv")]).to_csv(f"{t}/harness20-terse.csv", index=False)
PYEOF
"$PY" "$STATS/plot_score_change.py" "$W/llr-focus40.db" "$W/scicomp-focus40.db" \
    --mode dots --cost-model billed --include-incomplete --dots-row-height 0.8 --row-width iclr \
    --comparison "title=LLR CPU;intervention=packets;pairs=$T/llr-cpu-packets-terse.csv;comparators=$T/comparators.csv;comparator-set=pluto:C" \
    --comparison "title=LLR GPU;intervention=packets;pairs=$T/llr-gpu-skills-terse.csv;comparators=$T/comparators.csv;comparator-set=ppcg_hip:HIP" \
    --comparison "title=Perf. Toolkit;intervention=perf-playbook-cpu;pairs=$T/scicomp-toolkit.csv" \
    --out "$F/efficacy_packets_and_scope" --table "$T/fig2.csv"
