# llr8-kimi27sglang-c-a

Slurm job `608448` - 7 nodes - 02:52:23 - RUNNING (started 15:44:51, ended Unknown)

| | |
|---|---|
| model | `moonshotai/Kimi-K2.7-Code` |
| language | c |
| skills | off |
| problems file | `problems-llr8kimi-c-a.jsonl` |
| inference nodes | 4 |
| agent nodes | 1 |
| judge nodes | 2 |
| agents per node | 12 |
| run root | `${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hpcagent-bench-runs` |

## Submit

```bash
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.llr8-kimi27sglang-c-a)" --time=12:00:00 --job-name=llr8-kimi27sglang-c-a \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.llr8-kimi27sglang-c-a" beverin.sbatch
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

Not yet -- the arm was still running when this record was written. Regrade from the
judge shards with `collect.py` once it lands.
