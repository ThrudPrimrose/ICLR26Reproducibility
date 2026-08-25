# llr8-oss120b-c

Slurm job `608447` - 8 nodes - 01:05:44 - COMPLETED (started 15:44:51, ended 16:50:35)

| | |
|---|---|
| model | `openai/gpt-oss-120b` |
| language | c |
| skills | off |
| problems file | `problems-llr6-c.jsonl` |
| inference nodes | 1 |
| agent nodes | 1 |
| judge nodes | 6 |
| agents per node | 120 |
| run root | `${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hpcagent-bench-runs` |

## Submit

```bash
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.llr8-oss120b-c)" --time=08:00:00 --job-name=llr8-oss120b-c \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.llr8-oss120b-c" beverin.sbatch
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

See `data*/summary.csv` for the row named after this arm.
