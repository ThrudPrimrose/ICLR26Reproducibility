# llr8-oss120b-c-skills

Slurm job `608987` - 8 nodes - 00:41:21 - RUNNING (started 17:55:53, ended Unknown)

| | |
|---|---|
| model | `openai/gpt-oss-120b` |
| language | c |
| skills | on |
| problems file | `problems-llr6-c-skills.jsonl` |
| inference nodes | 1 |
| agent nodes | 1 |
| judge nodes | 6 |
| agents per node | 120 |
| run root | `${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hpcagent-bench-runs` |

## Submit

```bash
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.llr8-oss120b-c-skills)" --time=08:00:00 --job-name=llr8-oss120b-c-skills \
    --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.llr8-oss120b-c-skills" beverin.sbatch
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

Not yet -- the arm was still running when this record was written. Regrade from the
judge shards with `collect.py` once it lands.
