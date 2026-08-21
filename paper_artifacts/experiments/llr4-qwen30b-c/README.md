# llr4-qwen30b-c

Slurm job `601850` - 10 nodes - 06:12:10 - COMPLETED

| | |
|---|---|
| model | `qwen30b` |
| language | `c` |
| skills | `off` |
| problems file | `problems-llr2-c.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=c ARMS=off MODEL=qwen30b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 222 / 242 |
| solved | 162 |
| solved / 242 | 66.9% |
| solved / reached | 73.0% |
| judge calls | 2059 |
| tokens | 0.59 B |
| median speedup | 1.0x |
| max speedup | 100.0x |
