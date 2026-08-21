# llr4-qwen30b-c-skills

Slurm job `601851` - 10 nodes - 06:10:33 - COMPLETED

| | |
|---|---|
| model | `qwen30b` |
| language | `c` |
| skills | `on` |
| problems file | `problems-llr4-c-skills.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=c ARMS=skills MODEL=qwen30b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 237 / 242 |
| solved | 177 |
| solved / 242 | 73.1% |
| solved / reached | 74.7% |
| judge calls | 2150 |
| tokens | 0.93 B |
| median speedup | 1.0x |
| max speedup | 25.0x |
