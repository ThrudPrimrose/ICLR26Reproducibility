# llr4-oss120b-c-skills

Slurm job `602071` - 8 nodes - 04:03:33 - COMPLETED

| | |
|---|---|
| model | `oss120b` |
| language | `c` |
| skills | `on` |
| problems file | `problems-llr4-c-skills.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=c ARMS=skills MODEL=oss120b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 130 / 242 |
| solved | 47 |
| solved / 242 | 19.4% |
| solved / reached | 36.1% |
| judge calls | 639 |
| tokens | 1.40 B |
| median speedup | 1.0x |
| max speedup | 25.0x |
