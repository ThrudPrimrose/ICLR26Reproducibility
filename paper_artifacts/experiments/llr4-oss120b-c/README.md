# llr4-oss120b-c

Slurm job `602070` - 8 nodes - 04:29:49 - COMPLETED

| | |
|---|---|
| model | `oss120b` |
| language | `c` |
| skills | `off` |
| problems file | `problems-llr2-c.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=c ARMS=off MODEL=oss120b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 192 / 242 |
| solved | 52 |
| solved / 242 | 21.5% |
| solved / reached | 27.1% |
| judge calls | 843 |
| tokens | 1.40 B |
| median speedup | 1.0526x |
| max speedup | 100.0x |
