# llr4-qwen30b-fortran

Slurm job `601852` - 10 nodes - 04:52:01 - COMPLETED

| | |
|---|---|
| model | `qwen30b` |
| language | `fortran` |
| skills | `off` |
| problems file | `problems-llr2-fortran.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=fortran ARMS=off MODEL=qwen30b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 161 / 242 |
| solved | 99 |
| solved / 242 | 40.9% |
| solved / reached | 61.5% |
| judge calls | 1300 |
| tokens | 0.39 B |
| median speedup | 1.0x |
| max speedup | 20.0x |
