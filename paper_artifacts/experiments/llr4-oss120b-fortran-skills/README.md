# llr4-oss120b-fortran-skills

Slurm job `602073` - 8 nodes - 04:12:05 - COMPLETED

| | |
|---|---|
| model | `oss120b` |
| language | `fortran` |
| skills | `on` |
| problems file | `problems-llr4-fortran-skills.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=fortran ARMS=skills MODEL=oss120b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 128 / 242 |
| solved | 36 |
| solved / 242 | 14.9% |
| solved / reached | 28.1% |
| judge calls | 527 |
| tokens | 1.49 B |
| median speedup | 1.0x |
| max speedup | 14.2857x |
