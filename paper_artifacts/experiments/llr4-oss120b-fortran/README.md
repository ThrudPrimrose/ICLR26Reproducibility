# llr4-oss120b-fortran

Slurm job `602072` - 8 nodes - 04:10:14 - COMPLETED

| | |
|---|---|
| model | `oss120b` |
| language | `fortran` |
| skills | `off` |
| problems file | `problems-llr2-fortran.jsonl` (242 kernels, loop_level_reasoning) |
| agents | 80 (1 agent node x 80) |

## Submit

```bash
cd containers/cluster/example-script
LANGS=fortran ARMS=off MODEL=oss120b ./submit-llr4.sh
```

`arm.env` is the exact configuration this run used. The skills packet is **inlined into the
problems file at generation time**, so a run never re-reads the SKILL.md pages; regenerate the
jsonl after editing a page or the arm measures the old text.

## Result

| metric | value |
|---|---|
| problems reached | 132 / 242 |
| solved | 35 |
| solved / 242 | 14.5% |
| solved / reached | 26.5% |
| judge calls | 549 |
| tokens | 1.30 B |
| median speedup | 1.0x |
| max speedup | 9.0909x |
