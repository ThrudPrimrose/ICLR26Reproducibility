# cpf-llr-focus40 -- does a packet help the agent?

Claude Code driving four locally served LLMs over the forty `llr-focus40` kernels on the CPU, one
agent per kernel, under four conditions: no packet, the Language Skill Packet, the Canonical Parallel
Form page (the CPF tool and its page), and the Canonical Parallel Form as the starting source. C runs
all four; Fortran runs the first two, since the form has no Fortran spelling.

## Reproduce

- `./submit.sh` -- launch every arm (cluster; needs the drop-in forms rendered, see the script).
- `./collect.sh` -- judge databases -> `data/observations.csv` (cluster).
- `./plot.sh` -- `data/observations.csv` -> `figures/` and one table per figure in `data/`.

`experiment.sh` holds the run roots and the jobs that do not count, with the reason for each. A wave
after the first re-runs only kernels with no judge row (`$OPTARENA/experiments/remaining_kernels.py`),
so every kernel carries one agent across all waves.

## Reading the tables

`data/<figure>.csv` are medians over kernels, per arm. Claims rest on `data/paired_*.csv`: per-kernel
ratios paired against the no-packet arm, Hodges-Lehmann with a Wilcoxon signed-rank test.

## Open

- GLM-5.3 arms have no rows: the SGLang loader fails on `format_ue8m0` (patch not reaching the job).
- GPT-OSS-120B Fortran with the packet never lands `argmax_with_index` or `tsvc_2_s4112` (six waves).
