# gpu-llr-focus40 -- the same forty kernels on the GPU

Claude Code + LLM agents writing HIP, Triton (Python delivery) and C + OpenMP offload for the
`llr-focus40` kernels on the MI300A GPU, with and without the Language Skill Packet. Speedups are
over the same Numba baseline as the CPU track.

## Reproduce

- `./submit.sh` -- launch every arm (cluster).
- `./collect.sh` -- judge databases -> `data/observations.csv` (cluster).
- `./plot.sh` -- figures and per-figure tables.

`experiment.sh` excludes 631274-631277: Triton graded before the judge accepted Python, so every
row in them is a C submission.

## Open

- Kimi K2.7 arms are still running; the tables are partial until they finish.
- Triton coverage is low on every model (21-27 of 40): agents rarely land a correct kernel.
- Device residency is not enforced per submission (`languages.has_offload_entry` has no caller).
