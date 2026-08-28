# data-llr8w2 -- C vs Fortran, skills on/off

Collected 2026-08-29 from `$SCRATCH/hpcagent-bench-runs/llr8w1-20260827`, after every arm landed.

    python3 collect.py --run-root $SCRATCH/hpcagent-bench-runs/llr8w1-20260827 \
                       --campaign llr8w2 --out data-llr8w2
    python3 analysis/optimizations_llr8w2.py --run-root $SCRATCH/hpcagent-bench-runs/llr8w1-20260827 \
                       --out data-llr8w2
    python3 plot_llr8w2.py

| job | arm | state | elapsed |
|---|---|---|---|
| 610669 | llr8w2-qwen38-c | COMPLETED | 06:56:32 |
| 610671 | llr8w2-qwen38-c-skills | COMPLETED | 07:09:31 |
| 610670 | llr8w2-qwen38-fortran | COMPLETED | 07:33:45 |
| 610672 | llr8w2-qwen38-fortran-skills | COMPLETED | 07:40:20 |
| 610668 | llr8w2-oss120b-fortran | COMPLETED | 02:31:59 |
| 610653 | llr8w2-oss120b-fortran-skills | COMPLETED | 04:06:04 |
| 610662 | llr8w2-kimi27sglang-c-skills-r1 | COMPLETED | 08:14:23 |

`oss120b-c` and its skilled twin are ABSENT: those arms failed and were never re-run, so there is
no C pair for oss120b. `kimi27sglang-c` (unskilled) failed too, which leaves the kimi arm without a
pair -- it appears in `summary.csv` and in the cost figure, and cannot appear in `matched.csv`.

## Read `matched.csv`, not `summary.csv`, for the skills effect

Both are here and they DISAGREE on C, which is the point. The arms reached different kernels (29 vs
34 on C, 31 vs 35 on qwen Fortran), so an arm-level geomean is taken over a different problem set
per arm and its skills difference mixes the effect with which problems each arm got to. `matched.csv`
pairs on kernels BOTH arms timed and is the only valid comparison of the two.

| model | language | paired n | geomean off | geomean skills | ratio | sign test |
|---|---|---|---|---|---|---|
| qwen38 | c | 14 | 18.31 | 16.83 | **0.92** | 2W/10L, p=0.039 |
| qwen38 | fortran | 14 | 4.90 | 8.34 | 1.70 | 9W/5L, p=0.42 |
| oss120b | fortran | 20 | 3.85 | 4.22 | 1.10 | 10W/9L, p=1.00 |

Arm-level, the same C pair reads 8.66 -> 11.46, i.e. skills LOOK like a 32% gain. Paired, they are a
significant 8% LOSS. The unskilled C arm reached fewer and harder kernels; the arm-level figure is
reporting that, not the packet.

## Optimiser label

`.env.llr8-qwen38-c` and its skilled twin set `OPTARENA_OPTIMIZER=openai/gpt-oss-120b` while serving
`VLLM_MODEL=Qwen/Qwen3.8-27B-FP8`, so the `optimizer` column of those two shards names the wrong
model. The Slurm log for 610669 and 610670 shows `Qwen3.8-27B-FP8` served under
`served_model_name='optarena-vllm'` in both, so the label is cosmetic and all four qwen arms really
are one model. `model` in these CSVs comes from `collect.py`'s arm registry and is correct; the raw
`optimizer` string in the shard is not.

## Foreign rows

Seven `run_id='adhoc'` submissions sit in the wave-2 shards, one of them a 24x outlier, from a
hand-run probe that shared the results directory. `collect.py` now selects rows by the arm's own
`run_id` pattern, so they are excluded; a collection made before that fix credited them to whichever
arm's job wrote the file.
