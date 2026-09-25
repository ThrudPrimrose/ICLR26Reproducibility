# daint-transfer

The final LLR40 answers (C, Fortran, HIP, Triton) graded a second time on CSCS Daint GH200 nodes
(72 Grace cores, one H100 per grade), against the parallel Numba baseline re-timed there, and
compared with their MI300A (Beverin) grade.

| path | what |
|---|---|
| `data/gh200/` | the Daint regrade as delivered on 2026-09-25: per-rank `regrade-cells-0.db`, the authoritative `final_worklists/*/{worklist,skipped}.jsonl`, `exclude_rows.jsonl`, the harness changes in `changes/` and the operator's `NOTES_daint_regrade.md` |
| `data/transfer.csv` | one row per answer: GH200 grade (`s_bar_gh200`, `status_gh200` = graded / error / not-portable) beside its MI300A final grade (`s_bar_mi300a`, NaN where the answer has only its live grade `original_speedup`) |
| `collect.py` | builds `data/transfer.csv` from `data/gh200/` and the Beverin final-grade waves |

Notes:
- HIP is built through HIP's CUDA backend (ROCm 7.2.4 headers), not hipify.
- C ran on the `debug` partition (same GH200 nodes, 30 min jobs, resumed). Items graded on both partitions agree within run-to-run noise.
- An answer that uses x86-only or AMD-only code is skipped as not portable and is never graded as wrong: 225 of 789 C answers, 109 of 437 HIP answers and 7 of 368 Triton answers.

Summary (all models pooled; MI300A final grade, live grade where no final grade exists):

| backend | answers | not portable | still correct | GH200/MI300A speed-up (geomean) | Spearman rho, per kernel |
|---|---:|---:|---:|---:|---:|
| C | 789 | 225 (29%) | 97.9% | 1.60 | 0.92 |
| Fortran | 72 | 0 | 93.1% | 1.37 | 0.90 |
| HIP | 437 | 109 (25%) | 93.8% | 1.16 | 0.84 |
| Triton | 368 | 7 (2%) | 93.4% | 1.08 | 0.92 |

Per cell, the C answers run 1.75x faster on GH200 than on MI300A, while the Numba baseline runs only 1.1x
faster. That is why C speed-ups are higher on GH200.
