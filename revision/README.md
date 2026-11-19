# The revision (v2) of the paper

This folder holds the measurements, scripts and outputs behind the revised paper. The revision reruns every
experiment of the comparison tables under one limit per episode (50M total tokens, 12M output tokens, 16 h; see
Table 2) and adds an A/A study. The v1 pipeline in the parent folder stays as it was; the CPF comparison keeps its
v1 runs and v1 rows. Paths are anonymized like the rest of the package.

## Folders

| Path | Holds |
|---|---|
| `data/raw/` | the raw run data: one results database per job, the job's node and plan files, and the regrade databases |
| `data/merged/` | observations extracted from `data/raw/` with the bench's extractor (`obs/`), the paired tables per experiment, the run and job inventories, `STATUS.tsv` (one line per extraction and table step) and `CHANGES.txt` |
| `data/obs-final/` | the observation databases the paper's tables and figures read (see below) |
| `data/xfinal/` | `obs-final-all.db`: every setup of the reused November database (Harness20, Terse, blind open-submission arms) |
| `data/dataset-harness20-final.db` | the Harness20 dataset of the rerun |
| `data/pairs50-1125/`, `data/pairs48/` | paired comparisons, one CSV per family with a `.settings.json` sidecar each |
| `data/allrows-1050/` | the pooled rows before the toolkit's Fortran and HIP pairs were added |
| `data/sec45/` | the numbers of Section 4.5: model agreement, model gaps, HIP against Triton, agents against Pluto and PPCG |
| `data/fig34-50m/` | inputs (`in/`), driver (`run.sh`, `in/figures.py`) and outputs (`out/`) of Figures 4 and 5 |
| `data/fig7/` | input (`all-obs.db`) and outputs (`ngB3g*`) of Figure 7 |
| `data/temperature3.db`, `data/repeat3-ml-obs-1130.csv`, `data/repeat3-promoted/` | the A/A study: the temperature study's results database, the `gelu` observations of the ML repeat study, and the run-to-run noise with the 31 promoted repeat-study answers |
| `data/comparators-rerun-1114.csv` | Pluto and PPCG speedups of the comparator rerun |
| `scripts/` | every script that turns `data/` into the tables and figures |
| `figures/` | the paper's figures (PDF) |
| `tables/` | the paper's tables (LaTeX) |
| `SHA256SUMS` | checksums of `figures/`, `tables/` and `scripts/` |
| `DATA_SHA256SUMS` | checksums of `data/`, one line per file |

`data/obs-final/` and `data/merged/obs/` come from the same runs. They agree on every setup except the A/A study's
`rev-repeat3-ml-feedback-*`, where `obs-final/` already holds the promoted answers (`data/repeat3-promoted/` adds
them to `merged/`), and in 69 rows whose grade fields differ (a min-of-k fallback reason). The tables and figures below
were computed from `obs-final/`.

### Raw run data

`data/raw/` is the run bundle of the revision. Its run roots:

| Run root | Experiment |
|---|---|
| `rev-llr40-20261114`, `rev-llr40-20261116` | LLR-Focus40: language packets and the toolkit on CPU and GPU |
| `llr40-blind-20261114`, `llr40-blind-20261116` | Blind vs. Open and Blind vs. Single on Focus40 |
| `rev-scicomp40-20261115` | SciComp40: the toolkit and Blind vs. Single |
| `rev-gitscicomp40-20261115` | Git framing |
| `rev-mlscale20-20261115` | MLScale: 1 to 16 GPUs |
| `rev-repeat3-ml-feedback-20261117` | the ML repeat study (A/A, `gelu`) |
| `harness20-20261114`, `-20261115`, `-20261116` | Harness20 reruns |
| `regrade-nov-20261113/final.db` | the November Harness20 episodes (160) and the LLR-Focus40 Terse and blind arms (445), all within the limit; over-limit episodes truncated |

Every row ran at 50M total tokens, 12M output tokens and 16 h. Two exceptions: the 160 November Harness20 episodes
ran at 48M total and 20M output tokens (6 stopped at that limit), and Kimi-K2.7 on SciComp40 had 32 h (29
episodes). The regrades under `data/raw/regrades/` replace grades as follows:

| Experiment | Regrade |
|---|---|
| MLScale, 1 node | `mlscale20-1118/out-abi-1118` and `out-abi-retry`; the OSS-120B `dist_adamw_zero` grade of job 742814 comes from `out-adamw-nows` |
| MLScale, 16 GPUs | `mlscale20-1118-p16/out`, with `out-taint` replacing 19 items |
| SciComp40 and Git framing | `rg227-1117/out2` |
| `lstm` | `lstm-1119/out2` |
| A/A promotions (31) | `promote-repeat3-1119/out-e00a` |

The analysis excludes eight Git-framing jobs (75284545, 75284546, 75284552 to 75284555, 75377503, 75377504); two
cut jobs (75284516, 75284524) count with their best score.

## Which output feeds which table and figure

| Paper | Output | Made by | From |
|---|---|---|---|
| Table 2 | `tables/experiments.tex` | written by hand | `data/merged/inventory/`, `data/merged/job-inventory.tsv` |
| Table 12 (A/A study) | `tables/repeat_groups.tex` | `kmp`, `addusxx_g`, `scan_affine_decay`: from `data/temperature3.db`; `gelu`: `data/repeat3-promoted/r3.sh` (`repeat3.py`) | `data/temperature3.db`, `data/merged/obs/`, `data/repeat3-promoted/promotions-repeat3-1119.csv` |
| Table 17 (all comparisons) | `tables/all_comparisons.tex` | `scripts/gen_table18.py` | `data/pairs50-1125/` with `data/pairs48/` on top (Harness20, Terse on Harness20, toolkit), `data/merged/mlscale20/mlscale20-rccl.csv` |
| Table 18 (pooled) | `tables/pooled.tex` | `scripts/gen_table19.py` | `data/pairs48/all-rows.csv` (`scripts/tables48.sh`, `scripts/lib/all_rows.py`) |
| Figure 2 (A/A study) | `figures/aa_four_kernels_api.pdf` | `scripts/aa-api/draw.py` | `data/temperature3.db`, `data/repeat3-ml-obs-1130.csv` |
| Figure 4 | `figures/figure3_packets_and_scope.pdf` | `data/fig34-50m/run.sh` | `data/obs-final/`, `data/xfinal/`, `data/pairs50-1125/`, `data/fig34-50m/in/` |
| Figure 5 | `figures/figure4_scope_row.pdf` | `data/fig34-50m/run.sh` | as Figure 4 |
| Figure 6 | `figures/figure5_cost_weighting.pdf` | `scripts/fig-appendix-50m/` (`figure5_numbers.py`, `figure5_rerun.py`, `v1/figure5.py`) | `data/obs-final/`, `data/xfinal/`, `data/fig34-50m/in/llr-focus40.db` |
| Figure 7 | `figures/ml_scaling.pdf` | `scripts/drawB3g.sh` | `data/fig7/all-obs.db` |
| Figure 8 | `figures/cheating_per_kernel.pdf` | `scripts/fig-appendix-50m/cheating_per_kernel.py` | `data/obs-final/`, `data/merged/tags/llr40.txt` |
| Section 4.5 | `data/sec45/*.csv` | `hpcagent-bench/statistics/revision/sec45.sh` | `data/obs-final/`, `data/fig34-50m/in/comparators.csv` |
| (not printed) | `figures/temperature3_t0_t1.pdf` | `hpcagent-bench/statistics/plot_temperature.py` | `data/temperature3.db` |

The script names count tables as an earlier draft did: `gen_table18.py` writes the rows of Table 17 and
`gen_table19.py` those of Table 18. `figures/figure3_*`, `figure4_*` and `figure5_*` are Figures 4, 5 and 6 of
the revision. `tables/` also holds the paper's other tables, which the revision did not recompute. The CPF rows of
Tables 17 and 18 are the v1 rows; the two scripts copy them verbatim.

## Rerun order

The scripts ran in the judge container with the `hpcagent-bench` checkout as working directory and on
`PYTHONPATH`. Each reads `X` (the analysis directory, laid out as `X/out/<stamp>/obs/`), `JUDGE_SIF` (the container
image) and `PY` (its Python). The same scripts, cleaned up, are in `hpcagent-bench/statistics/revision/`, whose
README gives this order:

1. `scripts/xfinal.sh`: extracts every setup of the reused November database into `data/xfinal/obs-final-all.db`.
2. `scripts/pairs50.sh`, `pairs50b.sh`, `pairs50c.sh`: one `statistics/paired_setups.py` family per experiment,
   one row per comparison of Table 17, into `data/pairs50-1125/`.
3. `scripts/tables48.sh`: the toolkit with its Fortran and HIP pairs, Harness20 from
   `dataset-harness20-final.db`, and `scripts/lib/all_rows.py` (every pooled row of Table 18, Benjamini-Hochberg
   over its rows per measure), into `data/pairs48/`. `scripts/all_rows.py` is the same without the Fortran and HIP
   toolkit pairs (`data/allrows-1050/`).
4. `scripts/gen_table18.py PAIRS_DIR RCCL_CSV` and `scripts/gen_table19.py all-rows.csv`: the LaTeX rows of
   Tables 17 and 18. `scripts/rows.py` is an earlier form of `gen_table18.py`.
5. `sec45.sh` (`hpcagent-bench/statistics/revision/`): the numbers of Section 4.5.
6. Figures: `data/fig34-50m/run.sh` (Figures 4 and 5), `scripts/fig-appendix-50m/run.sh` (Figures 6 and 8; it
   copied the scripts to the cluster and ran them there), `scripts/drawB3g.sh` (Figure 7), `scripts/aa-api/draw.py`
   (Figure 2; `AA_DATA` names the folder of `temperature3.db`, `AA_OUT` the PDF), and `scripts/aa4.sh` (the same A/A
   figure through `hpcagent-bench plot kernel-runs`).
7. `data/repeat3-promoted/r3.sh`: the run-to-run noise of Table 12 (`BENCH` names the checkout).

To check the files:

```sh
sha256sum --quiet -c SHA256SUMS && (cd data && sha256sum --quiet -c ../DATA_SHA256SUMS) && echo OK
```
