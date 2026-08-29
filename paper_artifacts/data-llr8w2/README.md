# data-llr8w2 -- C vs Fortran, skills on/off

Collected 2026-08-29 from `$SCRATCH/hpcagent-bench-runs/llr8w1-20260827`, after every arm landed.\n\nDenominator is **40 kernels per arm**: the problems file holds 120, but the launcher dispatches 40\n(highest problem index in every wave-2 run_id is `p39`; the two oss120b arms dispatched all 40).

ONE command fuses the rank shards, writes every CSV and renders the figure:

    python3 collect.py --run-root $SCRATCH/hpcagent-bench-runs/llr8w1-20260827 \
                       --campaign llr8w2 --out data-llr8w2 --plot

Add `--discover` to read the arms out of the run directories instead of `collect.py`'s registry --
useful for a directory you have not registered yet. It reports one row per ARM, keeping the job with
the most calls when a failed arm was resubmitted, and names every job it drops. The registry stays
authoritative for a paper number, because only it can distinguish "this arm ran and returned
nothing" from "this arm was never in the campaign", and only it separates waves that share a run
directory.

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

## Optimiser label -- a label defect, now fixed at the source

`.env.llr8-qwen38-c` and its skilled twin set `OPTARENA_OPTIMIZER=openai/gpt-oss-120b` while serving
`VLLM_MODEL=Qwen/Qwen3.8-27B-FP8`, so the `optimizer` column of those two shards names the wrong
model. The Slurm logs for 610669 and 610670 both show `Qwen3.8-27B-FP8` served under
`served_model_name='optarena-vllm'`, so the served model was the same in all four qwen arms and only
the recorded string was wrong -- the measurements stand. Both env files now say
`OPTARENA_OPTIMIZER=Qwen/Qwen3.8-27B-FP8`. The CSVs here take `model` from the arm registry and were
never affected; the raw `optimizer` column inside the archived shards still carries the old string
and is left alone, being the immutable record of what ran.

## C is not better than Fortran -- coverage is

The per-arm geomeans invite the reading that this model optimises C far better than Fortran
(`qwen38-c` 8.66 against `qwen38-fortran` 3.59). On the 26 kernels BOTH languages solved the two are
equivalent: C 11.24, Fortran 10.38, and C wins 13 of the 26. Reference baselines are within 1%
(median ratio 1.007), so the languages are not being scored against differently-tuned references.

What differs is throughput. The unskilled Fortran arm made 142 calls to C's 212 and 35 submissions
to C's 69, over a similar number of kernels reached (31 vs 29) -- roughly 4.6 turns per kernel
against 7.3. Build errors were 3 in both, so this is iteration count, not a Fortran-specific failure.
C therefore solved 8 kernels Fortran never reached (geomean 12.5, including a 127x) while Fortran
solved 4 that C did not (geomean 6.1), and that difference in WHICH kernels landed is the whole
apparent language gap.

## The 150x kernels

`tsvc_2_s1232` returns a median 153x across six independent arms and `tsvc_2_s2275` 127x, so neither
is a noisy measurement. Both are legitimate: `s1232` walks `aa[i,j]` with `i` innermost over a
row-major array, which is a stride-N access on a 12724^2 matrix, and the accepted answers interchange
the loops to make the inner index contiguous. That is the transformation the TSVC kernel exists to
test, and 150x is what removing a cache miss per element on 24 cores is worth.

They do distort the LEVEL: dropping both moves the per-arm geomeans by -11% to -20%. They do not
distort the EFFECT, because a paired ratio divides them out -- the skills ratios move only
0.919 -> 0.908, 1.702 -> 1.778 and 1.095 -> 0.985. So the recommendation is to keep the kernels and
quote the paired ratio, not to trim the corpus: a benchmark that excludes its highest-headroom
kernels stops measuring the transformation it was built to measure.

## Foreign rows

Seven `run_id='adhoc'` submissions sit in the wave-2 shards, one of them a 24x outlier, from a
hand-run probe that shared the results directory. `collect.py` now selects rows by the arm's own
`run_id` pattern, so they are excluded; a collection made before that fix credited them to whichever
arm's job wrote the file.
