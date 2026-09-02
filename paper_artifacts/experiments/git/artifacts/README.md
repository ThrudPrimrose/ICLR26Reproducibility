# git-scicomp: repo framing vs kernel framing

## Question

Same kernel, same model, same judge. Only the FRAMING changes. Does handing an agent a git repo
with an `ISSUE.md` make it do better or worse than handing it a bare kernel name?

## Arms

Four arms. Two models x two framings. Run root `hpcagent-bench-runs/git-scicomp-20260901/`.

| folder | arm | model | framing | job |
|---|---|---|---|---|
| `oss120b_kernel/` | git-scicomp-oss120b-kernel | openai/gpt-oss-120b | kernel | 617395 |
| `oss120b_repo/`   | git-scicomp-oss120b-repo   | openai/gpt-oss-120b | repo   | 617396 |
| `qwen38_kernel/`  | git-scicomp-qwen38-kernel  | Qwen/Qwen3.8-27B-FP8 | kernel | 617397 |
| `qwen38_repo/`    | git-scicomp-qwen38-repo    | Qwen/Qwen3.8-27B-FP8 | repo   | 617398 |

The arms differ in FOUR env lines and nothing else (plus the arm's own name):

    REPO_LAYOUT=1
    REPO_LAYOUT_PYTHON=<venv python>
    REPO_LAYOUT_LANGUAGE=c
    AGENT_PROMPT_FILE=prompt-repo.md

`prompt-repo.md` is `prompt.md` with `containers/agent/repo-workflow.md` spliced in, so the arms
cannot drift in anything but the framing.

Jobs 613303-613306 (launched 08-30) left NO run directory and NO database rows anywhere under
`hpcagent-bench-runs/`. Everything here is the 09-01 re-run. Do not cite 6133xx as a data source.

## Kernels

Ten, three independent agents each, so 30 cells per arm and 120 rows total.

    gesummv  covariance  trisolv  ludcmp  jacobi_2d
    heat_3d  laplacian_stencil_3d  cg  spgemm_hash  mandelbrot1

`cg` is level 3 (sparse_linear_algebra); the other nine are level 2. Both arms run the same list,
so the A/B is unaffected -- but NEVER describe this set as "10 level-2 kernels".

CAVEAT on the `level` column: every manifest under `hpcagent_bench/benchmarks/` currently reads
`level: 2`, `cg` included, which contradicts the campaign kernel list. The column reports what the
manifest says. The manifests are correctness oracles and were not touched.

## Layout

    experiments/git/
      collect_git.py                     the script that produced everything here
      plot_git.py                        figures/git_framing.{png,pdf}
      data/git_experiment_all.csv        all 120 rows, all arms
      data/git_experiment_summary.csv    one row per arm
      artifacts/README.md                this file
      artifacts/<arm>/results.csv        that arm's 30 rows
      artifacts/<arm>/sources/
        <kernel>__a<N>__original.<ext>   what the agent started from
        <kernel>__a<N>__submitted.<ext>  what it ended with

Arm, kernel and attempt are readable from the path alone.

## Which text is saved

ORIGINAL depends on the framing, because the two framings hand over different files:

- repo arm: `repo/src/<kernel>.c`, read from the seed COMMIT, never the working tree. Agents wrote
  into the shared task templates -- in `oss120b_repo` the `spgemm_hash` and `mandelbrot1` templates
  are left with modified sources -- so the checked-out file is not reliably the starting text.
- kernel arm: `<kernel>_reference.*`. The suffix VARIES: `.c` for six kernels, `.py` for
  mandelbrot1, `.cu` for spgemm_hash. `cg` and `laplacian_stencil_3d` ship NO reference in either
  kernel arm, so their 6 cells per arm are `missing`. A bare `cg.c` does sit in the 617395 cg task
  dir, but it was written an hour after provisioning, by an agent -- it is not a served starting
  point and is deliberately not reported as one.

SUBMITTED, per the aggregation rule: several submissions -> the LAST one; no submission -> the last
kernel state, which is what `submit` would have taken. `submitted_provenance` says which:

    graded_submission  the exact text of the LAST graded submission
    last_graded        the last text the judge graded on any route
    last_saved         the last file in the agent workspace, NOT necessarily what was submitted
    missing            nothing recoverable

`last_saved` matches the canonical basename `<kernel>.c` only. That is the harness contract -- the
prompt says the submitted basename must be exactly `<kernel>.<ext>` and tells agents to "park
backups under other names" -- so a workspace with no `<kernel>.c` genuinely had nothing to submit.

## Re-running

    source /capstor/scratch/cscs/ybudanaz/x86_64/dace-env.sh
    python3 collect_git.py

Defaults resolve from the script's own location; override with `--runs`, `--problems`,
`--benchmarks`, `--out`. Databases are opened read-only. Output is deterministic: two runs over
unchanged inputs produce byte-identical files.

## Columns

Identity: `experiment` `arm` `arm_dir` `model` `framing` `kernel` `level` `dwarf` `attempt`
(0-2) `worker_index` `problem_index` `run_id` `job` `run_root` `db_paths`.

`attribution` is always `run_id`: every row is tied to a cell through the launcher's
`<arm>.n0.p<N>.w<N>` stamp. The `experiment` COLUMN in the database agrees, but it was added by
ALTER on connect, so rows written before it read NULL; `run_id` is what the cells are built on.

Outcome, taken from the LAST terminal submit for that cell:

- `verdict` -- `ok`, a rejection reason (`incorrect`, `too_slow`, `score_error`,
  `nondeterministic-or-public-mismatch`), `no_submission` (rows exist, never submitted) or
  `no_data` (the cell reached the judge zero times).
- `correct` 1/0, empty when nothing was submitted. `submitted` 1/0. `reason` the rejection text.
- `speedup` `baseline_ns` `native_ns` `runtime_s` `baseline_s` `suspect` -- only on an ACCEPTED
  submission. `runtime_s` is `native_ns / 1e9`.
- `best_score_speedup` -- best correct `score`-route grade during the run. Iteration telemetry,
  NOT a result: `score` grades on the public seed only, `submit` re-checks on a held-out one.

Cost: `n_calls` `n_submit_calls` `n_submissions` `n_failed_submits` `turns` `tokens`
`ts_first_ms` `ts_last_ms` `ts_first_iso` `ts_last_iso` `wall_clock_s`. `tokens` is the per-cell
HIGH WATER, not a sum -- the judge records cumulative usage per call, so adding calls double-counts.

`off_task_benchmarks` -- benchmarks graded under this cell's run_id that are NOT its kernel. One
occurrence: `oss120b_kernel` mandelbrot1 attempt 1 also scored `arc_distance`. Those rows are
excluded from the cell's metrics.

Provenance: `original_path` `original_sha256` `original_provenance` `original_origin`
`submitted_path` `submitted_sha256` `submitted_provenance`. Paths are relative to this folder;
`original_origin` and `db_paths` are absolute pointers back into the run root.

## Reading the numbers

Aggregate speedup is GEOMEAN, never median. Coverage is thin -- 40 of 120 cells reached the judge
and 9 produced an accepted submission -- so treat per-arm speedup as descriptive, not as a test.

`qwen38_repo`'s geomean is carried by one cell: `laplacian_stencil_3d` attempt 2 at 104.3x, 5.3 ms
against a 573 ms baseline. It passed the held-out seed and `suspect` is 0, but a 100x on a stencil
is the shape of eliminated work. Without it the arm's geomean is 2.34x, in line with every other
arm. Report both or verify that cell before citing it.

A missing kernel is a FINDING, not a dropped row: every cell is written out, including `no_data`.
`ludcmp` reached the judge in NONE of the four arms.

## Contamination found

Agents can write the shared tree, and did. Recorded here so nobody trusts the run root blindly:

- `oss120b_repo` task templates `spgemm_hash` and `mandelbrot1` are left with MODIFIED
  `src/*.c` in the working tree. The extractor reads seeds from the commit, so the CSVs are clean.
- `oss120b_kernel` task dir gained an agent's `cg/cg.c`, `cg/cg.o`, and a rewritten
  `gesummv/gesummv.py`, all after provisioning.
- `oss120b_kernel` mandelbrot1 attempt 1 scored `arc_distance`, a kernel outside this experiment.

Verified clean: all ten repo seeds are byte-identical between the two repo arms.
